"""
Flask backend for University Classifier ML Dashboard.
Endpoints:
  GET  /                → serves index.html
  POST /train           → trains all models, returns scores JSON
  POST /test            → evaluates ensemble, returns report + CM JSON
  GET  /stream/<job_id> → SSE stream of stdout lines during a job
"""

from flask import Flask, render_template, jsonify, request, Response
import threading
import sys
import queue
import json
import uuid
import os
import re
import traceback

app = Flask(__name__)

# ── global state ────────────────────────────────────────────────────────────
_state = {
    "model":    None,
    "test_x":   None,
    "test_y":   None,
    "busy":     False,
}
_streams: dict[str, queue.Queue] = {}   # job_id → Queue of log lines


# ══════════════════════════════════════════════════════════════════════════
# stdout capture
# ══════════════════════════════════════════════════════════════════════════
class StreamCapture:
    def __init__(self, q: queue.Queue, original=None):
        self.q, self.original, self.buf = q, original, ""

    def write(self, text):
        if self.original:
            self.original.write(text)
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.q.put(line)

    def flush(self):
        if self.original:
            self.original.flush()
        if self.buf.strip():
            self.q.put(self.buf)
            self.buf = ""


# ══════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train", methods=["POST"])
def train():
    if _state["busy"]:
        return jsonify({"error": "busy"}), 409

    job_id = str(uuid.uuid4())
    q = queue.Queue()
    _streams[job_id] = q

    def worker():
        _state["busy"] = True
        orig = sys.stdout
        sys.stdout = StreamCapture(q, orig)
        score_map = {}
        try:
            from sklearn.neighbors    import KNeighborsClassifier
            from sklearn.tree         import DecisionTreeClassifier
            from sklearn.ensemble     import (RandomForestClassifier,
                                              GradientBoostingClassifier,
                                              AdaBoostClassifier,
                                              VotingClassifier)
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline     import Pipeline
            from data import Data

            d = Data()
            train_x, test_x, train_y, test_y = d.feature_selection()

            def fit(tag, est):
                pipe = Pipeline([(tag, est)])
                pipe.fit(train_x, train_y)
                s = pipe.score(test_x, test_y)
                score_map[tag] = round(s, 4)
                print(f"{tag.upper()} Score: {s:.4f}")
                return pipe

            knn = fit("knn", KNeighborsClassifier(n_neighbors=20, weights='distance', algorithm='brute'))
            dtc = fit("dt",  DecisionTreeClassifier(criterion='log_loss', splitter='best', max_depth=10))
            rfc = fit("rf",  RandomForestClassifier(n_estimators=200, criterion='log_loss', max_features=6, class_weight='balanced'))
            lrc = fit("lr",  LogisticRegression(C=0.5, solver='sag', max_iter=1000))
            gbc = fit("gb",  GradientBoostingClassifier(n_estimators=300, learning_rate=0.05))
            ada = fit("ada", AdaBoostClassifier(n_estimators=300, learning_rate=0.05))

            ensemble = VotingClassifier(
                [('KNN', knn), ('DTC', dtc), ('RFC', rfc),
                 ('LRC', lrc), ('GBC', gbc), ('ADA', ada)],
                voting='soft')
            ensemble.fit(train_x, train_y)
            ens = ensemble.score(test_x, test_y)
            score_map["ensemble"] = round(ens, 4)
            print(f"Ensemble Score: {ens:.4f}")

            _state["model"]  = ensemble
            _state["test_x"] = test_x
            _state["test_y"] = test_y

            # signal done with scores payload
            q.put("__DONE__:" + json.dumps({"scores": score_map}))

        except Exception:
            q.put("__ERROR__:" + traceback.format_exc())
        finally:
            sys.stdout = orig
            _state["busy"] = False

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/test", methods=["POST"])
def test():
    if _state["busy"]:
        return jsonify({"error": "busy"}), 409
    if _state["model"] is None:
        return jsonify({"error": "not_trained"}), 400

    job_id = str(uuid.uuid4())
    q = queue.Queue()
    _streams[job_id] = q

    def worker():
        _state["busy"] = True
        orig = sys.stdout
        sys.stdout = StreamCapture(q, orig)
        try:
            from sklearn.metrics import classification_report, confusion_matrix
            import numpy as np

            model  = _state["model"]
            test_x = _state["test_x"]
            test_y = _state["test_y"]

            preds  = model.predict(test_x)
            report = classification_report(test_y, preds, output_dict=False)
            report_dict = classification_report(test_y, preds, output_dict=True)
            cm     = confusion_matrix(test_y, preds).tolist()
            labels = [str(l) for l in sorted(test_y.unique())]

            print(report)

            payload = {
                "report_text": report,
                "report_dict": report_dict,
                "cm": cm,
                "labels": labels,
            }
            q.put("__DONE__:" + json.dumps(payload))

        except Exception:
            q.put("__ERROR__:" + traceback.format_exc())
        finally:
            sys.stdout = orig
            _state["busy"] = False

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream(job_id):
    q = _streams.get(job_id)
    if q is None:
        return "not found", 404

    def generate():
        while True:
            try:
                line = q.get(timeout=120)
                yield f"data: {json.dumps(line)}\n\n"
                if line.startswith("__DONE__") or line.startswith("__ERROR__"):
                    _streams.pop(job_id, None)
                    break
            except queue.Empty:
                yield "data: __TIMEOUT__\n\n"
                break

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, threaded=True, host="0.0.0.0", port=port)