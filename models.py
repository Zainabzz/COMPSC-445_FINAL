# Developement and train of multiple classfiication models into pipelines and merge into ensemble model
from matplotlib import pyplot as plt

from data import Data

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

class Models:
    def __init__(self):
        self.d = Data()
        self.train_x, self.test_x, self.train_y, self.test_y = self.d.feature_selection()
        print(self.test_y.shape)
        print(self.train_y.shape)

    def KNN(self):
        model = KNeighborsClassifier(n_neighbors=20, weights='distance', algorithm='brute')
        pipe = Pipeline([('knn', model)])
        pipe.fit(self.train_x, self.train_y)
        score = pipe.score(self.test_x, self.test_y)
        return pipe, score

    def DTC(self):
        model = DecisionTreeClassifier(criterion='log_loss', splitter='best', max_depth=10)
        pipe = Pipeline([('dt', model)])
        pipe.fit(self.train_x, self.train_y)
        score = pipe.score(self.test_x, self.test_y)
        return pipe, score

    def RFC(self):
        model = RandomForestClassifier(n_estimators=200, criterion='log_loss', max_features=6, class_weight='balanced')
        pipe = Pipeline([('rf', model)])
        pipe.fit(self.train_x, self.train_y)
        score = pipe.score(self.test_x, self.test_y)
        return pipe, score

    def LRC(self):
        model = LogisticRegression(C=0.5, solver='sag', max_iter=1000)
        pipe = Pipeline([('lr', model)])
        pipe.fit(self.train_x, self.train_y)
        score = pipe.score(self.test_x, self.test_y)
        return pipe, score

    def GBC(self):
        model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05)
        pipe = Pipeline([('gb', model)])
        pipe.fit(self.train_x, self.train_y)
        score = pipe.score(self.test_x, self.test_y)
        return pipe, score

    def ADA(self):
        model = AdaBoostClassifier(n_estimators=300, learning_rate=0.05)
        pipe = Pipeline([('ada', model)])
        pipe.fit(self.train_x, self.train_y)
        score = pipe.score(self.test_x, self.test_y)
        return pipe, score

    def SVC(self):
        model = SVC(class_weight='balanced')
        pipe = Pipeline([('svc', model)])
        pipe.fit(self.train_x, self.train_y)
        score = pipe.score(self.test_x, self.test_y)
        return pipe, score

    def gen(self):
        # Can change which models we add into the ensemble depending on the performance and data
        KNN, score1 = self.KNN()
        DTC, score2 = self.DTC()
        RFC, score3 = self.RFC()
        LRC, score4 = self.LRC()
        GBC, score5 = self.GBC()
        ADA, score6 = self.ADA()
        #SVC, score7 = self.SVC()
        print("KNN Score: ", score1)
        print("DTC Score: ", score2)
        print("RFC Score: ", score3)
        print("LRC Score: ", score4)
        print("GBC Score: ", score5)
        print("ADA Score: ", score6)
        #print("SVC Score: ", score7)
        ensembleModel = VotingClassifier([('KNN', KNN), ('DTC', DTC), ('RFC', RFC), ('LRC', LRC), ('GBC', GBC), ('ADA', ADA)], voting='soft')
        #ensembleModel2 = StackingClassifier(estimators=[('KNN', KNN), ('RFC', RFC), ('DTC', DTC)], final_estimator=LogisticRegression())
        return ensembleModel #, ensembleModel2

    def train(self):
        model = self.gen()
        #print(self.train_y.shape)
        model.fit(self.train_x, self.train_y)
        #model2.fit(self.train_x, self.train_y)
        preds = model.predict(self.test_x)
        #preds2 = model2.predict(self.test_x)
        print("Ensemble Score: ", model.score(self.test_x, self.test_y))
        #print("Test Score 2: ", model2.score(self.test_x, self.test_y))
        #print("Train Score: ", model.score(self.train_x, self.train_y))
        print(classification_report(self.test_y, preds))
        disp = ConfusionMatrixDisplay(confusion_matrix(self.test_y, preds))
        disp.plot()
        plt.show()
        #print(classification_report(self.test_y, preds2))
        # (0.5983791085096803, 0.3237410071942446) --> train & test .score on model originally, needs improvement

models = Models()
models.train()