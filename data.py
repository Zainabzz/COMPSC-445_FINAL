# Data Collection, Preprocessing, and Merging
# Using multiple datasets, creating a merged dataset of university data across PA
# Goal of project: classify best university for a user based on inputs
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer  # Explicitly enable
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

class Data:
    def collect(self):
        higherEdBySector = pd.read_csv('data/HigherEd_BySector.csv')
        higherEdBySector.name = "HigherEducationBySector"

        higherEdInstitutions = pd.read_csv('data/HigherEd_Insitutions.csv')
        higherEdInstitutions = higherEdInstitutions[higherEdInstitutions["County"] != "Out-of-State"]
        higherEdInstitutions.name = "HigherEducationInstitutions"
        #print(higherEdInstitutions['Category'].unique())

        cost = pd.read_csv('data/Data_4-19-2026---701.csv')
        cost.name = "Cost"

        """ PASSHE Data - unused
        totalHeadcount = pd.read_csv('data/passhe_data/Total_Headcount.csv', encoding='utf-16', sep='\t')
        totalHeadcount.name = "TotalHeadcount"

        fullTime = pd.read_csv('data/passhe_data/FullTime.csv', encoding='utf-16', sep='\t')
        fullTime.name = "FullTime"

        livingOnCampus = pd.read_csv('data/passhe_data/LivingOnCampus.csv', encoding='utf-16', sep='\t')
        livingOnCampus.name = "LivingOnCampus"

        adultLearner = pd.read_csv('data/passhe_data/AdultLearner.csv', encoding='utf-16', sep='\t')
        adultLearner.name = "AdultLearner"

        educationMajor = pd.read_csv('data/passhe_data/EducationMajor.csv', encoding='utf-16', sep='\t')
        educationMajor.name = "EducationMajor"

        minority = pd.read_csv('data/passhe_data/Minority.csv', encoding='utf-16', sep='\t')
        minority.name = "Minority"

        stem_hMajor = pd.read_csv('data/passhe_data/STEM-H_Major.csv', encoding='utf-16', sep='\t')
        stem_hMajor.name = "STEMHMajor"

        transferStudent = pd.read_csv('data/passhe_data/TransferStudent.csv', encoding='utf-16', sep='\t')
        transferStudent.name = "TransferStudent"
        """

        data = [higherEdBySector,
                higherEdInstitutions,
                cost]
        print(higherEdBySector.head())
        print(higherEdInstitutions.head())
        print(cost.head())
        return data

    def merge(self):
        data = self.collect()
        for frame in data:
            new_cols = list(frame.columns)
            for i in range(len(new_cols)):
                new_cols[i] = new_cols[i] + f" ({frame.name})"
            #print(new_cols)
            frame.rename(columns=dict(zip(frame.columns, new_cols)), inplace=True)
        # Column merging - insitution name
        data[0] = data[0].rename(columns={"College (HigherEducationBySector)" : "Institution"})
        data[1] = data[1].rename(columns={"Institution Name (HigherEducationInstitutions)" : "Institution"})
        data[2] = data[2].rename(columns={"Institution Name (Cost)" : "Institution"})

        data[0] = data[0].rename(columns={"County (HigherEducationBySector)": "County"})
        data[1] = data[1].rename(columns={"County (HigherEducationInstitutions)": "County"})

        data[0] = data[0].rename(columns={"Sector (HigherEducationBySector)": "Category"})
        data[1] = data[1].rename(columns={"Category (HigherEducationInstitutions)": "Category"})

        # Column dropping
        data[0].drop(columns=['Georeferenced Latitude & Longitude (HigherEducationBySector)'], inplace=True)
        data[1].drop(columns=["Administrative Unit Number (HigherEducationInstitutions)",
                              "School Branch (HigherEducationInstitutions)",
                              "Status (HigherEducationInstitutions)",
                              "Phone Number (HigherEducationInstitutions)",
                              "Phone_Number_Extn (HigherEducationInstitutions)",
                              "Web_Address (HigherEducationInstitutions)",
                              "Primary Administrator (HigherEducationInstitutions)",
                              "School Year (HigherEducationInstitutions)"], inplace=True)
        data[2].drop(columns=["UnitID (Cost)",
                              "Unnamed: 9 (Cost)"], inplace=True)


        # Merge
        dataset = pd.concat(data, axis=0)
        #print(dataset['Category'].unique())
        #print(dataset['Category'].unique())

        # Category filter - Dropped technical programs, admin offices, filtering to colleges & unis
        categories = ['Private College/University',
                                'Community College',
                           'Rural Regional College',
            'State-Related Commonwealth University',
                   'Private College and University',
                     'Other College and University',
                                 'State University',
                  'Private State-Aided Institution',
                         'Private Two-Year College',
                             'Theological Seminary',
                             'Other Post Secondary']
        dataset = dataset[dataset['Category'].isin(categories)]

        return dataset

    def distance(self, point, city, axis):
        PHILLY = (39.9526, -75.16462)
        PITTSBURGH = (40.4387, -79.9972)
        ALLEN = (40.6023, -75.4714)
        READING = (40.337, -75.9214)
        ERIE = (42.129, -80.085)

        if city == "PHILLY" and axis == 'lat':
            return point - PHILLY[0]
        elif city == "PHILLY" and axis == 'lon':
            return point - PHILLY[1]
        elif city == "PITTSBURGH" and axis == 'lat':
            return point - PITTSBURGH[0]
        elif city == "PITTSBURGH" and axis == 'lon':
            return point - PITTSBURGH[1]
        elif city == "ALLEN" and axis == 'lat':
            return point - ALLEN[0]
        elif city == "ALLEN" and axis == 'lon':
            return point - ALLEN[1]
        elif city == "READING" and axis == 'lat':
            return point - READING[0]
        elif city == "READING" and axis == 'lon':
            return point - READING[1]
        elif city == "ERIE" and axis == 'lat':
            return point - ERIE[0]
        elif city == "ERIE" and axis == 'lon':
            return point - ERIE[1]

    def featureEng(self):
        dataset = self.merge()

        dataset['Distance to Philly (Latitude)'] = dataset["Latitude (HigherEducationBySector)"].apply(self.distance, args=("PHILLY", "lat"))
        dataset['Distance to Philly (Longitude)'] = dataset["Longitude (HigherEducationBySector)"].apply(self.distance, args=("PHILLY", "lon"))

        dataset['Distance to Pittsburgh (Latitude)'] = dataset["Latitude (HigherEducationBySector)"].apply(self.distance,args=("PITTSBURGH", "lat"))
        dataset['Distance to Pittsburgh (Longitude)'] = dataset["Longitude (HigherEducationBySector)"].apply(self.distance,args=("PITTSBURGH", "lon"))

        dataset['Distance to Allentown (Latitude)'] = dataset["Latitude (HigherEducationBySector)"].apply(self.distance, args=("ALLEN", "lat"))
        dataset['Distance to Allentown (Longitude)'] = dataset["Longitude (HigherEducationBySector)"].apply(self.distance, args=("ALLEN", "lon"))

        dataset['Distance to Reading (Latitude)'] = dataset["Latitude (HigherEducationBySector)"].apply(self.distance,args=("READING", "lat"))
        dataset['Distance to Reading (Longitude)'] = dataset["Longitude (HigherEducationBySector)"].apply(self.distance, args=("READING", "lon"))

        dataset['Distance to Erie (Latitude)'] = dataset["Latitude (HigherEducationBySector)"].apply(self.distance,args=("ERIE","lat"))
        dataset['Distance to Erie (Longitude)'] = dataset["Longitude (HigherEducationBySector)"].apply(self.distance, args=("ERIE", "lon"))

        # County region - north, west, east, south PA vs county
        #print(dataset['County'].unique()) # Looked through all the 57 counties, mnaully divided them into sectors, created binary cols for region of institution
        NORTH = ['Luzerne', 'McKean', 'Lackawanna', 'Venango', 'Bradford', 'Crawford', 'Forest', 'Erie', 'Elk', 'Warren', 'Lycoming', 'Mercer', 'Jefferson', 'Potter', 'Cameron', 'Clarion', 'Clinton', 'Tioga', 'Wayne']
        EAST = ['Luzerne', 'Montgomery', 'York', 'Dauphin', 'Lehigh', 'Schuylkill', 'Berks', 'Lackawanna', 'Delaware', 'Bucks', 'Northampton', 'Chester', 'Bradford', 'Philadelphia', 'Lebanon', 'Lancaster', 'Northumberland', 'Lycoming', 'Columbia', 'Monroe', 'Tioga', 'Wayne']
        SOUTH = ['Franklin', 'York', 'Montgomery', 'Berks', 'Delaware', 'Bucks', 'Cambria', 'Allegheny', 'Washington', 'Chester', 'Somerset', 'Westmoreland', 'Philadelphia', 'Lebanon', 'Cumberland', 'Greene', 'Lancaster', 'Fayette', 'Adams', 'Blair', 'Huntingdon', 'Juniata']
        WEST = ['Lawrence', 'McKean', 'Venango', 'Cambria', 'Allegheny', 'Beaver', 'Washington', 'Somerset', 'Westmoreland', 'Crawford', 'Forest', 'Erie', 'Elk', 'Greene', 'Fayette', 'Indiana', 'Butler', 'Warren', 'Mercer', 'Jefferson', 'Clearfield', 'Blair', 'Armstrong', 'Clarion']
        CENTRAL = ['Franklin', 'Lawrence', 'Dauphin', 'Lehigh', 'Schuylkill', 'Cambria', 'Elk', 'Lebanon', 'Cumberland', 'Northumberland', 'Indiana', 'Butler', 'Lycoming', 'Jefferson', 'Clearfield', 'Centre', 'Union', 'Adams', 'Blair', 'Snyder', 'Potter', 'Armstrong', 'Huntingdon', 'Columbia', 'Monroe', 'Cameron', 'Clinton', 'Tioga', 'Juniata']

        dataset['North'] = dataset['County'].apply(lambda x: 1 if x in NORTH else 0)
        dataset['East'] = dataset['County'].apply(lambda x: 1 if x in EAST else 0)
        dataset['South'] = dataset['County'].apply(lambda x: 1 if x in SOUTH else 0)
        dataset['West'] = dataset['County'].apply(lambda x: 1 if x in WEST else 0)
        dataset['Central'] = dataset['County'].apply(lambda x: 1 if x in CENTRAL else 0)

        return dataset

    def impute_encode(self):
        dataset = self.featureEng()
        num_imputer = IterativeImputer(max_iter=15)
        cat_imputer = SimpleImputer(strategy="constant", fill_value="Other")
        #print(dataset.select_dtypes(include='number').columns.tolist())
        #print(dataset.select_dtypes(include=['object', 'category']).columns.tolist())

        preprocessor = ColumnTransformer([
            ("num_imputer", num_imputer, ['zip code (HigherEducationBySector)',
                                          'Latitude (HigherEducationBySector)',
                                          'Longitude (HigherEducationBySector)',
                                          'Total price for in-district students living on campus  2024-25 (DRVCOST2024) (Cost)',
                                          'Total price for in-state students living on campus 2024-25 (DRVCOST2024) (Cost)',
                                          'Total price for in-district students living off campus (not with family)  2024-25 (DRVCOST2024) (Cost)',
                                          'Total price for in-state students living off campus (not with family)  2024-25 (DRVCOST2024) (Cost)',
                                          'Total price for in-district students living off campus (with family)  2024-25 (DRVCOST2024) (Cost)',
                                          'Total price for in-state students living off campus (with family)  2024-25 (DRVCOST2024) (Cost)',
                                          'Distance to Philly (Latitude)',
                                          'Distance to Philly (Longitude)',
                                          'Distance to Pittsburgh (Latitude)',
                                          'Distance to Pittsburgh (Longitude)',
                                          'Distance to Allentown (Latitude)',
                                          'Distance to Allentown (Longitude)',
                                          'Distance to Reading (Latitude)',
                                          'Distance to Reading (Longitude)',
                                          'Distance to Erie (Latitude)',
                                          'Distance to Erie (Longitude)',
                                          'North',
                                          'East',
                                          'South',
                                          'West',
                                          'Central']),
            ("cat_imputer", cat_imputer, ['Institution',
                                          'Type (HigherEducationBySector)',
                                          'County',
                                          'Address (HigherEducationBySector)',
                                          'Sector and Type (HigherEducationBySector)',
                                          'City (HigherEducationInstitutions)',
                                          'State abbreviation (HD2024) (Cost)'])
        ])

        #print(dataset.columns)
        dataset.to_csv("data.csv", index=False)

        x = dataset.drop(columns=['Category'], inplace=False)
        y = dataset['Category']
        #print(x.columns)
        #print(y.columns)
        preprocessor.set_output(transform="pandas")
        x_imputed = preprocessor.fit_transform(x.astype(object)) # Google search AI Overview
        x_imputed.to_csv("x_imputed.csv", index=False)

        le = LabelEncoder()
        cols = x_imputed.select_dtypes(include=['object', 'category']).columns
        for col in cols:
            x_imputed[col] = le.fit_transform(x_imputed[col])

        train_x, test_x, train_y, test_y = train_test_split(x_imputed, y, test_size=0.10, random_state=42)
        return train_x, test_x, train_y, test_y

    def scale(self):
        train_x, test_x, train_y, test_y = self.impute_encode()

        ss = StandardScaler()
        train_x = ss.fit_transform(train_x)
        test_x = ss.transform(test_x)
        #print(train_x)

        return train_x, test_x, train_y, test_y

    def feature_selection(self):
        train_x, test_x, train_y, test_y = self.scale()
        selector = SelectKBest(k=15)
        selector.fit(train_x, train_y)
        print("Selector scores: ", selector.scores_)
        print("Selector top 15: ", selector.get_feature_names_out())
        selector.set_output(transform="pandas")
        train_x = selector.transform(train_x)
        test_x = selector.transform(test_x)

        print("Train X: ", train_x)
        print("Train X shape: ", train_x.shape)
        print("Test X: ", test_x)
        print("Test X shape: ", test_x.shape)
        print("Train Y: ", train_y)
        print("Train Y shape: ", train_y.shape)
        print("Test Y: ", test_y)
        print("Test Y shape: ", test_y.shape)

        return train_x, test_x, train_y, test_y

    #
    def nanCheck(self):
        data = self.merge()
        for col in data:
            print(f"Total NaN in {col}: {data[col].isna().sum()}")



#data = Data()
#data.collect()
#data.merge()
#data.clean_split()
#data = data.encode_scale()
#data.nanCheck()