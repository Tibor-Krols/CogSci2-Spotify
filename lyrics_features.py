import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR 
from sklearn.preprocessing import MinMaxScaler



class LyricsTable:

    def __init__(self, regressor, testset = None):
        self.regressor = regressor
        self.testset = testset

        self.features_train = pd.read_csv('lyrics_features_train.csv', delimiter = ',').values
        self.features_test = pd.read_csv('lyrics_features_test.csv', delimiter = ',').values
        self.features_val = pd.read_csv('lyrics_features_val.csv', delimiter = ',').values
        self.labels_train_v = pd.read_csv('merged_cleaned_sentiment_train.csv', delimiter = ',').dropna(axis = 0)['y_valence']
        self.labels_train_a = pd.read_csv('merged_cleaned_sentiment_train.csv', delimiter = ',').dropna(axis = 0)['y_arousal']
        self.labels_test_v = pd.read_csv('merged_cleaned_sentiment_test.csv', delimiter = ',').dropna(axis = 0)['y_valence']
        self.labels_test_a = pd.read_csv('merged_cleaned_sentiment_test.csv', delimiter = ',').dropna(axis = 0)['y_arousal']
        self.labels_val_v = pd.read_csv('merged_cleaned_sentiment_validation.csv', delimiter = ',').dropna(axis = 0)['y_valence']
        self.labels_val_a = pd.read_csv('merged_cleaned_sentiment_validation.csv', delimiter = ',').dropna(axis = 0)['y_arousal']

    def create_table(self):
            
        self.combos = ['tfidf', 'anew', 'vader', 'tfidf+anew', 'tfidf+vader', 'anew+vader', 'tfidf+anew+vader']

        self.results = dict()

        for combo in self.combos:
            features_train = self.get_features(combo, self.features_train)

            if self.testset == 'test':
                features_test = self.get_features(combo, self.features_test)
                self.results[combo] = self.regression_r2(features_train, features_test, self.labels_train_a, self.labels_train_v, self.labels_test_a, self.labels_test_v)

            elif self.testset == 'val':
                features_val = self.get_features(combo, self.features_val)
                self.results[combo] = self.regression_r2(features_train, features_val, self.labels_train_a, self.labels_train_v, self.labels_val_a, self.labels_val_v)

        self.generate_table(self.results)
        
    def init_regr(self):
        if self.regressor == 'linreg':
            return LinearRegression(positive = True, fit_intercept= True)
        elif self.regressor == 'rf':
            return RandomForestRegressor()
        elif self.regressor == 'mlp':
            scaler = MinMaxScaler()
            self.features_train = scaler.fit_transform(self.features_train)
            self.features_val = scaler.transform(self.features_val)
            self.features_test = scaler.transform(self.features_test)
            return MLPRegressor()
        elif self.regressor == 'svr':
            scaler = MinMaxScaler()
            self.features_train = scaler.fit_transform(self.features_train)
            self.features_val = scaler.transform(self.features_val)
            self.features_test = scaler.transform(self.features_test)
            return SVR()

    def get_features(self, features, data):

        if len(features) > 5:
            features = features.split('+')
        else:
            features = [features]

        index_dict = {'vader': (0, 5), 'tfidf': (5,156), 'anew': (156, 356)}

        res_array = np.zeros((data.shape[0], 2))

        if len(features) == 3:
            return data
        else:
            for feature in features: 
                index = index_dict[feature]
                feature_array = data[:, index[0]:index[1]]
                res_array = np.concatenate((res_array, feature_array), axis = 1)

        return res_array[:, 2:]

    def regression_r2(self, train, test, lab_train_a, lab_train_v, lab_a, lab_v):
        regr_val = self.init_regr()
        regr_val.fit(train, lab_train_v)
        self.prediction_val = regr_val.predict(test)
        r2_val = r2_score(lab_v, self.prediction_val)

        regr_ar = self.init_regr()
        regr_ar.fit(train, lab_train_a)
        self.prediction_ar = regr_ar.predict(test)
        r2_ar = r2_score(lab_a, self.prediction_ar)

        return (r2_val, r2_ar)

    def generate_table(self, results):
        print(f"Model: {self.regressor}\t\t\t\t R^2")
        print("Features\t\tValence\tArousal")
        for key in results.keys():
            print(f"{key}\t\t{results[key][0]}\t{results[key][1]}")

l = LyricsTable(regressor='svr', testset='val')
l.create_table()

    

