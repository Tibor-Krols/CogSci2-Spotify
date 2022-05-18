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

class LyricsTable:

    def __init__(self, regressor):
        self.regressor = regressor

        self.features_train = np.loadtxt('lyrics_features_train.csv', delimiter = ',')
        self.features_test = np.loadtxt('lyrics_features_test.csv', delimiter = ',')
        self.labels_train_v = pd.read_csv('merged_cleaned_sentiment_train.csv', delimiter = ',')['y_valence']
        self.labels_train_a = pd.read_csv('merged_cleaned_sentiment_train.csv', delimiter = ',')['y_arousal']
        self.labels_test_v = pd.read_csv('merged_cleaned_sentiment_test.csv', delimiter = ',')['y_valence']
        self.labels_test_a = pd.read_csv('merged_cleaned_sentiment_test.csv', delimiter = ',')['y_arousal']
        self.combos = ['tfidf', 'anew', 'vader', 'tfidf+anew', 'tfidf+vader', 'anew+vader', 'tfidf+anew+vader']
        self.results = dict()

        for combo in self.combos:
            features_train = self.get_features(combo, self.features_train)
            features_test = self.get_features(combo, self.features_test)

            self.results[combo] = self.regression_r2(features_train, features_test)
        
        self.generate_table(self.results)

    def init_regr(self):
        if self.regressor == 'linreg':
            return LinearRegression()
        elif self.regressor == 'rf':
            return RandomForestRegressor(max_depth = 10)
        elif self.regressor == 'mlp':
            return MLPRegressor(max_iter= 500)

    def get_features(self, features, data):

        if len(features) > 5:
            features = features.split('+')
        else:
            features = [features]

        index_dict = {'vader': (0, 5), 'tfidf': (5,156), 'anew': (156, 256)}

        res_array = np.zeros((data.shape[0]))

        if len(features) == 3:
            return data
        else:
            for feature in features: 
                index = index_dict[feature]
                feature_array = data[index[0]:index[1]]
                res_array = np.concatenate((res_array, feature_array), axis = 1)

        return res_array[:, 1:]

    def regression_r2(self, train, test):
        regr_val = self.init_regr()
        self.regr_val.fit(train, self.labels_train_v)
        prediction_val = self.regr.predict(test)
        r2_val = r2_score(self.labels_test_v, prediction_val)

        regr_ar = self.init_regr()
        self.regr_ar.fit(train, self.labels_train_ar)
        prediction_ar = self.regr.predict(test)
        r2_ar = r2_score(self.labels_test_ar, prediction_ar)

        return (r2_val, r2_ar)

    def generate_table(self, results):
        print(f"Model: {self.regressor}\t\t\t\t R^2")
        print("Features\t\tValence\tArousal")
        for key in results.keys():
            print(f"{key}\t\t{results[key][0]}\{results[key][1]}")



    

    
