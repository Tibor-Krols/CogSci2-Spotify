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
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

class LyricsTable:

    def __init__(self, regressor, testset, params):
        self.regressor = regressor
        self.testset = testset
        self.params = params 

        self.features_train = (pd.read_csv('lyrics_features_train.csv', delimiter = ',').to_numpy()).astype(np.float32)
        self.features_test = (pd.read_csv('lyrics_features_test.csv', delimiter = ',').to_numpy()).astype(np.float32)
        self.features_val = (pd.read_csv('lyrics_features_val.csv', delimiter = ',').to_numpy()).astype(np.float32)
        self.labels_train_v = pd.read_csv('merged_cleaned_sentiment_train.csv', delimiter = ',')['y_valence']
        self.labels_train_a = pd.read_csv('merged_cleaned_sentiment_train.csv', delimiter = ',')['y_arousal']
        self.labels_test_v = pd.read_csv('merged_cleaned_sentiment_test.csv', delimiter = ',')['y_valence']
        self.labels_test_a = pd.read_csv('merged_cleaned_sentiment_test.csv', delimiter = ',')['y_arousal']
        self.labels_val_v = pd.read_csv('merged_cleaned_sentiment_validation.csv', delimiter = ',')['y_valence']
        self.labels_val_a = pd.read_csv('merged_cleaned_sentiment_validation.csv', delimiter = ',')['y_arousal']

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
        
    def init_regr(self, params = None):
        if self.regressor == 'linreg':
            return LinearRegression()
        elif self.regressor == 'rf':
            return RandomForestRegressor(n_estimators= params[0], max_depth= params[1])
        elif self.regressor == 'mlp':
            return MLPRegressor(params)
        elif self.regressor == 'svr':
            return SVR()

    def get_features(self, features, data):

        if len(features) > 5:
            features = features.split('+')
        else:
            features = [features]

        index_dict = {'vader': (0, 5), 'tfidf': (5,156), 'anew': (156, 256)}

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
        regr_val = self.init_regr(self.params)
        regr_val.fit(train, lab_train_v)
        self.prediction_val = regr_val.predict(test)
        r2_val = r2_score(lab_v, self.prediction_val)

        regr_ar = self.init_regr(self.params)
        regr_ar.fit(train, lab_train_a)
        self.prediction_ar = regr_ar.predict(test)
        r2_ar = r2_score(lab_a, self.prediction_ar)

        return (r2_val, r2_ar)

    def generate_table(self, results):
        print(f"Model: {self.regressor}\t\t\t\t R^2")
        print("Features\t\tValence\tArousal")
        for key in results.keys():
            print(f"{key}\t\t{results[key][0]}\t{results[key][1]}")

### HYPERPARAMETER OPTIMIZATION: RANDOM FOREST

params = {'max_depth':[10, 15, 20], 'n_estimators':[50,100]}

for i in params['n_estimators']:
    for j in params['max_depth']:
        m = LyricsTable(regressor='rf', testset='val', params = [i,j]) 
        kf = KFold(n_splits = 5)

        r2s = []

        for train_index, test_index in kf.split(m.features_train):
            X_train, X_test = m.features_train[train_index], m.features_train[test_index]
            y_train_v, y_test_v = m.labels_train_v[train_index], m.labels_train_v[test_index]
            y_train_a, y_test_a = m.labels_train_a[train_index], m.labels_train_a[test_index]

            r2_v, r2_a = m.regression_r2(X_train, X_test, y_train_a, y_train_v, y_test_a, y_test_v)
            r2s.append(np.array([r2_v, r2_a]))

        print(f'Estimator:{i}, Depth:{j}, R2 = {np.sum(r2s, axis = 0)/5}')

"""
Results:
Estimator:10, Depth:2, R2 = [ 0.00172388 -0.00253295]
Estimator:10, Depth:5, R2 = [-0.00969125 -0.01562575]
Estimator:10, Depth:10, R2 = [-0.04541811 -0.04255748]
Estimator:10, Depth:15, R2 = [-0.08558572 -0.08581147]
Estimator:10, Depth:20, R2 = [-0.09596518 -0.1100134 ]
Estimator:50, Depth:2, R2 = [ 0.00227057 -0.00204339]
Estimator:50, Depth:5, R2 = [ 0.00140073 -0.00258205]
"""




    

