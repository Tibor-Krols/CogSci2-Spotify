import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import IncrementalPCA 


data = pd.read_csv('../data/merged/merged_cleaned_sentiment_train.csv', delimiter = ',')
data_test = pd.read_csv('../data/merged/merged_cleaned_sentiment_test.csv', delimiter = ',')

data_lyrics_train = np.array(data['lyrics_cleaned'])
data_lyrics_test = np.array(data_test['lyrics_cleaned'])

y_train = data[['y_valence', 'y_arousal']]
y_test = data_test[['y_valence', 'y_arousal']]

print("Vectorizing counts...")

vectorizer = CountVectorizer(analyzer= 'word', ngram_range = (2,2))
X_train = vectorizer.fit_transform(data_lyrics_train)
X_test = vectorizer.transform(data_lyrics_test)

print('Performing PCA...')

pca = IncrementalPCA(n_components = 80, batch_size = 500)
pca.fit(X_train.toarray().astype(float))
pca_lyrics_train = pca.transform(X_train.toarray().astype(float))
pca_lyrics_test = pca.transform(X_test.toarray().astype(float))
print('Explained variance:', pca.explained_variance_ratio_)

print("Performing Linear Regression...")

reg = LinearRegression()
reg.fit(pca_lyrics_train,y_train)
prediction = reg.predict(pca_lyrics_test)
print('RMSE:', mean_squared_error(y_test, prediction))
print('R2:', r2_score(y_test, prediction))
