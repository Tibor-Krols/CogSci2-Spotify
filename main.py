from utils import features_to_csv
import os 
import pandas as pd

if os.path.isfile('features_validation.csv'): 
    validation = pd.read_csv('features_validation.csv', delimiter=',')
else:
    validation = features_to_csv(dataset = 'validation', whole_dataset = True)

if os.path.isfile('features_test.csv'): 
    train = pd.read_csv('features_test.csv', delimiter=',')
else:
    train = features_to_csv(dataset = 'test', whole_dataset = True)