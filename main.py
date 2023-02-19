from data.features import features_to_csv
import os 
import pandas as pd


# 2712/3863 on spotify 
if os.path.isfile('data/features/features_validation.csv'):
    validation = pd.read_csv('data/features/features_validation.csv', delimiter=',')
else:
    validation = features_to_csv(dataset = 'data/deezer_mood_detection_dataset/validation', whole_dataset = True)

# 2572/3514 on spotify 
if os.path.isfile('data/features/features_test.csv'):
    train = pd.read_csv('data/features/features_test.csv', delimiter=',')
else:
    train = features_to_csv(dataset = 'data/deezer_mood_detection_dataset/test', whole_dataset = True)

# 3152/11267 on spotify 
if os.path.isfile('data/features/features_train.csv'):
    train = pd.read_csv('features_train.csv', delimiter=',')
else:
    train = features_to_csv(dataset = 'data/deezer_mood_detection_dataset/train', whole_dataset = True)

