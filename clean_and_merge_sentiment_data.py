import pandas as pd
import numpy as np
import re
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def preprocess_text(text: str) -> str:
    '''removes everything between square brackets (like [chorus])
    deletes \\m lineseperator'''
    text = re.sub("[\[].*?[\]]", " ", text)
    text = re.sub('\\\\m', ' ',text)
    return text

def get_sentiment(lyrics):
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(lyrics)
    return ss

def clean_merge_data_to_csv(train_test_validation:str):
    file = 'features_' + train_test_validation + '_complete.csv'
    file2 = train_test_validation + '.csv' 
    df_feat = pd.read_csv(file)
    df_feat = df_feat.drop(columns = df_feat.columns[0])
    df_emo = pd.read_csv(os.path.join('deezer_mood_detection_dataset',file2))
    df_feat = df_feat.dropna()
    # apply to whole df
    df_feat['lyrics_cleaned'] = df_feat['lyrics'].apply(lambda x: preprocess_text(x))
    df_emo.rename(columns = {'artist_name':'artist','track_name':'trackname','valence':'y_valence', 'arousal': 'y_arousal'},inplace = True)
    df_emo.drop(columns = ['dzr_sng_id','MSD_sng_id','MSD_track_id'], inplace = True)
    merged_data= df_feat.merge(df_emo,how = 'inner',on=["artist","trackname"])
    print(f'{train_test_validation} data cleaned')

    # get sentiment
    print(f'Getting sentiment for {train_test_validation} data...')
    merged_data['vader'] = merged_data['lyrics_cleaned'].apply(lambda x: get_sentiment(x))
    new_df = pd.DataFrame(list(merged_data['vader']))
    df_vader_cleaned = merged_data.join(new_df, how = 'outer')
    df_vader_cleaned = df_vader_cleaned.drop(columns = 'vader')
    print(df_vader_cleaned.head())
    # save csv
    new_filename = 'merged_cleaned_sentiment_'+ train_test_validation + '.csv'
    df_vader_cleaned.to_csv(new_filename)
    print(f'new csv for {train_test_validation} saved')
    
clean_merge_data_to_csv(train_test_validation = 'validation')
# clean_merge_data_to_csv(train_test_validation = 'train')
# clean_merge_data_to_csv(train_test_validation = 'test')




