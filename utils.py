import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import string
import os
from tqdm import tqdm
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


#spotify token and credentials
cid = '3a13f1e179e64b1194a92ca3a81fe54f'
secret = '883a5a29ea7e442aa3dd47ee2f74a9d7'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)


def scrape_lyrics(artistname, songname):
    artistname2 = str(artistname.replace(' ','-')) if ' ' in artistname else str(artistname)
    songname2 = str(songname.replace(' ','-')) if ' ' in songname else str(songname)
    page = requests.get('https://genius.com/'+ artistname2 + '-' + songname2 + '-' + 'lyrics')
    html = BeautifulSoup(page.text, 'html.parser')
    #print(html)
    lyrics1 = html.find("div", class_="lyrics")
    lyrics2 = html.find_all("div", class_="Lyrics__Container-sc-1ynbvzw-6 jYfhrf")
    if lyrics1:
        lyrics = lyrics1.get_text("\m")
    elif lyrics2:
        #print(lyrics2)
        lyrics = ""
        for l in lyrics2: 
            lyrics += l.get_text("\m")
    elif lyrics1 == lyrics2 == None:
        lyrics = None
    return lyrics

#function to attach lyrics onto data frame
#artist_name should be inserted as a string
def lyrics_onto_frame(df1, artist_name):
    for i,x in enumerate(df1['track']):
        test = scrape_lyrics(artist_name, x)
        df1.loc[i, 'lyrics'] = test
    return df1

def features_to_csv(dataset, n_songs = 0, whole_dataset = False):
    """function that gets all the audio features from spotify and the lyrics from genius and saves it as a csv
    dataset: string value that is either 'train', 'test' or 'validation' to indicate which dataset to use
    n_songs = int value, depening on how many songs you want to get the track features and lyrics from
    whole_dataset: bool, if true it will get the whole dataset, otherwise, the specified n_songs
    it returns df_features"""
    # features dictionary 


    file = dataset + '.csv'
    path = os.path.join('deezer_mood_detection_dataset',file)
    songs = pd.read_csv(path, delimiter = ',')
    
#     tester = songs.head(n_songs)
#     df = pd.concat([tester['track_name'], tester['artist_name']], axis = 1, keys = ['track_name', 'artist_name'])

    if whole_dataset == False:
        data = songs.head(n_songs)
    else:
        data = songs
        n_songs = len(data)

    df = pd.concat([data['track_name'], data['artist_name']], axis = 1, keys = ['track_name', 'artist_name'])
    
    data_dict = {'artist':[],
                'trackname': [],
                'danceability': [],
                 'energy':[],
                 'key': [],
                 'loudness': [],
                 'mode': [],
                 'speechiness': [],
                 'acousticness': [],
                 'instrumentalness': [],
                 'liveness': [],
                 'valence': [],
                 'tempo': [],
                 'id': [],
                 'time_signature': [], 
                 'lyrics': []
                }

    data_dict_copy = data_dict.copy()

    pbar = tqdm(total=n_songs)

    headers = pd.DataFrame(data_dict)
    headers.to_csv('features_'+ dataset +'.csv')


    for i in range(n_songs):
        artist_name = df.iloc[i][1]
        track_name = df.iloc[i][0]

        #search for spotify tack 
        track_results = sp.search(q=f'artist: {artist_name}, track:{track_name}', type='track', limit=10,offset=0)
        if len(track_results['tracks']['items']) > 0: #if track exists
            if track_results['tracks']['items'][0]['name'] == track_name: #validating track name 

                #get audio features and append to feature dictionary 
                ID = track_results['tracks']['items'][0]['id']
                audio_features = sp.audio_features(ID)[0] 
                if audio_features:
                    for key in list(data_dict_copy.keys())[2:-1]:#until -3 because the last 3 features are lyrics artist, trackname a
                        data_dict_copy[key].append(audio_features[key])

                    #scrape lyrics from genius.com
                    try:
                        track_name_process = re.sub(r'[^\w\s]', '', track_name)   
                        lyrics = scrape_lyrics(artist_name, track_name_process)
                        data_dict_copy['lyrics'].append(lyrics)
        #                 print(lyrics)
                    #if not found append None
                    except:  
                        #print(track_name)
                        data_dict_copy['lyrics'].append(None)


        #             attempt to addd artist and trackname, but i think somehting is still wrong
                    data_dict_copy['artist'].append(artist_name)
                    data_dict_copy['trackname'].append(track_name)

        pbar.update(1) 
        
        # if i %50 ==0: #every 100 songs, save the dateframe to a csv
        #     df_features = pd.DataFrame(data_dict_copy) #convert dict to df
        #     df_features.to_csv('features_'+ dataset +'.csv', mode='a', header=False)
        #     data_dict_copy = data_dict.copy()
        #     #print(f'df saved at the {i}th iteration')

    pbar.close()

#     save df at the end
    df_features = pd.DataFrame(data_dict_copy) #convert dict to df
    df_features.to_csv('features_'+ dataset +'.csv', mode='a', header=False)
    return df_features

    