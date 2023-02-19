# Multi-Modality in Music: Predicting Emotion in Music from High-Level Audio Features and Lyrics

This project aims at testing if a multi-modal approach for music emotion recognition (MER) performs better than a uni-modal one on high-level song features and lyrics. We use 11 song features retrieved from the Spotify API, combined lyrics features including sentiment, TF-IDF and Anew to predict valence and arousal [Russel, 1980](https://psycnet.apa.org/record/1981-25062-001) scores on the [Deezer Mood Detection Dataset (DMDD)](https://research.deezer.com/publication/2018/09/26/ismir-delbouys.html) (Delbouys, 2018) with 4 different regression models. We find that out of the 11 high-level song features, mainly 5 contribute to the performance, multi-modal features do better than audio alone when predicting valence.

## Files
All the data is in the data directory, splitted into features, for music features like loudness, key, etc., lyrics, for the pure lyrics from which you can extract the sentiment via [VADER](https://github.com/cjhutto/vaderSentiment), the original deezer_dataset that comes with the valence and arousal scores which we used for the model, and a folder that holds all merged features.

The python files and respective notebooks are in the utils folder. The naming should be self-explanatory.

## Usage
We have a main.py and a finalmodel.ipynb. You should use the finalmodel to get the results we got (the main.py is mainly just importing stuff). We're planning on adding a setup.py very soon.

## Contact
If you have any questions, we're happy to answer them if you contact us directly.

## Citing
TBD
