# Multi-Modality in Music: Predicting Emotion in Music from High-Level Audio Features and Lyrics

This project aims at testing if a multi-modal approach for music emotion recognition (MER) performs better than a uni-modal one on high-level song features and lyrics. We use 11 song features retrieved from the Spotify API, combined lyrics features including sentiment, TF-IDF and Anew to predict valence and arousal [Russel, 1980](https://psycnet.apa.org/record/1981-25062-001) scores on the Deezer Mood Detection Dataset (DMDD) \shortcite{delbouys_music_2018} with 4 different regression models. We find that out of the 11 high-level song features, mainly 5 contribute to the performance, multi-modal features do better than audio alone when predicting valence.
