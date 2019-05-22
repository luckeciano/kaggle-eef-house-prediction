# kaggle-eef-house-prediction

Code from some models used in First Place's solution of Kaggle Data Science Challenge for EEF for ITA/Unifesp SJC students.

Kaggle Competition Link: https://www.kaggle.com/c/data-science-challenge-at-eef-2019/leaderboard

The code here is related to models I developed (before and after join "Los Bertones" team). Here you will encounter the following models:

The main repository of Los Bertones' team is: https://github.com/brunoklaus/EEF_2019

There is two main models: One is a stacking model using a diversity of 18 models. The other is a single network that uses
all kinds of data to train a predictor.

* baseline.py   -- Baseline with just tabular data                          final_model.py
* image_house_pred.py - Predictor using features extracted from images using NASNetMobile
* embedding_nlp_glove_300.py - GloVe Word Embeddings using description text data (length of 300)
* embedding_nlp_glove.py - Same as above, length = 500
* embeddings_nlp_fast_text.py - FastText Word Embeddings using description text data (length of 300)
* lstm_house_pred.py - Trainable embeddings, length of 500
* embeddings_source_300.py - Trainable embeddings, length of 300
* embeddings_source_attention_capsule.py - Trainable embeddings + Attention + Capsule Layer
* embeddings_source_attention_cnn.py - Trainable embeddings + Attention + CNN Layer
* tfidvectorizer_nlp.py - Using TF IDF features from description in a MLP network
* wang2vec_attention_capsule.py - Wang2Vec Word Embeddings + Attention + Capsule
* final_model.py -- Stacking model using 18 models previously trained
* final_model_merged.py -- Another model that shares embedding layer between all text data, uses image features and tabular data in a single network.

OBS: The other models with same nomenclature + title refers to the same network used by title text.

# Stacking Model
![alt text](https://github.com/luckeciano/kaggle-eef-house-prediction/blob/master/model2.png "Model 2")


# Single network model
![alt text](https://github.com/luckeciano/kaggle-eef-house-prediction/blob/master/model3.png "Model 3")
