#loading need libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler

features_train_names = ['embbedings_source_features.csv', 'embbedings_glove_features.csv', 'embbedings_fast_text_features.csv', 
'embbedings_wang2vec_features.csv', 'tfidfvectorizer_features.csv', 'wang2vec_attention_capsule_features.csv', 'wang2vec_attention_cnn_features.csv',
 'embeddings_source_attention_capsule_features.csv', 'embeddings_source_attention_cnn_features.csv', 'embedding_nlp_glove_300_features.csv',
 'embeddings_source_300_features.csv', 'embeddings_source_title_features.csv', 'wang2vec_title_features.csv', 'tfidfvectorizer_title_features.csv',
  'wang2vec_attention_cnn_title_features.csv', 'wang2vec_attention_capsule_title_features.csv' , 'feature-target-train.csv', 'image-features.csv']


features_test_names = ['embeddings_source_lb.csv', 'embeddings_glove_lb.csv', 'embeddings_fast_text_lb.csv', 'embeddings_wang2vec_lb.csv',
'tfidfvectorizer_lb.csv', 'wang2vec_attention_capsule_lb.csv', 'wang2vec_attention_cnn_lb.csv', 'embeddings_source_attention_capsule_lb.csv',
 'embeddings_source_attention_cnn_lb.csv', 'embedding_nlp_glove_300_lb.csv', 'embeddings_source_300_lb.csv', 'embeddings_source_title_lb.csv',
  'wang2vec_title_lb.csv', 'tfidfvectorizer_title_lb.csv', 'wang2vec_attention_cnn_title_lb.csv', 'wang2vec_attention_capsule_title_lb.csv', 'feature-target-test.csv', 'image-lb.csv']

needs_to_transform = ['feature-target-train.csv', 'feature-target-test.csv', 'tfidfvectorizer_lb.csv', 'tfidfvectorizer_title_lb.csv']

dataframes = []
for f in features_train_names:
	print(f)
	t = pd.read_csv('feature-embeddings/' + f, names=['feature'], header=None)['feature']
	if f in needs_to_transform:
		print("Need to log: ")
		print(t.head())
		t = np.log1p(t)
	print(t.head())
	dataframes.append(t)

features_train = pd.concat(dataframes, axis=1)

features_train.to_csv('luck-features-train.csv')

dataframes = []
for f in features_test_names:
	print(f)
	if f in ['image-lb.csv', 'feature-target-test.csv']:
		t = pd.read_csv('feature-embeddings/' + f, names=['feature'], header=None)['feature']
	else:
		t = pd.read_csv('feature-embeddings/' + f, names=['id', 'feature'], header=None)['feature']
	if f in needs_to_transform:
		print("Need to log: ")
		t = np.log1p(t)
	print(t.head())
	dataframes.append(t)
features_test = pd.concat(dataframes, axis=1)


features_test.to_csv('luck-features-test.csv')

