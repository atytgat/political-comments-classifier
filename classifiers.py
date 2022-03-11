# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 17:51:26 2020

@author: Beuseling Niels, Cherpion Alexis, Tytgat Alexandre


"""

import pandas as pd
import numpy as np
import tqdm
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import gensim
from gensim.test.utils import get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags       # strip html tags
from gensim.parsing.preprocessing import strip_short      
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import stem_text
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation, strip_non_alphanum

from sentence_transformers import SentenceTransformer


# load the data
biden_train_csv = "C:/Users/alext/OneDrive - UCL/UCL/DATA M2/LINMA2472 - Algorithms in Data Science/projet embeddings/data/JoeBiden_train.csv"
trump_train_csv = "C:/Users/alext/OneDrive - UCL/UCL/DATA M2/LINMA2472 - Algorithms in Data Science/projet embeddings/data/The_Donald_train.csv"
biden_test_csv = "C:/Users/alext/OneDrive - UCL/UCL/DATA M2/LINMA2472 - Algorithms in Data Science/projet embeddings/data/JoeBiden_test.csv"
trump_test_csv = "C:/Users/alext/OneDrive - UCL/UCL/DATA M2/LINMA2472 - Algorithms in Data Science/projet embeddings/data/The_Donald_test.csv"

biden_train = pd.read_csv(biden_train_csv)['body'].to_numpy()
trump_train = pd.read_csv(trump_train_csv)['body'].to_numpy()
biden_test = pd.read_csv(biden_test_csv)['body'].to_numpy()
trump_test = pd.read_csv(trump_test_csv)['body'].to_numpy()


#%%
# sentences separation in each set
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


biden_train_sents = []
for doc in tqdm(biden_train):
    sents = sent_detector.tokenize(doc.strip())
    biden_train_sents = biden_train_sents + sents

trump_train_sents = []
for doc in tqdm(trump_train):
    sents = sent_detector.tokenize(doc.strip())
    trump_train_sents = trump_train_sents + sents
    
biden_test_sents = []
for doc in tqdm(biden_test):
    sents = sent_detector.tokenize(doc.strip())
    biden_test_sents = biden_test_sents + sents

trump_test_sents = []
for doc in tqdm(trump_test):
    sents = sent_detector.tokenize(doc.strip())
    trump_test_sents = trump_test_sents + sents
    

# add the labels
biden_train_df = pd.DataFrame({'body' : biden_train_sents, 'label':0})
trump_train_df = pd.DataFrame({'body' : trump_train_sents, 'label':1})
biden_test_df = pd.DataFrame({'body' : biden_test_sents, 'label':0})
trump_test_df = pd.DataFrame({'body' : trump_test_sents, 'label':1})

# create full training set and test set
train_set = biden_train_df.append(trump_train_df)
test_set = biden_test_df.append(trump_test_df)

# shuffle
train_set = shuffle(train_set, random_state=123)
test_set = shuffle(test_set, random_state=123)

# separate sentences from their labels
text_train = np.ravel( train_set.loc[:, ['body']].to_numpy() )
label_train = np.ravel( train_set.loc[:, ['label']].to_numpy() )
text_test = np.ravel( test_set.loc[:, ['body']].to_numpy() )
label_test = np.ravel( test_set.loc[:, ['label']].to_numpy() )

# Additional preprocessing for Doc2Vec embedding
CUSTOM_FILTERS = [lambda x: x.lower(),
                  strip_multiple_whitespaces, strip_punctuation, strip_non_alphanum,
                  remove_stopwords, strip_tags, strip_short, stem_text]

train_sents_processed = []
for sent in tqdm(text_train):
    parsed_line = preprocess_string(sent, CUSTOM_FILTERS)
    train_sents_processed.append(parsed_line)
    
test_sents_processed = []
for sent in tqdm(text_test):
    parsed_line = preprocess_string(sent, CUSTOM_FILTERS)
    test_sents_processed.append(parsed_line)
    
    
#%% Doc2Vec encoding

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_sents_processed)] 

# DEFINE MODEL
encoder_D2V = gensim.models.Doc2Vec(vector_size=300, window=4,  compute_loss = True,
                               min_count=5, alpha = 0.1, worker=-1)

# BUILD VOCABULARY
encoder_D2V.build_vocab(documents)

# AND TRAIN THE MODEL
iterations = tqdm(range(1))
for i in iterations:
    encoder_D2V.train(documents, total_examples=encoder_D2V.corpus_count,
                epochs = 100)
    msg = f"Iter :: {i}"
    iterations.set_postfix_str(s = msg, refresh=True)

# save model !
fname4 = get_tmpfile("encoder_D2Vbis")
encoder_D2V.save(fname4)

#%%
fname4 = get_tmpfile("encoder_D2Vbis")
encoder_D2V = Doc2Vec.load(fname4) 
test_inferred = []
for sent in tqdm(test_sents_processed):
    test_inferred.append(encoder_D2V.infer_vector(sent, min_alpha = 1e-16, epochs = 5000))
    
# create list of vectors for all sets
train_D2V = []
for vec in tqdm(range(len(encoder_D2V.docvecs))):
    train_D2V.append(list(encoder_D2V.docvecs[vec]))

test_D2V = []
for vec in tqdm(test_inferred):
    test_D2V.append(list(vec))
    
#%%
# save !
pd.DataFrame(train_D2V).to_csv("C:/Users/alext/OneDrive - UCL/UCL/DATA M2/LINMA2472 - Algorithms in Data Science/projet embeddings/data/train_D2V.csv", index=False) 
pd.DataFrame(test_D2V).to_csv("C:/Users/alext/OneDrive - UCL/UCL/DATA M2/LINMA2472 - Algorithms in Data Science/projet embeddings/data/test_D2V.csv", index=False) 
pd.DataFrame(label_train).to_csv("C:/Users/alext/OneDrive - UCL/UCL/DATA M2/LINMA2472 - Algorithms in Data Science/projet embeddings/data/label_train.csv", index=False) 
pd.DataFrame(label_test).to_csv("C:/Users/alext/OneDrive - UCL/UCL/DATA M2/LINMA2472 - Algorithms in Data Science/projet embeddings/data/label_test.csv", index=False) 

#%% BERT encoding 

embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
    
train_bert = embedder.encode(text_train)
test_bert = embedder.encode(text_test)

# save !
pd.DataFrame(train_bert).to_csv("C:/Users/alext/OneDrive - UCL/UCL/DATA M2/LINMA2472 - Algorithms in Data Science/projet embeddings/data/train_bert.csv", index=False) 
pd.DataFrame(test_bert).to_csv("C:/Users/alext/OneDrive - UCL/UCL/DATA M2/LINMA2472 - Algorithms in Data Science/projet embeddings/data/test_bert.csv", index=False) 


#%% Load the processed data

train_D2V = pd.read_csv("C:/Users/alext/OneDrive - UCL/UCL/DATA M2/LINMA2472 - Algorithms in Data Science/projet embeddings/data/train_D2V.csv").to_numpy()
test_D2V = pd.read_csv("C:/Users/alext/OneDrive - UCL/UCL/DATA M2/LINMA2472 - Algorithms in Data Science/projet embeddings/data/test_D2V.csv").to_numpy()
train_bert = pd.read_csv("C:/Users/alext/OneDrive - UCL/UCL/DATA M2/LINMA2472 - Algorithms in Data Science/projet embeddings/data/train_bert.csv").to_numpy()
test_bert = pd.read_csv("C:/Users/alext/OneDrive - UCL/UCL/DATA M2/LINMA2472 - Algorithms in Data Science/projet embeddings/data/test_bert.csv").to_numpy()
label_train = np.ravel( pd.read_csv("C:/Users/alext/OneDrive - UCL/UCL/DATA M2/LINMA2472 - Algorithms in Data Science/projet embeddings/data/label_train.csv").to_numpy() )
label_test = np.ravel( pd.read_csv("C:/Users/alext/OneDrive - UCL/UCL/DATA M2/LINMA2472 - Algorithms in Data Science/projet embeddings/data/label_test.csv").to_numpy() )

#%% Cross-validation on model based on Doc2Vec embedding

# add preprocessing ? pca,ica, normalization,...
classifier_D2V = RandomForestClassifier(random_state=0)

parameters_D2V = {'n_estimators':[300,400,500], 'max_depth':[9], 'max_features':['log2','sqrt'],
              'min_samples_leaf':[4,6], 'min_samples_split':[4]}
               

clf_D2V = GridSearchCV(classifier_D2V, parameters_D2V, cv=5, scoring='accuracy', n_jobs=-1)
clf_D2V.fit(train_D2V, label_train)

pred_train_D2V = clf_D2V.predict(train_D2V)
pred_test_D2V = clf_D2V.predict(test_D2V)

print(clf_D2V.best_params_)
print(f"Accuracy:\n  training :: {accuracy_score(label_train, pred_train_D2V)}\n  test     :: {accuracy_score(label_test, pred_test_D2V)}")

#%% Cross-validation on model based on BERT embedding

classifier_BERT = RandomForestClassifier(random_state=0)

# parameters_BERT = {'n_estimators':[100,200,300], 'max_depth':[5,7,9], 'max_features':['log2','sqrt'],
#               'min_samples_leaf':[2,4]}
parameters_BERT = {'n_estimators':[300,400,500], 'max_depth':[9,12], 'max_features':['log2','sqrt'],
              'min_samples_leaf':[2,4]}

clf_BERT = GridSearchCV(classifier_BERT, parameters_BERT, cv=5, scoring='accuracy', n_jobs=-1)
clf_BERT.fit(train_bert, label_train)

pred_train_BERT = clf_BERT.predict(train_bert)
pred_test_BERT = clf_BERT.predict(test_bert)


print(clf_BERT.best_params_)
print(f"Accuracy:\n  training :: {accuracy_score(label_train, pred_train_BERT)}\n  test     :: {accuracy_score(label_test, pred_test_BERT)}")

#%% Final classifier based on Doc2Vec embedding

classifier_D2V = RandomForestClassifier(n_estimators=200,
                                      max_depth=9,
                                      max_features='log2',
                                      min_samples_leaf=2,
                                      random_state=0)
classifier_D2V.fit(train_D2V, label_train)


pred_train_D2V = classifier_D2V.predict(train_D2V)
pred_test_D2V = classifier_D2V.predict(test_D2V)

print(f"Accuracy:\n  training :: {accuracy_score(label_train, pred_train_D2V)}\n  test     :: {accuracy_score(np.ravel(label_test), pred_test_D2V)}")

#%% Final classifier based on BERT embedding

classifier_BERT = RandomForestClassifier(n_estimators=200,
                                      max_depth=9,
                                      max_features='log2',
                                      min_samples_leaf=2,
                                      random_state=0)

classifier_BERT.fit(train_bert, label_train)


pred_train_BERT = classifier_BERT.predict(train_bert)
pred_test_BERT = classifier_BERT.predict(test_bert)

# print(confusion_matrix(Y_train_bert, pred1))
print(f"Accuracy:\n  training :: {accuracy_score(label_train, pred_train_BERT)}\n  test     :: {accuracy_score(np.ravel(label_test), pred_test_BERT)}")


