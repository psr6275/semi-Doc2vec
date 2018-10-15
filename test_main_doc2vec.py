# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:30:58 2016

@author: SaeromPark
"""

#import gensim
#from gensim.models.doc2vec import *
import os,sys
os.chdir('C:/Users/SaeromPark/Documents/Python Scripts/doc2vec_test')

from doc2vec_revised import *
import word2vec
import pickle
import numpy as np

import re
import copy
import nltk

import pandas as pd


data_path = '../docvec_data'

f = open(data_path+'/kitchen_processed_unlabeled.txt','rb')
unla_text = pickle.load(f)
f.close()

f = open(data_path+'/kitchen_processed_unlabeled_label.txt','rb')

unla_label = pickle.load(f)
f.close()
unla_text = np.array(unla_text)
unla_label = np.array(unla_label)
nega_idx = unla_label<3
posi_idx = unla_label>3
nega_text = list(unla_text[nega_idx])
posi_text = list(unla_text[posi_idx])
len_text = min(len(nega_text),len(posi_text))
text = posi_text[2000:len_text] + nega_text[2000:len_text] + posi_text[:2000]+ nega_text[:2000]
#text = posi_text + nega_text + unla_text[:1000]

documents = []
for document in text:
    temp = document.replace('!','.').replace('?','.').replace(';','.').replace(':','.').replace('\n',' ').strip()
    documents.append(temp.split('.'))

# we need to make the setences as lower cases or only letters!   
doc_labels = np.ones(len(text))
doc_labels[len_text-2000:-4000] = -1
doc_labels[-4000:]= 0
true_labels = doc_labels
true_labels[-4000:-2000] = 1
true_labels[-2000:] = -1

labels = {'train_label':doc_labels,'true_label':true_labels}
labels = pd.DataFrame(labels)

f = open(data_path+'/kitchen_train_label_more_unlabel.txt','wb')
pickle.dump(labels,f)
f.close()

sentences = []
for uid, doc in enumerate(documents):
    for sen in doc:
        sen.lower()
        sen = re.sub("[^a-zA-Z]"," ", sen)
        if uid < len_text-2000:
            sentence = TaggedDocument(words = sen.split(),tags = ['POS_%s' %uid])
        elif uid <2*(len_text-2000):
            sentence = TaggedDocument(words = sen.split(),tags = ['NEG_%s' %(uid-1000)])
        else:
            sentence = TaggedDocument(words = sen.split(),tags = ['UNLA_%s' %(uid-2000)])
        sentences.append(sentence)
        
        
d_size = 400


model_dm_mean = Doc2Vec(sentences,doc_labels, size=d_size, window = 3, min_count=3, workers = 8,dm_mean = 1,iter = 10)
"""
for epoch in range(3):
    model_dm_mean.train(sentences)
    model_dm_mean.alpha -= 0.003
    #model.min_alpha = model.alpha
"""
model_dm_mean.save('doc2vec_dm_mean.doc2vec')    

model_dbow = Doc2Vec(sentences,doc_labels, size=d_size, window = 3, min_count=3, workers = 8, dm = 0,iter = 10)
"""
for epoch in range(5):
    model_dbow.train(sentences)
    model_dbow.alpha -= 0.003
"""
model_dbow.save('doc2vec_dbow_gensim.doc2vec')        
model_neighbor = Doc2Vec(sentences,doc_labels,beta = 0.015, size=d_size, window = 3, min_count=3, workers = 8, dbow_neighbor = 1,iter = 10)
"""
for epoch in range(5):
    model_neighbor.train(sentences)
    model_neighbor.alpha -= 0.003
"""
model_neighbor.save('doc2vec_dbow_neighbor_kitchen_more_unlabel_beta0.015.doc2vec')        