# -*- coding: utf-8 -*-
"""
Created on Tue May 30 19:18:15 2017

@author: SaeromPark
"""

import os,sys
os.chdir('C:/Users/SaeromPark/Documents/Python Scripts/doc2vec_test')

#from doc2vec_revised2 import *
import numpy as np
import gensim
from gensim import corpora, models
import pickle
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from gensim.matutils import corpus2dense
tokenizer = RegexpTokenizer(r'\w+')

#import word2vec

data_path = '../docvec_data'
category = 'books'
unla_ratio = [0.3,0.4,0.5,0.6,0.7]


#rr = unla_ratio[4]
for rr in unla_ratio:
    f = open(data_path+'/results2/'+category+'_'+str(rr)+'_texts.pickle','rb')
    texts_df = pickle.load(f)
    f.close()
    text = list(texts_df['text'])
    doc_label = texts_df['train_label']
    true_label = texts_df['true_label']
    
    d_size = 200
    
    texts = [tokenizer.tokenize(tx) for tx in text]

    dictionary = corpora.Dictionary(texts)

    corpus = [dictionary.doc2bow(text) for text in texts]
    
    ## applying LDA model
    print("start to training LDA")
    ldamodel = gensim.models.ldamodel.LdaModel(corpus,num_topics = d_size,id2word = dictionary)
    doc_mat = ldamodel[corpus]
    doc_vecs = list(corpus2dense(doc_mat,d_size).T)
    file_name = data_path + '/results2/lda_'+category+'_'+str(rr)+'_data.pickle'
    ldamat = {'train_label':doc_label,'true_label':true_label,'docvec':doc_vecs}
    ladmat = pd.DataFrame(ldamat)
    f = open(file_name,'wb')
    pickle.dump(ladmat,f)
    f.close()
    print("end for training :",category,rr)
    
