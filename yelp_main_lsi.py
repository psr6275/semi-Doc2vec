# -*- coding: utf-8 -*-
"""
Created on Tue May 30 19:18:15 2017

@author: SaeromPark
"""

import os,sys

import numpy as np
import gensim
from gensim import corpora, models
import pickle
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from gensim.matutils import corpus2dense

tokenizer = RegexpTokenizer(r'\w+')

data_path = '/data/doc2vec_data/data/'
result_path = '/data/doc2vec_data/results/'

year = '2010'
unla_ratio = [0.3,0.4,0.5,0.6,0.7]


#rr = unla_ratio[4]
for rr in unla_ratio:
    f = open(data_path+'yelp_reviews_'+year+'_'+str(rr)+'_texts.pickle','rb')
    texts_df = pickle.load(f)
    f.close()
    text = list(texts_df['text'])
    doc_label = texts_df['train_label']
    true_label = texts_df['true_label']
    
    d_size = 200
    print('Load data',rr) 
    texts = [tokenizer.tokenize(tx) for tx in text]

    dictionary = corpora.Dictionary(texts)

    corpus = [dictionary.doc2bow(text) for text in texts]
    ### obtaining tf-idf
    tfidfmodel = models.TfidfModel(corpus,normalize=True)
    tfidf_corpus = tfidfmodel[corpus]
    ## applying LSI model
    print("start to training LSI")
    lsimodel = gensim.models.LsiModel(tfidf_corpus, id2word = dictionary,num_topics = d_size)
    doc_mat = lsimodel[tfidf_corpus]
    doc_vecs = list(corpus2dense(doc_mat,d_size).T)
    file_name = result_path + 'yelp_lsi_'+year+'_'+str(rr)+'_data.pickle'
    lsimat = {'train_label':doc_label,'true_label':true_label,'docvec':doc_vecs}
    lsimat = pd.DataFrame(lsimat)
    f = open(file_name,'wb')
    pickle.dump(lsimat,f)
    f.close()
    print("end for training :",year,rr)
    
