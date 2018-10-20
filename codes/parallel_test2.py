# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 09:47:19 2016

@author: slcf-wserver-12
"""
from joblib import Parallel, delayed
import os
import word2vec
import doc2vec 
from word2vec import Word2Vec, Vocab, train_cbow_pair, train_sg_pair, train_batch_sg, train_cbow_sentiment_pair, train_sswe
from doc2vec import Doc2Vec, DocvecsArray, Doctag
import pickle
from numpy import dot, empty, divide as np_divide, sqrt
import utils, matutils
import timeit
import time
import multiprocessing
from IPython import embed

def find_nn(i,syn0norm):
    #print(i)
    topn=10
    dists = dot(syn0norm, syn0norm[i])
    if not topn:
        return dists
    best = matutils.argsort(dists, topn=topn + 1, reverse=True)
    # ignore (don't return) docs from the input
    result = [(sim, float(dists[sim])) for sim in best[1:]]
    return result[:topn]
    
    #return doc_model.docvecs.most_similar(i)    
    
    
class MNN(object):
    def __init__(self,doc_model):
        self.doc_model=doc_model
    def run(self):
        doc_model=self.doc_model
        docvecs=doc_model.docvecs
        n=len(doc_model.docvecs)
        #p=len(doc_model.docvecs[0])
        #docvecs.init_sims()
        #self.doctag_syn0norm = docvecs.doctag_syn0norm
        self.syn0norm=empty(docvecs.doctag_syn0.shape,dtype='float32')
        np_divide(docvecs.doctag_syn0,sqrt((docvecs.doctag_syn0**2).sum(-1))[...,None],self.syn0norm)
        #nn=Parallel(n_jobs=6)(delayed(find_nn)(i,self.syn0norm) for i in range(n))
        nn=[find_nn(i,self.syn0norm) for i in range(n)]
        self.nn=nn

if __name__ == "__main__":
    #data='yelp'
    #name='01'
    doc_model=pickle.load(open('../results/'%(data,name),'rb'))
    mnn=MNN(doc_model)    
    t1=time.time()
    mnn.run()
    print(time.time()-t1)
    
    t1=time.time()
    nn=[doc_model.docvecs.most_similar(i) for i in range(len(doc_model.docvecs))]
    print(time.time()-t1)
    
    syn0norm=mnn.syn0norm
    n=len(syn0norm)
    '''    
    t1=time.time()
    nn_pll=Parallel(n_jobs=6)(delayed(find_nn)(i,syn0norm) for i in range(n))    
    print(time.time()-t1)
    '''
    t1=time.time()
    nn_mlt=[multiprocessing.Process(target=find_nn, args=(i,syn0norm,)) for i in range(n)]
    print(time.time()-t1)
    
    
