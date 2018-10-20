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
from multiprocessing import Process, Queue
from IPython import embed
import numpy as np

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
def find_nn_idx(i,syn0norm,result_idxes):
    #print(i)
    topn=10
    dists = dot(syn0norm, syn0norm[i])
    if not topn:
        return dists
    best = matutils.argsort(dists, topn=topn + 1, reverse=True)
    # ignore (don't return) docs from the input
    result = best[1:]
    result_idxes = np.array(result[:topn],copy = True)
    return result[:topn]
    
    
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
       # nn_ = np.array(nn,copy = True)
       # self.nn=nn
       # print(nn[:2])
       # print(nn_[:2])
       # print(type(nn),type(nn_))

if __name__ == "__main__":
    #data='yelp'
    #name='01'
    doc_model=pickle.load(open('../results/kitchen_doc2vec_dbow_neighbor_0.3_beta_0.03.doc2vec','rb'))
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
    nn_mlt = []
    
    nn_list = np.zeros((n,10)) 
    t1=time.time()
    mlt = [Process(target = find_nn_idx, args = (i,syn0norm,nn_list[i])) for i in range(n)]
#    for i in range(n):
#        mlt[i].start()
#        mlt[i].join()
    print(time.time()-t1)
    embed() 
    
    
