# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 08:51:40 2016

@author: SaeromPark
"""

import os,sys
os.chdir('C:/Users/SaeromPark/Documents/Python Scripts/doc2vec_test')

#from doc2vec_revised import *
#import word2vec
import pickle
import numpy as np

#import copy
from time import time 

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
import pandas as pd
from sklearn import manifold
from sklearn.decomposition import PCA
#from sklearn.preprocessing import scale
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

import matplotlib.pyplot as plt
import random

import copy



data_path = '../docvec_data'
category = 'electronics'
unla_ratio = [0.3,0.4,0.5,0.6,0.7]
rr = unla_ratio[3]
#bal_num = round(len_text*rr)

"""
#Doc2vec_load
from doc2vec_revised import *
file_name = data_path + '/doc2vec_dbow_neighbor_'+category+'_'+str(2*bal_num)+'_beta_'+str(be) +'.doc2vec'
model = Doc2Vec.load(file_name)
    
words = ["funny",'boring','cheap','expensive','terrible']
word_vectors = []
for wr in words:
    wr_idx = model.vocab[wr].index
    word_vectors.append(model.syn0[wr_idx])

"""


beta = [0.01,0.02,0.03,0.05,0.1]
docvecs_list = []
for be in beta:
    f = open(data_path+'/results/'+category+'_dbow_neighbor_'+str(rr)+'_beta_'+str(be)+'_data.pickle','rb')
    docvecs = pickle.load(f)
    f.close()
    docvecs_list.append(docvecs)

beta_idx = 4
be = beta[beta_idx]
docvecs = docvecs_list[beta_idx]

"""
docvecs_list = []
for rr in unla_ratio:
    f = open(data_path+'/results/'+category+'_'+str(rr)+'_dbow_data.pickle','rb')
    docvecs = pickle.load(f)
    f.close()
    docvecs_list.append(docvecs)

r_idx = 3
rr = unla_ratio[r_idx]
docvecs = docvecs_list[r_idx]
"""




doc_vec = []
for doc in docvecs['docvec']:
    doc_vec.append(list(doc))
doc_vec = np.array(doc_vec)

true_label = docvecs['true_label']
train_label = np.array(docvecs['train_label'],copy = True)

train_idx = (train_label !=0)
train_data = copy.deepcopy(doc_vec[train_idx,:])
trainy = copy.deepcopy(true_label[train_idx])

test_idx = (train_label ==0)
test_data = copy.deepcopy(doc_vec[test_idx,:])
testy = copy.deepcopy(true_label[test_idx])

testY = copy.deepcopy(testy)



#we want to train with only unlabeled data

trainx, testx, trainy, testy = train_test_split(test_data,testY,test_size = 0.3,random_state=3)


pca = PCA(n_components = 2)
trainx = pca.fit_transform(trainx)
testx = pca.transform(testx)
plt.scatter(testx[:,0],testx[:,1],c = testy)
#plt.savefig(data_path+'/results/plots/'+category+'_dbow_'+str(rr)+'_figure.png')



"""
doing SVM!
"""
"""
trainx = train_data
testx = test_data
"""

scaler = StandardScaler()
trainx = scaler.fit_transform(trainx)
#testx = X_iso_test
testx = scaler.transform(testx)

print('training phase')
clf = svm.SVC(C = 10,gamma = 0.01)
clf.fit(trainx, trainy)

# predict phase
print('test phase')
predicts = clf.predict(testx)

# measure function


print( 'svm: measure phase')
print( confusion_matrix(testy, predicts))
print( f1_score(testy, predicts, average=None))
print( precision_score(testy, predicts, average=None))
print( recall_score(testy, predicts, average=None))
print( accuracy_score(testy, predicts))

"""
Logistic Regression
"""
from sklearn import linear_model

regr = linear_model.LogisticRegression()
regr.fit(trainx,trainy)
predicts = regr.predict(testx)
print( 'logistic regression: measure phase')
print( confusion_matrix(testy, predicts))
print( f1_score(testy, predicts, average=None))
print( precision_score(testy, predicts, average=None))
print( recall_score(testy, predicts, average=None))
print( accuracy_score(testy, predicts))