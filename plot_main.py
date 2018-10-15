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
category = 'kitchen'
unla_ratio = [0.3,0.4,0.5,0.6,0.7]
rr = unla_ratio[0]
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

beta_idx = 3
be = beta[beta_idx]
docvecs = docvecs_list[beta_idx]

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
"""
aaa = list(range(len(testy)))
idx = random.shuffle(aaa)
train_idx = idx[:1000]
test_idx = idx[1000:]
trainx = test_data[train_idx,:]
testx = test_data[test_idx,:]
"""
"""
## parameter estimation using Grid function
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(trainy, n_iter=5, test_size=0.3, random_state=42)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
grid.fit(trainx, trainy)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
"""



#we want to train with only unlabeled data

trainx, testx, trainy, testy = train_test_split(test_data,testY,test_size = 0.3,random_state=3)





pca = PCA(n_components = 2)
trainx = pca.fit_transform(trainx)
testx = pca.transform(testx)
plt.scatter(testx[:,0],testx[:,1],c = testy)
plt.savefig(data_path+'/results/plots/'+category+'_dbow_neighbor_'+str(rr)+'_beta_'+str(be)+'_figure.png')

"""
print("Computing LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=2,
                                      method='standard')
t0 = time()
trainx = clf.fit_transform(trainx)
testx = clf.transform(testx)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
print(time()-t0)
#plt.scatter(x[:,0],x[:,1],c = label_unla)
plt.scatter(testx[:,0],testx[:,1],c = testy)


print("Computing Isomap embedding")
t0 = time()
#X_iso = manifold.Isomap(n_neighbors, n_components=50).fit_transform(X)
iso = manifold.Isomap(n_neighbors=10, n_components=2)
trainx = iso.fit_transform(trainx)
testx = iso.transform(testx)
print("Done.")
print( time()-t0)
plt.scatter(trainx[:,0],trainx[:,1],c = trainy)

tsne = manifold.TSNE(n_components = 3,random_state = 2)
trainx = tsne.fit_transform(trainx)
testx = tsne.transform(testx)
plt.scatter(testx[:,0],testx[:,1],c = trainy)
#tsne = manifold.TSNE(n_components= 2,random_state = 0)
#trainx = tsne.fit_transform(trainx)
#plt.scatter(trainx[:,0],trainx[:,1],c = trainy)



## parameter estimation using Grid function
C_range = np.logspace(-2, 10, 5)
gamma_range = np.logspace(-9, 3, 3)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(trainy, n_iter=5, test_size=0.3, random_state=42)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
grid.fit(trainx, trainy)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
#grid.score
"""
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