import is, sys
#os.chdir
os.getcwd()

import pickle
import numpy as np

from time import time
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

import pandas as pd
from sklearn import manifold
from sklearn.decomposition import PCA

from sklearn import svm
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

import matplotlib.pyplot as plt
import random

import copy


data_path = '../data'
category = 'kitchen'
unla_ratio = [0.3,0.4,0.5,0.6,0.7]
rr = unla_ratio[0]


beta = [0.01,0.02,0.03,0.05,0.1]
docvecs_list = []

for be in beta:
    f = open(data_path+'/results/'+category+'_dbow_neighbor_'+str(rr)+'_beta_'+str(be)+'_data.pickle')
    docvecs = pickle.load(f)
    f.close()
    docvecs_list.append(docvecs)

beta_idx = 4
be = beta[beta_idx]
docvecs = docvecs_list[beta_idx]

docvec = []

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


trainx, testx, trainy, testy = train_test_split(test_data,testY, test_size = 0.3,random_state=3)

pca = PCA(n_components = 2)
trainx = pca.fit_transform(trainx)
testx = pca.transform(testx)
plt.scatter(testx[:,0],testx[:,1],c = testy)
plt.savefig(data_path+'/results/plots/'+'category'+'_dbow_neighbor_'+str(rr)+'_beta_'+str(be)+'_figure.png')

"""
doing SVM!!
"""

scaler = StandardScaler()
trainx = scaler.fit_transform(trainx)

testx = scaler.transform(testx)

print('training phase')
clf = svm.SVC(C = 10,gamma = 0.01)
clf.fit(trainx,trainy)

print('test phase')
predicts = clf.predict(testx)

print('svm: measure phase')
print(confusion_matrix(testy,predicts))
print(accuracy_score(testy,predicts))
