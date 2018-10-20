import pickle
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import os
os.chdir("../codes")
#import matplotlib.pyplot as plt
import random

import copy
import numpy as np

from time import time

data_path = '../results/amazon'

year = '2013'
semi_ratio = [0.3,0.4,0.5,0.6,0.7]
rr = semi_ratio[1]
print("semi_ratio = ",str(rr))

beta = [0.01, 0.02,0.03,0.05,0.1]
docvec_list = []
for be in beta:
    f = open(data_path+'/books_dbow_neighbor_'+str(rr)+'_beta_'+str(be)+"_data.pickle",'rb')
    docvecs = pickle.load(f)
    f.close()
    docvec_list.append(docvecs)   

for be_idx, be in enumerate(beta):
    docvecs = docvec_list[be_idx]
    doc_vec = []
    for doc in docvecs['docvec']:
        doc_vec.append(list(doc))
    doc_vec = np.array(doc_vec)
    true_label = np.array(docvecs['true_label'])
    train_label = np.array(docvecs['train_label'],copy = True)
    
    train_idx = (train_label !=0)
    train_data = copy.deepcopy(doc_vec[train_idx,:])
    trainY = copy.deepcopy(true_label[train_idx])

    test_idx = (train_label ==0)
    test_data = copy.deepcopy(doc_vec[test_idx,:])
    testY = copy.deepcopy(true_label[test_idx])
 
    trainx, testx, trainy, testy = train_test_split(test_data,testY,test_size =0.3,random_state = 3)
    pca = PCA(n_components = 2)
    trainx = pca.fit_transform(trainx)
    test_pca = pca.transform(test_data)

    clf = svm.SVC(C = 10,gamma = 0.01)
    scores = cross_val_score(clf,test_data,testY, cv = 5)
    print("SVM"+str(year)+" beta"+str(be)+":",np.average(scores))
    regr = linear_model.LogisticRegression()
    scores2 = cross_val_score(regr,test_data,testY, cv = 5)
    print("Logistic Regression"+str(year)+" beta"+str(be)+":",np.average(scores2))

    clf = svm.SVC(C = 10,gamma = 0.01)
    scores = cross_val_score(clf, test_pca,testY,cv=10)
    print("SVM"+str(year)+" beta"+str(be)+" PCA:",np.average(scores))
    regr = linear_model.LogisticRegression()
    scores2 = cross_val_score(regr,test_pca,testY, cv = 5)
    print("Logistic Regression"+str(year)+" beta"+str(be)+" PCA:",np.average(scores2))

