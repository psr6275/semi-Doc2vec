# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 08:51:40 2016

@author: SaeromPark
"""

import os,sys
os.chdir('C:/Users/SaeromPark/Documents/Python Scripts/doc2vec_test')

from doc2vec_revised import *
import word2vec
import pickle
import numpy as np

import copy
from time import time 

import pandas as pd
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import matplotlib.pyplot as plt

data_path = '../docvec_data'
model = Doc2Vec.load('doc2vec_dm_mean_kitchen.doc2vec')
model = Doc2Vec.load('doc2vec_dbow_neighbor_kitchen_5000_laern_unlabel_beta0.025.doc2vec')

doc_vec = model.docvecs.doctag_syn0
test_data = doc_vec[-5000:,:]

f = open(data_path+'/kitchen_train_label_5000_unlabel.txt','rb')
labels = pickle.load(f)
f.close()

label_unla = np.array(labels['true_label'],copy = True)


"""
label_unla = np.array(unla_label[:1000])
aa = np.array(ones(len(label_unla)))
aa[label_unla<3] = -1
label = model.doc_label

visual_label = label
visual_label[2000:] = aa
"""
#doc_labels = np.ones(len(doc_vec))
#doc_labels[len_text-1000:-2000] = -1
#doc_labels[-2000:]= 0


pca = PCA(n_components = 2)
x = pca.fit_transform(doc_vec[:-2000,])
testx = pca.transform(test_data)
plt.scatter(testx[:,0],testx[:,1],c = label_unla[-2000:])

print("Computing LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=2,
                                      method='standard')
t0 = time()
x = clf.fit_transform(doc_vec[:-2000])
testx = clf.transform(test_data)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
print(time()-t0)
plt.scatter(x[:,0],x[:,1],c = label_unla)
plt.scatter(testx[:,0],testx[:,1],c = label_unla[-2000:])

x = pca.fit_transform(doc_vec)
plt.scatter(x[:,0],x[:,1],c = label_unla)
plt.scatter(x[-5000:,0],x[-5000:,1],c = label_unla[-5000:])