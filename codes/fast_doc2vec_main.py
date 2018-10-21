# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:30:58 2016

@author: SaeromPark
"""

#import gensim
#from gensim.models.doc2vec import *
import os,sys

cwd = os.getcwd()

from doc2vec_revised4 import *
import word2vec
import pickle
import numpy as np
import random
import re
import copy
import nltk
from time import time
import pandas as pd

from IPython import embed

data_path ='/data/doc2vec_data/data/amazon_data/'
#year = '2014'
category = "kitchen"
train_type = "dbow"# you can select "dbow or semi-dbow"
f = open(data_path+category+'_processed_unlabeled.txt','rb')
texts = pickle.load(f)
f.close()

f = open(data_path+category+'_processed_unlabeled_label.txt','rb')
labels = pickle.load(f)
f.close()

"""
f = open('../data/yelp_reviews_text_'+year,'rb')
texts = pickle.load(f)
f.close()

f = open('../data/yelp_reviews_label_'+year+'.pickle','rb')
labels = pickle.load(f)
f.close()
"""
texts = np.array(texts)
labels = np.array(labels)
print(category)
#aa = random.shuffle(list(range(labels.shape[0])))
aa = list(range(labels.shape[0]))
random.shuffle(aa)
texts = texts[aa]
labels = labels[aa]
nega_idx = labels<3
posi_idx = labels>3
nega_text = list(texts[nega_idx])
posi_text = list(texts[posi_idx])
len_text = min(len(nega_text),len(posi_text))
#semi_ratio = [0.3,0.4,0.5,0.6,0.7] # the ratio of unlabeled data is 0.3
rr = 0.5
unl_num = round(rr*len_text)
print("amazon: ",category)
print("The ratio of data: ",rr)
print("The number of unlabeled data: ",unl_num)

#variable delete!!
del texts

text = posi_text[unl_num:len_text] + nega_text[unl_num:len_text] + posi_text[:unl_num]+ nega_text[:unl_num]

label = list(labels[posi_idx][unl_num:len_text]) + list(labels[nega_idx][unl_num:len_text]) + list(labels[posi_idx][:unl_num])+list(labels[nega_idx][:unl_num])
print("The number of documents: ",len(label))
#text = posi_text + nega_text + unla_text[:1000]

documents = []
for document in text:
    temp = document.replace('!','.').replace('?','.').replace(';','.').replace(':','.').replace('\n',' ').strip()
    documents.append(temp.split('.'))

del text
# we need to make the setences as lower cases or only letters!   

true_labels = label 
doc_labels = np.array(label,copy = True)
doc_labels[:-2*unl_num] = 0 
##should shuffle the training data!!
random.shuffle(aa)
doc_labels = doc_labels(aa)
documents = list(np.array(documents)[aa])
doc_labels = list(doc_labels)
true_labels = list(np.array(label)[aa])

labels = {'train_label':doc_labels,'true_label':true_labels}
labels = pd.DataFrame(labels)


f = open(data_path+category+'_semi_ratio_'+str(rr)+'_label.txt','wb')
pickle.dump(labels,f)
f.close()

sentences = []
"""
for uid, doc in enumerate(documents):
    for sen in doc:
        sen.lower()
        sen = re.sub("[^a-zA-Z]"," ", sen)
        if uid < len_text-unl_num:
            sentence = TaggedDocument(words = sen.split(),tags = ['POS_%s' %uid])
        elif uid <2*(len_text-unl_num):
            sentence = TaggedDocument(words = sen.split(),tags = ['NEG_%s' %(uid-len_text+unl_num)])
        else:
            sentence = TaggedDocument(words = sen.split(),tags = ['UNLA_%s' %(uid-2*(len_text-unl_num))])
        sentences.append(sentence)
"""
for uid, doc in enumerate(documents):
    for sen in doc:
        sen = re.sub("[^a-zA-Z]"," ",sen)
        sentence = TaggedDocument(words = sen.split(),tags = ['DOC_%s' %uid])
        sentences.append(sentence)

print("length of sentences = ",len(sentences))
print("start to training")       

del documents

d_size = 200
be = 0.05

##pretrain the doc2vec model or train the original dbow model
print("start to training dbow model")
t1 = time()
if train_type is "semi-dbow":
    model_dbow = Doc2Vec(sentences, doc_labels, size = d_size, window=3, min_count=3, workers = 8, dm = 0,iter = 5)
    t2 = time()
elif train_type is "dbow":
    model_dbow = Doc2Vec(sentences, doc_labels, size = d_size, window=3, min_count=3, workers = 8, dm = 0,iter = 10)
    model_dbow.save(data_path+"dbow_"+category+"_ratio0.5_beta0.05_nb10.doc2vec")
    
    t2 = time()
else:
    print("You should specify your train_type among semi-dbow or dbow")
    t2 = t1

##caculate neighbor_graph
print("start to construct neighbor_graph")
num_doc = model_dbow.docvecs.doctag_syn0.shape[0]
num_neighbor = 10
senti_graph = np.zeros(shape = (num_doc,num_neighbor))
for i in range(num_doc):
    senti_graph[i,:] = np.array([int(j.split('_')[1]) for j,k in model_dbow.docvecs.most_similar(i,topn=num_neighbor)])
t3 = time()
## train the neighbor_dbow model
if train_type = "semi-dbow":
    model_neighbor = Doc2Vec(sentences, doc_label, size = d_size, window = 3, min_count = 3, workers = 8, nb_list = senti_graph, beta = be,docvecs = model_dbow.docvecs,dm = 0,dbow_neighbors = 1,jter = 5) 
    t4 = time()
    print("complete training model_neighbor")
else:
    print("not for model neighbor")
    t4 = time()

print("training time for dbow =",t2-t1)

## recalculate the neighbor_graph
if train_type = "semi-dbow":
    new_graph = np.zeros(shape = (num_doc,num_neighbor))
    for i in range(num_doc):
        new_graph[i:1] = np.array([int(j.split('_')[1]) for j,k in model_neighbor.docvecs.most_similar(i,topn = num_neighbor)])
    t5 = time()
    model_dbow.save(data_path+'pre-dbow_'+category+'_ratio0.5_beta0.05_nb10.doc2vec')
    model_neighbor.save(data_path+'semi-dbow_'+category+'_ratio0.5_beta0.05_nb10.doc2vec')
    doctag = list(model_neighbor.docvecs.doctag_syn0)
    pre_doctag = list(model_dbow.docvecs.doctag_syn0)
    doc2vec = {'train_label':doc_label,'true_label':true_label,'docvec':doctag,'pre_graph':senti_graph, 'post_graph':new_graph,'pre_doctag':pre_doctag}
    doc2vec = pd.DataFrame(doc2vec)
    f = open(data_path+'semi-dbow_'+category+'_ratio0.5_beta0.05_nb10_data.pickle','wb')
    pickle.dump(doc2vec,f)
    f.close()
    print("training pre_model:",t2-t1)
    print("obtaining pre_graph:",t3-t2)
    print("training senti_model:",t4-t3)
    print("constructing post_graph:",t5-t4)
else:
    doctag = list(model_dbow.docvecs.doctag_syn0)
    doc2vec = {'train_label':doc_label,'true_label':true_label,'docvec':doctag,'senti_graph':senti_graph}
    doc2vec = pd.DataFrame(doc2vec)

    f = open(data_path+"dbow_"+category+"_ratio0.5_beta0.05_nb10_data.pickle",'wb')
    pickle.dump(doc2vec,f)
    f.close()
    print("training dbow_model:",t2-t1)
    print("obtaining senti_graph:",t3-t2)

"""
beta = [0.01,0.02,0.03,0.05,0.1]
for be in beta:
#embed()
    print('beta = ',be)
    model_neighbor = Doc2Vec(sentences,doc_labels,beta = be,learn_unlabel = 0, size=d_size, window = 3, min_count=3, workers = 8, dbow_neighbors = 1,iter = 10)
    

    file_name = '../results/yelp_'+year+'_doc2vec_dbow_neighbor_'+str(rr)+'_beta_'+str(be)+ '.doc2vec'
    model_neighbor.save(file_name)

    doctag = list(model_neighbor.docvecs.doctag_syn0)
    doc2vec = {'train_label':doc_labels,'true_label':true_labels,'docvec':doctag}

    doc2vec = pd.DataFrame(doc2vec)

    f = open('../results/yelp_'+year+'_doc2vec_dbow_neighbor_'+str(rr)+'_beta_'+str(be)+'_data.pickle','wb')
    pickle.dump(doc2vec,f)
    f.close()

#embed()
"""
