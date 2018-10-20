# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:35:01 2016

@author: SaeromPark
"""

import os,sys
#os.chdir('C:/Users/SaeromPark/Documents/Python Scripts/doc2vec_test')

from doc2vec_revised import *
import word2vec
import pickle
import numpy as np

import re
import copy
import nltk

import pandas as pd
from time import time


data_path = '../data'
category = 'electronics'

f = open(data_path+'/'+category+'_processed_unlabeled.txt.txt','rb')
unla_text = pickle.load(f)
f.close()

f = open(data_path+'/'+category+'_processed_unlabeled_label.txt.txt','rb')
unla_label = pickle.load(f)
f.close()

"""

#if category == 'books':
#    unla_text = unla_text[:150000]
#    unla_label = unla_label[:150000]
unla_text = np.array(unla_text)
unla_label = np.array(unla_label)
nega_idx = unla_label<3
posi_idx = unla_label>3
nega_text = list(unla_text[nega_idx])
posi_text = list(unla_text[posi_idx])
len_text = min(len(nega_text),len(posi_text))

unla_ratio = [0.3,0.4,0.5,0.6,0.7]
rr = unla_ratio[2]
bal_num = round(len_text*rr)
print("the category of data")
print(category)
print("the ratio of unlabeled data")
print(rr)
print("the number of unlabeled data")
print("total:"+str(len_text)+", and unlabeled:"+str(bal_num))

#text = posi_text[1000:len_text] + nega_text[1000:len_text] + posi_text[:1000]+ nega_text[:1000]
text = posi_text[bal_num:len_text] + nega_text[bal_num:len_text] + posi_text[:bal_num]+ nega_text[:bal_num]

doc_labels = np.ones(len(text))
doc_labels[len_text-bal_num:-2*bal_num] = -1
doc_labels[-2*bal_num:]= 0
true_labels = copy.deepcopy(doc_labels)
true_labels[-2*bal_num:-bal_num] = 1
true_labels[-bal_num:] = -1

del unla_text 
"""

## Start of the shuffled text
texts = np.array(unla_text)
labels= np.array(unla_label)

aa = list(range(len(texts)))
random.shuffle(aa)
texts = texts[aa]
labels = np.array(labels[aa])

nega_idx = labels<3
posi_idx = labels>3
labels[nega_idx] = -1
labels[posi_idx] = 1
nega_text = list(texts[nega_idx])
posi_text = list(texts[posi_idx])
len_text = min(len(nega_text),len(posi_text))
#semi_ratio = 0.5
rr = 0.5
unla_num = int(rr*len_text)
print("category",category)
print("The number of data:",len_text)
print("The number of unlabeled data",unla_num)

text = posi_text[unla_num:len_text] + nega_text[unla_num:len_text] + posi_text[:unla_num]+ nega_text[:unla_num]

label = list(labels[posi_idx][unla_num:len_text]) + list(labels[nega_idx][unla_num:len_text]) + list(labels[posi_idx][:unla_num])+list(labels[nega_idx][:unla_num])
del texts

true_labels = label 
doc_labels = np.array(label,copy = True)
doc_labels[:-2*unla_num] = 0 
#doc_labels = list(doc_labels)
labels2 = {'train_label':doc_labels,'true_label':true_labels}
labels2 = pd.DataFrame(labels2)

aa = list(range(len(true_labels)))
random.shuffle(aa)
text = np.array(text)[aa]
doc_labels = doc_labels[aa]
true_labels = np.array(true_labels)[aa]

text.tolist
#text = text[0]
doc_labels.tolist
#doc_labels = doc_labels[0]
true_labels.tolist
#true_labels = true_labels[0]
####enf of shuffled text!!!
print(true_labels)

documents = []
for document in text:
    temp = document.replace('!','.').replace('?','.').replace(';','.').replace(':','.').replace('\n',' ').strip()
    documents.append(temp.split('.'))

del text
# we need to make the setences as lower cases or only letters!   

sentences = []
for uid, doc in enumerate(documents):
    for sen in doc:
        sen = sen.lower()
        sen = re.sub("[^a-zA-Z]"," ", sen)
        if uid < len_text-unla_num:
            sentence = TaggedDocument(words = sen.split(),tags = ['POS_%s' %uid])
        elif uid <2*(len_text-unla_num):
            sentence = TaggedDocument(words = sen.split(),tags = ['NEG_%s' %(uid-len_text+unla_num)])
        else:
            sentence = TaggedDocument(words = sen.split(),tags = ['UNLA_%s' %(uid-2*(len_text-unla_num))])
        sentences.append(sentence)
        
        
d_size = 200
del documents
#   Hyper Parameters
#beta = [0.03,0.05,0.1]
beta = [0.05]
#beta = [0.01,0.02,0.03,0.05,0.1]
for be in beta:
    print("start to train")
    t1 = time()
    model_neighbor = Doc2Vec(sentences,doc_labels,learn_unlabel = 0,beta = be, size=d_size, window = 3, min_count=3, workers = 8, dbow_neighbor = 0,iter = 10)
    print("end training: ",time()-t1)
    #file_name = '../results/amazon/doc2vec_dbow_neighbor2_'+category+'_'+str(rr)+'_beta_'+str(be) +'.doc2vec'    
    file_name = '../results/amazon/doc2vec_dbow_'+category+'_'+str(rr)+'_beta_'+str(be) +'.doc2vec'    
    model_neighbor.save(file_name)      

    doctag = list(model_neighbor.docvecs.doctag_syn0)
    doc2vec = {'train_label':doc_labels,'true_label':true_labels,'docvec':doctag}

    doc2vec = pd.DataFrame(doc2vec)


    #f = open('../results/amazon/'+category+'_dbow_neighbori2_'+str(rr)+'_beta_'+str(be)+'_data.pickle','wb')
    f = open('../results/amazon/'+category+'_dbow_'+str(rr)+'_beta_'+str(be)+'_data.pickle','wb')
    pickle.dump(doc2vec,f)
    f.close()

"""
startint to make Dataframe for classification!
"""
"""
doctag = list(model_neighbor.docvecs.doctag_syn0)
doc2vec = {'train_label':doc_labels,'true_label':true_labels,'docvec':doctag}

doc2vec = pd.DataFrame(doc2vec)


f = open(data_path+'/electronics_dbow_neighbor_3000_random.txt','wb')
pickle.dump(doc2vec,f)
f.close()

"""

"""
original dbow model
"""
"""

model_dbow = Doc2Vec(sentences, size=d_size, window = 3, min_count=3, workers = 8, dm = 0,iter = 10)
file_name = data_path + '/results/doc2vec_dbow_'+category+'_'+str(rr)+'.doc2vec'
model_dbow.save(file_name)      

doctag = list(model_dbow.docvecs.doctag_syn0)
doc2vec = {'train_label':doc_labels,'true_label':true_labels,'docvec':doctag}

doc2vec = pd.DataFrame(doc2vec)


f = open(data_path+'/results/'+category+'_'+str(rr)+'_dbow_data.pickle','wb')
pickle.dump(doc2vec,f)
f.close()
"""
