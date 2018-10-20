# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:30:58 2016

@author: SaeromPark
"""

#import gensim
#from gensim.models.doc2vec import *
import os,sys
import itertools
cwd = os.getcwd()

from doc2vec_revised import *
import word2vec
import pickle
import numpy as np
import random
import re
import copy
import nltk

import pandas as pd

from IPython import embed

data_path = '../data/'
#year = '2006'
category = 'kitchen'
nbs_k = [3, 10, 30]
semi_ratio = [0.3,0.5,0.7] #this is the ration of unnlabeled texts
rr = semi_ratio[2]

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
print('semi_ratio = ',rr)
aa = random.shuffle(list(range(labels.shape[0])))
texts = texts[aa]
labels = labels[aa]
nega_idx = labels<3
posi_idx = labels>3
nega_text = list(texts[nega_idx])
posi_text = list(texts[posi_idx])
len_text = min(len(nega_text),len(posi_text))
#semi_ratio = [0.3,0.4,0.5,0.6,0.7] # the ratio of unlabeled data is 0.3
unl_num = round(rr*len_text)
#print("yelp ",year)
print("The ratio of data: ",rr)
print("The number of unlabeled data: ",unl_num)
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
# we need to make the sentences as lower cases or only letters!   

true_labels = label 
doc_labels = np.array(label,copy = True)
doc_labels[:-2*unl_num] = 0 
doc_labels = list(doc_labels)
labels = {'train_label':doc_labels,'true_label':true_labels}
labels = pd.DataFrame(labels)

f = open('../data/amazon'+category+'_semi_ratio_'+str(rr)+'_label.txt','wb')
pickle.dump(labels,f)
f.close()

sentences = []
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
print("length of sentences = ",len(sentences))
print("start to training")       
del documents
d_size = 200

beta = [0.01,0.1]
#beta = [0.01,0.02,0.03,0.05,0.1]
#for be in beta:
#embed()
for itl in itertools.product(beta, nbs_k):
    be = itl[0]
    nbs = itl[1]
    print('beta = ',be)
    print('number of neighbors = ',nbs)
    print('doc_labels must be array!!')
    doc_labels = np.array(doc_labels)
    model_neighbor = Doc2Vec(sentences,doc_labels,beta = be,learn_unlabel = 0, size=d_size, window = 3, min_count=3, workers = 8, dbow_neighbor = 1,iter = 10,num_neighbor = nbs)

    

    #file_name = '../results/yelp_'+year+'_doc2vec_dbow_neighbor_'+str(rr)+'_beta_'+str(be)+ '.doc2vec'
    file_name = '../results/amazon'+category+'doc2vec_dbow_neighbor'+str(rr)+'_beta_'+str(be)+'_nbs_'+str(nbs)
    model_neighbor.save(file_name+'.docvec')

    doctag = list(model_neighbor.docvecs.doctag_syn0)
    doc2vec = {'train_label':doc_labels,'true_label':true_labels,'docvec':doctag}

    doc2vec = pd.DataFrame(doc2vec)

    f = open(file_name+'_data.pickle','wb')
    pickle.dump(doc2vec,f)
    f.close()

#embed()

#model_neighbor.save('results/doc2vec_dbow_neighbor_yelp_test_beta0.025.doc2vec')        
