# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 20:06:55 2017

@author: SaeromPark
"""

#from gensim.models.doc2vec import *
import os
os.chdir('C:/Users/SaeromPark/Documents/Python Scripts/doc2vec_test')
from doc2vec_senti_graph import *
import numpy as np
import scipy.io as sio
import pickle
import re

category = 'electronics'
rr = 0.3
data_path2 = '../docvec_data/results3/'

f = open(data_path2+category+'_'+str(rr)+'_texts.pickle','rb')
texts_df = pickle.load(f)
f.close()
text = texts_df['text']
doc_label = texts_df['train_label']
true_label = texts_df['true_label']

documents = []
for document in text:
    temp = document.replace('!','.').replace('?','.').replace(';','.').replace(':','.').replace('\n',' ').strip()
    documents.append(temp.split('.'))

#del text
# we need to make the setences as lower cases or only letters!   

sentences = []
#sentences_label = {}
#pos_idx = 0
#neg_idx = 0
#unla_idx = 0
for uid, doc in enumerate(documents):
    
    for sen in doc:
        sen = sen.lower()
        sen = re.sub("[^a-zA-Z]"," ", sen)
        #if uid < len_text-bal_num:
        sentence = TaggedDocument(words = sen.split(),tags = ['DOC_%s' %uid])
        #sentences_label['POS_'+str(uid)] = uid
        #elif uid <2*(len_text-bal_num):
        #    sentence = TaggedDocument(words = sen.split(),tags = ['NEG_%s' %(uid-len_text+bal_num)])
        #   sentences_label['NEG_'+str(uid-len_text+bal_num)] = uid
        #else:
        #    sentence = TaggedDocument(words = sen.split(),tags = ['UNLA_%s' %(uid-2*(len_text-bal_num))])
        #    sentences_label['UNLA_'+str(uid-2*(len_text-bal_num))] = uid
        sentences.append(sentence)
        
del documents

d_size = 200
print("the number of sentences = ", len(sentences))

#beta = [0.01,0.02,0.03,0.05,0.1]
be = 0.05

#original_dbow = Doc2Vec(sentences, doc_label,size = d_size, dm=0,window=3, min_count=3, workers = 8,iter=10 )
original_model = Doc2Vec.load(data_path2+'doc2vec_dbow_'+category+'_'+str(rr)+'.doc2vec')
num_doc = original_model.docvecs.doctag_syn0.shape[0]
num_neighbor = 10
senti_graph = np.zeros(shape = (num_doc,num_neighbor))
for i in range(num_doc):
    senti_graph[i,:] = np.array([int(j.split('_')[1]) for j,k in original_model.docvecs.most_similar(i,topn=num_neighbor)])





model_neighbor = Doc2Vec(sentences,doc_label, senti_graph, dm = 1, dbow_graph=1,beta = be, size=d_size, window = 3, min_count=3, workers = 8, iter = 10)


file_name = data_path2 +category+'_dbow_graph_'+str(rr)+'_beta_'+str(be) +'.doc2vec'    
model_neighbor.save(file_name)      

doctag = list(model_neighbor.docvecs.doctag_syn0)
doc2vec = {'train_label':doc_label,'true_label':true_label,'docvec':doctag}

doc2vec = pd.DataFrame(doc2vec)


f = open(data_path2+category+'_dbow_graph_'+str(rr)+'_beta_'+str(be)+'_data.pickle','wb')
pickle.dump(doc2vec,f)
f.close()

new_senti_graph = np.zeros(shape = (num_doc,num_neighbor))
for l in range(num_doc):
    new_senti_graph[l,:] = np.array([int(j.split('_')[1]) for j,k in model_neighbor.docvecs.most_similar(l,topn=num_neighbor)])
    
    
print([true_label[l] for l in new_senti_graph[5,:]])
print([true_label[l] for l in senti_graph[5,:]])
