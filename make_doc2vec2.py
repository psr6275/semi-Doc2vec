# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:07:19 2015

@author: slcf
"""
import gensim
from gensim.models.doc2vec import *
#import cPickle as pickle
import numpy
#from collections import namedtuple
import nltk
import nltk.corpus as nc
import nltk.stem as nsl
import re
import scipy.io as sio
#import numpy as np
#from sklearn.manifold import TSNE

nltk.download()
stops = set(nc.stopwords.words("english"))
st = nsl.PorterStemmer()


def rev_to_words(rev_line):
    ## from stopping to stemming
    #1. remove non_letters
    letters_only = re.sub("[^a-zA-Z]"," ", rev_line)
    #2. convert to lower case, and split
    line = letters_only.lower()
    #3. stopword remove!
   # meaningful_words = [w for w in words if not w in stps]
    #4. return the result
    return(line)
    
#f  = open('books_processed_positive.txt','rb')
#text = pickle.load(f)
#f.close()


f  = open('electronics_processed_positive.txt','rb')
text_posi = pickle.load(f)
f.close()

f  = open('electronics_processed_negative.txt','rb')
text_nega = pickle.load(f)
f.close()

f  = open('electronics_processed_unlabeled.txt','rb')
text_unla = pickle.load(f)
f.close()


f = open("books_processed_unlabeled_label.txt",'rb')
unla_label = pickle.load(f)
f.close()

pos_unla = []
neg_unla = []

for idx, doc in enumerate(text_unla):
    if unla_label[idx]>3:
        pos_unla.append(doc)
    else:
        neg_unla.append(doc)
        
unla_size = len(neg_unla)
bal_unla = pos_unla[:unla_size]+neg_unla


text = text_posi+text_nega+ bal_unla
text = text_posi+text_nega+ text_unla


del text_posi, text_nega, text_unla
## save current revised data
#f = open('kitchen_revised_positive.txt','wb')
#pickle.dump(text_posi,f)
#f.close()

#f = open('kitchen_revised_negative.txt','wb')
#pickle.dump(text_nega,f)
#f.close()

#f = open('kitchen_revised_unlabeled.txt','wb')
#pickle.dump(text_unla,f)
#f.close()


#documents = [];
#for document in text:
#    temp = document.replace('!','.').replace('?','.').replace(';','.').replace(':','.').replace('\n',' ').strip()
#    documents.append(temp.split('.'))
    
    
#tokenizer
#docs_list = []
#docs_tag = []
#for document in documents:
    #tag_sen = []
#    line = ""
#    for sentence in document:
#        line = line + rev_to_words(sentence)
        #tag_sen = tag_sen + nltk.pos_tag(nltk.word_tokenize(line))
    #docs_tag.append(tag_sen)
#    docs_list.append(line)
    
    
docs_list = []
#docs_tag = []
for document in text:
    #tag_sen = []
   
    #for sentence in document:
     #   line = line + sentence
        #tag_sen = tag_sen + nltk.pos_tag(nltk.word_tokenize(line))
    #docs_tag.append(tag_sen)
    line = "NULL "*5 + document + " NULL"*5
    docs_list.append(line)


    
# make documetn vecotr
sentences = []
for uid, doc in enumerate(docs_list):
    if uid < 1000:
        sentence = LabeledSentence(words = doc.split(),labels = ['POS_%s' %uid])
    elif uid <2000:
        sentence = LabeledSentence(words = doc.split(),labels = ['NEG_%s' %(uid-1000)])
    else:
        sentence = LabeledSentence(words = doc.split(),labels = ['UNLA_%s' %(uid-2000)])
    sentences.append(sentence)


#sentences = []
#for uid, doc in enumerate(docs_list):
#    sentence = LabeledSentence(words = doc.split(),labels = ['DOC_%s' %uid])
#    sentences.append(sentence)


##make model
d_size = 400
model = Doc2Vec(sentences, size=d_size, window = 8, min_count=5, workers = 8)
"""
model2 = Doc2Vec(alpha=0.025,min_alpha=0.025, size=d_size)
model2.build_vocab(sentences)
for epoch in range(3):
    model2.train(sentences)
    model2.alpha -= 0.002
    model2.min_alpha = model2.alpha
"""



#unlb_size = len(bal_unla)
unlb_size = len(text_unla)
## save document vectors
train_arrays = numpy.zeros((2000,d_size))
train_labels = numpy.zeros(2000)

for i in range(1000):
    prefix_train_pos = 'POS_' + str(i)
    prefix_train_neg = 'NEG_' + str(i)
    prefix_train_unla = 'UNLA_' + str(i)
    
    train_arrays[i] = model2[prefix_train_pos]
    train_arrays[i+1000] = model2[prefix_train_neg]
    #train_arrays[i+2000] = model[prefix_train_unla]
    train_labels[i] = 1
    train_labels[i+1000] = -1
    #train_labels[i+2000] = 0

train_unlb = numpy.zeros((unlb_size,d_size))

for i in range(unlb_size):
   prefix_unlb = 'UNLA_'+str(i)
   try:
        train_unlb[i] = model2[prefix_unlb]
   except:
        train_unlb[i] = 100

sio.savemat("doc2vec_20151202_elec_epoch3",{"doc_vec":train_arrays}) 
sio.savemat("doc2vec_20151202_elec_unlb_epoch3",{"unl_vec":train_unlb}) 
##saving
model2.save('C:/Users/slcf/Documents/Python Scripts/sentiment/Parsed data/doc2vec/doc2vec_20151202_elec_model2_epoch3.txt')       
##model.save_word2vec_format('C:/Users/slcf/Documents/Python Scripts/sentiment/Parsed data/doc2vec/doc2vec_books_positive.txt')

#load the model back#
#model_loaded = Doc2Vec.load('C:/Users/slcf/Documents/Python Scripts/sentiment/Parsed data/doc2vec/doc2vec_books_positive.doc2vec')
 