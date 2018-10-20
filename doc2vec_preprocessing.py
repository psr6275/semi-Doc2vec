# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 20:29:07 2015

@author: slcf
"""

import gensim, logging
import pickle
import gensim.models.doc2vec as d2v
import nltk.corpus as nc
import nltk.stem as nsl
import re
#import re


stops = set(nc.stopwords.words("english"))
st = nsl.PorterStemmer()


def rev_to_words(rev_line,stps):
    ## from stopping to stemming
    #1. remove non_letters
    letters_only = re.sub("[^a-zA-Z]"," ", rev_line)
    #2. convert to lower case, and split
    words = letters_only.lower().split()
    #3. stopword remove!
    meaningful_words = [w for w in words if not w in stps]
    #4. return the result
    return(meaningful_words)



f  = open('books_processed_positive.txt','rb')
text = pickle.load(f)
f.close()

documents = [];
for document in text:
    document.replace('!','.').replace('?','.')
    documents.append(document.split('.'))

data2 =[]
data3 = []
## stopping
for document in documents:
    for line in document:
        data2.append(rev_to_words(line,stops))

## stemming
for line2 in data2:
    temp = ""
    for word in line2:
        temp = temp + " " + st.stem(word) 
    data3.append(temp)
        
