import os,sys
import pickle
import numpy as np
import re
import copy
import nltk
import pandas as pd
import random

data_path = '/data/doc2vec_data/data/'
result_path = '/data/doc2vec_data/result/'

year = ['2008','2010','2013']
unla_ratio = [0.3,0.4,0.5,0.6,0.7]
for yr in year:
    f = open(data_path+'yelp_reviews_text_'+yr,'rb')
    unla_text = pickle.load(f)
    f.close()
    f = open(data_path+'yelp_reviews_label_'+yr+'.pickle','rb')
    unla_label = pickle.load(f)
    f.close()

    unla_text = np.array(unla_text)
    unla_label = np.array(unla_label)

    aa = random.shuffle(list(range(unla_label.shape[0])))
    texts = unla_text[aa]
    labels = np.array(unla_label[aa])

    nega_idx = unla_label<3
    posi_idx = unla_label>3
    nega_text = list(unla_text[nega_idx])
    posi_text = list(unla_text[posi_idx])
    len_text = min(len(nega_text),len(posi_text))

    for rr in unla_ratio:
        bal_num = round(len_text*rr)
        print("year of data :",yr)
        print("the ratio of unlabeled data :",rr)

        text = posi_text[bal_num:len_text]+nega_text[bal_num:len_text]+posi_text[:bal_num]+nega_text[:bal_num]

        doc_labels = np.ones(len(text))
        doc_labels[len_text-bal_num:-2*bal_num] = -1
        doc_labels[-2*bal_num:] = 0
        true_labels = copy.deepcopy(doc_labels)
        true_labels[-2*bal_num:-bal_num] = 1
        true_labels[-bal_num:] = -1

        aa = np.array(range(0,len(text)))
        random.shuffle(aa)
        text = list(np.array(text)[aa])
        doc_labels = doc_labels[aa]
        true_labels = true_labels[aa]
        texts = {'text':text,'train_label':doc_labels,'true_label':true_labels}
        texts_df = pd.DataFrame(texts)

        f = open(data_path+'yelp_reviews_'+yr+'_'+str(rr)+'_texts.pickle','wb')
        pickle.dump(texts_df,f)
        f.close()


