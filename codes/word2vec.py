#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
"""
Deep learning via word2vec's "skip-gram and CBOW models", using either
hierarchical softmax or negative sampling [1]_ [2]_.

The training algorithms were originally ported from the C package https://code.google.com/p/word2vec/
and extended with additional functionality.

For a blog tutorial on gensim word2vec, with an interactive web app trained on GoogleNews, visit http://radimrehurek.com/2014/02/word2vec-tutorial/

**Make sure you have a C compiler before installing gensim, to use optimized (compiled) word2vec training**
(70x speedup compared to plain NumPy implementation [3]_).

Initialize a model with e.g.::

>>> model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)

Persist a model to disk with::

>>> model.save(fname)
>>> model = Word2Vec.load(fname)  # you can continue training with the loaded model!

The model can also be instantiated from an existing file on disk in the word2vec C format::

  >>> model = Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format
  >>> model = Word2Vec.load_word2vec_format('/tmp/vectors.bin', binary=True)  # C binary format

You can perform various syntactic/semantic NLP word tasks with the model. Some of them
are already built-in::

  >>> model.most_similar(positive=['woman', 'king'], negative=['man'])
  [('queen', 0.50882536), ...]

  >>> model.doesnt_match("breakfast cereal dinner lunch".split())
  'cereal'

  >>> model.similarity('woman', 'man')
  0.73723527

  >>> model['computer']  # raw numpy vector of a word
  array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)

and so on.

If you're finished training a model (=no more updates, only querying), you can do

  >>> model.init_sims(replace=True)

to trim unneeded model memory = use (much) less RAM.

Note that there is a :mod:`gensim.models.phrases` module which lets you automatically
detect phrases longer than one word. Using phrases, you can learn a word2vec model
where "words" are actually multiword expressions, such as `new_york_times` or `financial_crisis`:

>>> bigram_transformer = gensim.models.Phrases(sentences)
>>> model = Word2Vec(bigram_transformer[sentences], size=100, ...)

.. [1] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
.. [2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality.
       In Proceedings of NIPS, 2013.
.. [3] Optimizing word2vec in gensim, http://radimrehurek.com/2013/09/word2vec-in-python-part-two-optimizing/
"""
from __future__ import division  # py3 "true division"

#import sys
#sys.path.append(r'D:\Research\Journal\2016\8. Doc2Vec\Python\Word2Vec')
import os
#if 'slcf-wserver' in os.environ.get('USERNAME'):
#    path=r'D:\Kyoungok\Doc2Vec\Python'
#    datapath=r'D:\Kyoungok\Data\SavedSentences'
#    lexi=r'D:\Kyoungok\Doc2Vec\Data'
#    twit=r'D:\Kyoungok\Doc2Vec\Data\UMICH'
#else:
#    path=r'D:\Research\Journal\2016\8. Doc2Vec\Python\Final'
#    datapath=r'D:\Research\Journal\2015\1. Sentiment Analysis\Python\data\books\SavedSentences'
#os.chdir(path)

import logging
import heapq
from timeit import default_timer
from copy import deepcopy
from collections import defaultdict
import threading
import itertools
from utils import keep_vocab_item

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray, sign, matmul, argsort, in1d, tile, kron, eye

import utils, matutils  # utility fnc for pickling, common scipy operations etc
from six import iteritems, itervalues, string_types
from six.moves import xrange
from types import GeneratorType

logger = logging.getLogger(__name__)

try:
    from word2vec_inner import train_batch_sg, train_batch_cbow, train_batch_sswe, train_batch_sswe_bow, train_batch_cbow_sentiment
    from word2vec_inner import score_sentence_sg, score_sentence_cbow
    from word2vec_inner import FAST_VERSION, MAX_WORDS_IN_BATCH
    print("Fast Version Import Success")
    #a[11111]
except:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1
    MAX_WORDS_IN_BATCH = 10000
    
    def train_batch_sswe(model, sentences, alpha, work=None):
        result = 0
    
        for sentence, label in sentences:
            word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab and
                           model.vocab[w].sample_int > model.random.rand() * 2**32]
            nword = len(word_vocabs)
            if nword>=model.window:
                for pos in range(nword-model.window+1):
                    ngram = word_vocabs[pos:(pos+model.window)]
                    word_ind = [w.index for w in ngram]
                    train_sswe(model, label, word_ind ,alpha)
                result += len(word_vocabs)
    
        return result
        
    def train_batch_sswe_bow(model, sentences, alpha, work=None):
        result = 0
    
        for sentence, label in sentences:
            word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab and
                           model.vocab[w].sample_int > model.random.rand() * 2**32]
            nword = len(word_vocabs)
            if nword>=model.window:
                for pos in range(nword-model.window+1):
                    ngram = word_vocabs[pos:(pos+model.window)]
                    word_ind = [w.index for w in ngram]
                    train_sswe_bow(model, label, word_ind ,alpha)
                result += len(word_vocabs)
    
        return result
    
    def train_batch_sg(model, sentences, alpha, work=None):
        """
        Update skip-gram model by training on a sequence of sentences.
    
        Each sentence is a list of string tokens, which are looked up in the model's
        vocab dictionary. Called internally from `Word2Vec.train()`.
    
        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from word2vec_inner instead.
    
        """
        result = 0
        for sentence in sentences:
            word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab and
                           model.vocab[w].sample_int > model.random.rand() * 2**32]
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code
    
                # now go over all words from the (reduced) window, predicting each one in turn
                start = max(0, pos - model.window + reduced_window)
                for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
                    # don't train on the `word` itself
                    if pos2 != pos:
                        train_sg_pair(model, model.index2word[word.index], word2.index, alpha)
            result += len(word_vocabs)
    
        return result
    
    def train_batch_cbow(model, sentences, alpha, work=None, neu1=None):
        """
        Update CBOW model by training on a sequence of sentences.
    
        Each sentence is a list of string tokens, which are looked up in the model's
        vocab dictionary. Called internally from `Word2Vec.train()`.
    
        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from word2vec_inner instead.
    
        """
        result = 0
        for sentence in sentences:
            word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab and
                           model.vocab[w].sample_int > model.random.rand() * 2**32]
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code
                start = max(0, pos - model.window + reduced_window)
                window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
                word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
                l1 = np_sum(model.syn0[word2_indices], axis=0)  # 1 x vector_size
                if word2_indices and model.cbow_mean:
                    l1 /= len(word2_indices)
                train_cbow_pair(model, word, word2_indices, l1, alpha)
            result += len(word_vocabs)
    
        return result
    
    def train_batch_cbow_sentiment(model, sentences, alpha, obj_weight, work=None, neu1=None):
        """
        Update CBOW model by training on a sequence of sentences.
    
        Each sentence is a list of string tokens, which are looked up in the model's
        vocab dictionary. Called internally from `Word2Vec.train()`.
    
        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from word2vec_inner instead.
    
        """
        result = 0
        for sentence, label in sentences:
            word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab and
                           model.vocab[w].sample_int > model.random.rand() * 2**32]
            nword = len(word_vocabs)
            if nword>=model.window:
                slide=int(model.window/2)
                for pos in range(2,nword-slide):
                    ngram = word_vocabs[(pos-slide):(pos+slide+1)]
                    word_ind = [w.index for w in ngram]
                    word = word_vocabs[pos]
                    l1 = model.syn0[word_ind]
                    train_cbow_sentiment_pair(model, label, word, word_ind, l1, alpha, obj_weight)
                    result += len(word_vocabs)               
    
        return result
        
    def score_sentence_sg(model, sentence, work=None):
        """
        Obtain likelihood score for a single sentence in a fitted skip-gram representaion.
    
        The sentence is a list of Vocab objects (or None, when the corresponding
        word is not in the vocabulary). Called internally from `Word2Vec.score()`.
    
        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from word2vec_inner instead.
    
        """
    
        log_prob_sentence = 0.0
        if model.negative:
            raise RuntimeError("scoring is only available for HS=True")
    
        word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab]
        for pos, word in enumerate(word_vocabs):
            if word is None:
                continue  # OOV word in the input sentence => skip
    
            # now go over all words from the window, predicting each one in turn
            start = max(0, pos - model.window)
            for pos2, word2 in enumerate(word_vocabs[start : pos + model.window + 1], start):
                # don't train on OOV words and on the `word` itself
                if word2 is not None and pos2 != pos:
                    log_prob_sentence += score_sg_pair(model, word, word2)
    
        return log_prob_sentence
    
    def score_sentence_cbow(model, sentence, alpha, work=None, neu1=None):
        """
        Obtain likelihood score for a single sentence in a fitted CBOW representaion.
    
        The sentence is a list of Vocab objects (or None, where the corresponding
        word is not in the vocabulary. Called internally from `Word2Vec.score()`.
    
        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from word2vec_inner instead.
    
        """
        log_prob_sentence = 0.0
        if model.negative:
            raise RuntimeError("scoring is only available for HS=True")
    
        word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab]
        for pos, word in enumerate(word_vocabs):
            if word is None:
                continue  # OOV word in the input sentence => skip
    
            start = max(0, pos - model.window)
            window_pos = enumerate(word_vocabs[start:(pos + model.window + 1)], start)
            word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
            l1 = np_sum(model.syn0[word2_indices], axis=0)  # 1 x layer1_size
            if word2_indices and model.cbow_mean:
                l1 /= len(word2_indices)
            log_prob_sentence += score_cbow_pair(model, word, word2_indices, l1)
    
        return log_prob_sentence
    
def train_sswe(model, target, input_word_indices, alpha, learn_vectors=True, learn_hidden=True):
    l1 = model.syn0[input_word_indices].reshape(-1,1).flatten()
    lock_factor = model.syn0_lockf[input_word_indices]

    neu1e = zeros(l1.shape, dtype=REAL)
    #print(neu1e.shape, l1.shape)

    w1 = model.nn_parms['w1']
    b1 = model.nn_parms['b1']
    w2 = model.nn_parms['w2']
    b2 = model.nn_parms['b2']

    l2 = matmul(w1,l1) + b1 #t
    #print(dot(l1,w1.T).shape,l1.shape,l2.shape,w1.shape,b1.shape)
    l3 = array([x if (x>=-1) and (x<=1) else sign(x) for x in l2]) #hTanh(t)
    l4 = (matmul(w2,l3) + b2).flatten() # z
    l5 = 1./(1.+exp(-l4)) # logit(z)

    neuoe = l5-target
    #print("first:",input_word_indices,neuoe,l5,target)
    neu2e = matmul(w2.T, neuoe).flatten()
    #print("neu2e",neu2e)
    neuhte = neu2e.copy()
    neuhte[(l2<-1)|(l2>1)] = 0
    #print("neuhte",neuhte)
    neu1e = neu1e - alpha * matmul(w1.T, neuhte).flatten()
    #print("neu1e",neu1e)
    #print(neuoe.shape,w2_mask.shape,w2_oe.shape,neu1e.shape)
    
    #print("1:",target,l4,l5,neuoe)

    w2 = w2 - alpha * matmul(neuoe.reshape(-1,1),l3.reshape(1,-1))
    b2 = b2 - alpha * neuoe
    w1 = w1 - alpha * matmul(neuhte.reshape(-1,1),l1.reshape(1,-1))
    b1 = b1 - alpha * neuhte

    if learn_hidden:
        model.nn_parms['w1'] = w1
        model.nn_parms['b1'] = b1
        model.nn_parms['w2'] = w2
        model.nn_parms['b2'] = b2    
    
    #l2 = matmul(w1,l1) + b1 #t
    #print(dot(l1,w1.T).shape,l1.shape,l2.shape,w1.shape,b1.shape)
    #l3 = array([x if (x>=-1) and (x<=1) else sign(x) for x in l2]) #hTanh(t)
    #l4 = (matmul(w2,l3) + b2).flatten() # z
    #l5 = 1./(1.+exp(-l4)) # logit(z)
    
    #print("second:",input_word_indices,l5-target,l5,target)
    
    if learn_vectors:
        neu1e = neu1e.reshape(len(input_word_indices),-1)
        for pos,i in enumerate(input_word_indices):
            model.syn0[i] += neu1e[pos] * lock_factor[pos]
        neu1e = neu1e.reshape(l1.shape)

    #l2 = matmul(w1,l1) + b1 #t
    #print(dot(l1,w1.T).shape,l1.shape,l2.shape,w1.shape,b1.shape)
    #l3 = array([x if (x>=-1) and (x<=1) else sign(x) for x in l2]) #hTanh(t)
    #l4 = (matmul(w2,l3) + b2).flatten() # z
    #l5 = 1./(1.+exp(-l4)) # logit(z)

    #print("third:",input_word_indices,l5-target,l5,target)  
    return neu1e
   
def train_sswe_bow(model, target, input_word_indices, alpha, learn_vectors=True, learn_hidden=True):
    l1 = model.syn0[input_word_indices]
    l1 = l1.mean(0).flatten()
    lock_factor = model.syn0_lockf[input_word_indices]

    neu1e = zeros(l1.shape, dtype=REAL)
    #print(neu1e.shape, l1.shape)

    w1 = model.nn_parms['w1']
    b1 = model.nn_parms['b1']
    w2 = model.nn_parms['w2']
    b2 = model.nn_parms['b2']

    l2 = matmul(w1,l1) + b1 #t
    #print(dot(l1,w1.T).shape,l1.shape,l2.shape,w1.shape,b1.shape)
    l3 = array([x if (x>=-1) and (x<=1) else sign(x) for x in l2]) #hTanh(t)
    l4 = (matmul(w2,l3) + b2).flatten() # z
    l5 = 1./(1.+exp(-l4)) # logit(z)

    neuoe = l5-target
    #print("first:",input_word_indices,neuoe,l5,target)
    neu2e = matmul(w2.T, neuoe).flatten()
    #print("neu2e",neu2e)
    neuhte = neu2e.copy()
    neuhte[(l2<-1)|(l2>1)] = 0
    #print("neuhte",neuhte)
    neu1e = neu1e - alpha * matmul(w1.T, neuhte).flatten()
    #print("neu1e",neu1e)
    #print(neuoe.shape,w2_mask.shape,w2_oe.shape,neu1e.shape)
    
    #print("1:",target,l4,l5,neuoe)

    w2 = w2 - alpha * matmul(neuoe.reshape(-1,1),l3.reshape(1,-1))
    b2 = b2 - alpha * neuoe
    w1 = w1 - alpha * matmul(neuhte.reshape(-1,1),l1.reshape(1,-1))
    b1 = b1 - alpha * neuhte

    if learn_hidden:
        model.nn_parms['w1'] = w1
        model.nn_parms['b1'] = b1
        model.nn_parms['w2'] = w2
        model.nn_parms['b2'] = b2    
    
    #l2 = matmul(w1,l1) + b1 #t
    #print(dot(l1,w1.T).shape,l1.shape,l2.shape,w1.shape,b1.shape)
    #l3 = array([x if (x>=-1) and (x<=1) else sign(x) for x in l2]) #hTanh(t)
    #l4 = (matmul(w2,l3) + b2).flatten() # z
    #l5 = 1./(1.+exp(-l4)) # logit(z)
    
    #print("second:",l5-target)
    
    if learn_vectors:
        for pos,i in enumerate(input_word_indices):
            model.syn0[i] += neu1e * lock_factor[pos]

    return neu1e

def train_sg_pair(model, word, context_index, alpha, learn_vectors=True, learn_hidden=True,
              context_vectors=None, context_locks=None):
    if context_vectors is None:
        context_vectors = model.syn0
    if context_locks is None:
        context_locks = model.syn0_lockf

    if word not in model.vocab:
        return
    predict_word = model.vocab[word]  # target word (NN output)

    l1 = context_vectors[context_index]  # input word (NN input/projection layer)
    lock_factor = context_locks[context_index]

    neu1e = zeros(l1.shape)

    if model.hs:
        # work on the entire tree at once, to push as much work into numpy's C routines as possible (performance)
        l2a = deepcopy(model.syn1[predict_word.point])  # 2d matrix, codelen x layer1_size
        fa = 1.0 / (1.0 + exp(-dot(l1, l2a.T)))  # propagate hidden -> output
        ga = (1 - predict_word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1[predict_word.point] += outer(ga, l1)  # learn hidden -> output
        neu1e += dot(ga, l2a)  # save error

    if model.negative:
        # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
        word_indices = [predict_word.index]
        while len(word_indices) < model.negative + 1:
            w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
            if w != predict_word.index:
                word_indices.append(w)
        l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
        fb = 1. / (1. + exp(-dot(l1, l2b.T)))  # propagate hidden -> output
        gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1neg[word_indices] += outer(gb, l1)  # learn hidden -> output
        neu1e += dot(gb, l2b)  # save error

    if learn_vectors:
        l1 += neu1e * lock_factor  # learn input -> hidden (mutates model.syn0[word2.index], if that is l1)
    return neu1e


def train_cbow_pair(model, word, input_word_indices, l1, alpha, learn_vectors=True, learn_hidden=True):
    neu1e = zeros(l1.shape)

    if model.hs:
        l2a = model.syn1[word.point]  # 2d matrix, codelen x layer1_size
        fa = 1. / (1. + exp(-dot(l1, l2a.T)))  # propagate hidden -> output
        ga = (1. - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1[word.point] += outer(ga, l1)  # learn hidden -> output
        neu1e += dot(ga, l2a)  # save error

    if model.negative:
        # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
        word_indices = [word.index]
        while len(word_indices) < model.negative + 1:
            w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
            if w != word.index:
                word_indices.append(w)
        l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
        fb = 1. / (1. + exp(-dot(l1, l2b.T)))  # propagate hidden -> output
        gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1neg[word_indices] += outer(gb, l1)  # learn hidden -> output
        neu1e += dot(gb, l2b)  # save error

    if learn_vectors:
        # learn input -> hidden, here for all words in the window separately
        if not model.cbow_mean and input_word_indices: #if we will not train word vector, this indices are useless
            neu1e /= len(input_word_indices)
        for i in input_word_indices:
            model.syn0[i] += neu1e * model.syn0_lockf[i]

    return neu1e

def train_cbow_sentiment_pair(model, target, word, input_word_indices, l1, alpha, obj_weight, learn_vectors=True, learn_hidden=True):
    l1 = model.syn0[input_word_indices].reshape(-1,1).flatten()    
    neu1e_senti = zeros(l1.shape)
    
    w1 = model.nn_parms['w1']
    b1 = model.nn_parms['b1']
    w2 = model.nn_parms['w2']
    b2 = model.nn_parms['b2']
    
    l2 = matmul(w1,l1) + b1 #t
    #print(dot(l1,w1.T).shape,l1.shape,l2.shape,w1.shape,b1.shape)
    l3 = array([x if (x>=-1) and (x<=1) else sign(x) for x in l2]) #hTanh(t)
    l4 = (matmul(w2,l3) + b2).flatten() # z
    l5 = 1./(1.+exp(-l4)) # logit(z)

    neuoe = l5-target
    #print("first:",input_word_indices,neuoe,l5,target)
    neu2e = matmul(w2.T, neuoe).flatten()
    #print("neu2e",neu2e)
    neuhte = neu2e.copy()
    neuhte[(l2<-1)|(l2>1)] = 0
    #print("neuhte",neuhte)
    neu1e_senti = neu1e_senti - alpha * matmul(w1.T, neuhte).flatten()
    #print("neu1e",neu1e)
    #print(neuoe.shape,w2_mask.shape,w2_oe.shape,neu1e.shape)
    
    #print("1:",target,l4,l5,neuoe)

    w2 = w2 - (1-obj_weight) * alpha * matmul(neuoe.reshape(-1,1),l3.reshape(1,-1))
    b2 = b2 - (1-obj_weight) * alpha * neuoe
    w1 = w1 - (1-obj_weight) * alpha * matmul(neuhte.reshape(-1,1),l1.reshape(1,-1))
    b1 = b1 - (1-obj_weight) * alpha * neuhte

    if learn_hidden:
        model.nn_parms['w1'] = w1
        model.nn_parms['b1'] = b1
        model.nn_parms['w2'] = w2
        model.nn_parms['b2'] = b2
    
    word_ind = input_word_indices.index(word.index)
    input_word_indices.remove(word.index) # except the word oneself
    l0 = np_sum(model.syn0[input_word_indices],axis=0)
    if model.cbow_mean and input_word_indices:
        l0 /= len(input_word_indices)    
    T = model.nn_parms['T']
    l1= dot(l0,T)
    
    neu1e_cbow = zeros(l1.shape)
    if model.hs:
        l2a = model.syn1[word.point]  # 2d matrix, codelen x layer1_size
        fa = 1. / (1. + exp(-dot(l1, l2a.T)))  # propagate hidden -> output
        ga = (1. - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
        ia = dot(ga, l2a) # errors on input
        if learn_hidden:
            model.syn1[word.point] += outer(ga, l1)  # learn hidden -> output
            model.nn_parms['T'] += outer(ia, l0) * obj_weight # transformation matrix
        neu1e_cbow += dot(ia,T.T) # save error

    if model.negative:
        # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
        word_indices = [word.index]
        while len(word_indices) < model.negative + 1:
            w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
            if w != word.index:
                word_indices.append(w)
        l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
        fb = 1. / (1. + exp(-dot(l1, l2b.T)))  # propagate hidden -> output
        gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
        ib = dot(gb, l2b)
        if learn_hidden:
            model.syn1neg[word_indices] += outer(gb, l1)  # learn hidden -> output
            model.nn_parms['T'] += outer(ib, l0) * obj_weight # transformation matrix
        neu1e_cbow += dot(ib,T.T)  # save error

    if learn_vectors:
        # learn input -> hidden, here for all words in the window separately
        if not model.cbow_mean and input_word_indices:
            neu1e_cbow /= len(input_word_indices)
        neu1e = neu1e_senti.reshape(-1,len(l1)) * (1-obj_weight)
        neu1e += neu1e_cbow * obj_weight
        neu1e[word_ind] -= neu1e_cbow * obj_weight
        for pos, i in enumerate(input_word_indices):
            model.syn0[i] += neu1e[pos] * model.syn0_lockf[i]

    return neu1e
    
def score_sg_pair(model, word, word2):
    l1 = model.syn0[word2.index]
    l2a = deepcopy(model.syn1[word.point])  # 2d matrix, codelen x layer1_size
    sgn = -1.0**word.code  # ch function, 0-> 1, 1 -> -1
    lprob = -log(1.0 + exp(-sgn*dot(l1, l2a.T)))
    return sum(lprob)


def score_cbow_pair(model, word, word2_indices, l1):
    l2a = model.syn1[word.point]  # 2d matrix, codelen x layer1_size
    sgn = -1.0**word.code  # ch function, 0-> 1, 1 -> -1
    lprob = -log(1.0 + exp(-sgn*dot(l1, l2a.T)))
    return sum(lprob)

class Vocab(object):
    """
    A single vocabulary item, used internally for collecting per-word frequency/sampling info,
    and for constructing binary trees (incl. both word leaves and inner nodes).

    """
    def __init__(self, **kwargs):
        self.count = 0
        self.label = -1.0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))


class Word2Vec(utils.SaveLoad):
    """
    Class for training, using and evaluating neural networks described in https://code.google.com/p/word2vec/

    The model can be stored/loaded via its `save()` and `load()` methods, or stored/loaded in a format
    compatible with the original word2vec implementation via `save_word2vec_format()` and `load_word2vec_format()`.

    """
    def __init__(
            self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
            trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, hidden_size=None, obj_weight = 0.1):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (unicode strings) that will be used for training.

        The `sentences` iterable can be simply a list, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.
        See :class:`BrownCorpus`, :class:`Text8Corpus` or :class:`LineSentence` in
        this module for such examples.

        If you don't supply `sentences`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.

        `sg` defines the training algorithm. By default (`sg=0`), CBOW is used.
        Otherwise (`sg=1`), skip-gram is employed.

        `size` is the dimensionality of the feature vectors.

        `window` is the maximum distance between the current and predicted word within a sentence.

        `alpha` is the initial learning rate (will linearly drop to zero as training progresses).

        `seed` = for the random number generator. Initial vectors for each
        word are seeded with a hash of the concatenation of word + str(seed).

        `min_count` = ignore all words with total frequency lower than this.

        `max_vocab_size` = limit RAM during vocabulary building; if there are more unique
        words than this, then prune the infrequent ones. Every 10 million word types
        need about 1GB of RAM. Set to `None` for no limit (default).

        `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
            default is 1e-3, useful range is (0, 1e-5).

        `workers` = use this many worker threads to train the model (=faster training with multicore machines).

        `hs` = if 1, hierarchical softmax will be used for model training.
        If set to 0 (default), and `negative` is non-zero, negative sampling will be used.

        `negative` = if > 0, negative sampling will be used, the int for negative
        specifies how many "noise words" should be drawn (usually between 5-20).
        Default is 5. If set to 0, no negative samping is used.

        `cbow_mean` = if 0, use the sum of the context word vectors. If 1 (default), use the mean.
        Only applies when cbow is used.

        `hashfxn` = hash function to use to randomly initialize weights, for increased
        training reproducibility. Default is Python's rudimentary built in hash function.

        `iter` = number of iterations (epochs) over the corpus.

        `trim_rule` = vocabulary trimming rule, specifies whether certain words should remain
         in the vocabulary, be trimmed away, or handled using the default (discard if word count < min_count).
         Can be None (min_count will be used), or a callable that accepts parameters (word, count, min_count) and
         returns either util.RULE_DISCARD, util.RULE_KEEP or util.RULE_DEFAULT.
         Note: The rule, if given, is only used prune vocabulary during build_vocab() and is not stored as part
          of the model.

        `sorted_vocab` = if 1 (default), sort the vocabulary by descending frequency before
        assigning word indexes.

        `batch_words` = target size (in words) for batches of examples passed to worker threads (and
        thus cython routines). Default is 10000. (Larger batches can be passed if individual
        texts are longer, but the cython code may truncate.)

        """
        self.vocab = {}  # mapping from a word (string) to a Vocab object
        self.index2word = []  # map from a word's matrix index (int) to word (string)
        self.sg = int(sg)
        self.cum_table = None  # for negative sampling
        self.vector_size = int(size)
        self.layer1_size = int(size)
        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.alpha = float(alpha)
        self.window = int(window)
        self.max_vocab_size = max_vocab_size
        self.seed = seed
        #self.do_train = 0
        self.random = random.RandomState(seed)
        self.min_count = min_count
        self.sample = sample
        self.workers = int(workers)
        self.min_alpha = float(min_alpha)
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.hashfxn = hashfxn
        self.iter = iter
        self.null_word = null_word
        self.train_count = 0
        self.total_train_time = 0
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words
        self.obj_weight = obj_weight
        
        if hidden_size is not None:
            self.layer1_size = hidden_size
        elif self.sg==2:
            self.hidden_size = self.vector_size
        
        if sentences is not None:
            if isinstance(sentences, GeneratorType):
                raise TypeError("You can't pass a generator as the sentences argument. Try an iterator.")
            if isinstance(sentences, ClassSentences):
                self.nclass = sentences.get_nclass()
            else:
                self.nclass = 0
                
            self.build_vocab(sentences, trim_rule=trim_rule)                       
            self.train(sentences)    
        
    def make_cum_table(self, power=0.75, domain=2**31 - 1):
        """
        Create a cumulative-distribution table using stored vocabulary word counts for
        drawing random words in the negative-sampling training routines.

        To draw a word index, choose a random integer up to the maximum value in the
        table (cum_table[-1]), then finding that integer's sorted insertion point
        (as if by bisect_left or ndarray.searchsorted()). That insertion point is the
        drawn index, coming up in proportion equal to the increment at that slot.

        Called internally from 'build_vocab()'.
        """
        vocab_size = len(self.index2word)
        self.cum_table = zeros(vocab_size, dtype=uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = float(sum([self.vocab[word].count**power for word in self.vocab]))
        cumulative = 0.0
        for word_index in range(vocab_size):
            cumulative += self.vocab[self.index2word[word_index]].count**power / train_words_pow
            self.cum_table[word_index] = round(cumulative * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain

    def create_binary_tree(self):
        """
        Create a binary Huffman tree using stored vocabulary word counts. Frequent words
        will have shorter binary codes. Called internally from `build_vocab()`.

        """
        logger.info("constructing a huffman tree from %i words", len(self.vocab))

        # build the huffman tree
        heap = list(itervalues(self.vocab))
        heapq.heapify(heap)
        for i in xrange(len(self.vocab) - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, Vocab(count=min1.count + min2.count, index=i + len(self.vocab), left=min1, right=min2))

        # recurse over the tree, assigning a binary code to each vocabulary word
        if heap:
            max_depth, stack = 0, [(heap[0], [], [])]
            while stack:
                node, codes, points = stack.pop()
                if node.index < len(self.vocab):
                    # leaf node => store its path from the root
                    node.code, node.point = codes, points
                    max_depth = max(len(codes), max_depth)
                else:
                    # inner node => continue recursion
                    points = array(list(points) + [node.index - len(self.vocab)], dtype=uint32)
                    stack.append((node.left, array(list(codes) + [0], dtype=uint8), points))
                    stack.append((node.right, array(list(codes) + [1], dtype=uint8), points))

            logger.info("built huffman tree with maximum node depth %i", max_depth)

    def build_vocab(self, sentences, keep_raw_vocab=False, trim_rule=None):
        """
        Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """
        self.scan_vocab(sentences, trim_rule=trim_rule)  # initial survey
        self.scale_vocab(keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule)  # trim by min_count & precalculate downsampling
        self.finalize_vocab()  # build tables & arrays

    def scan_vocab(self, sentences, progress_per=10000, trim_rule=None):
        """Do an initial scan of all words appearing in sentences."""
        logger.info("collecting all words and their counts")
        sentence_no = -1
        total_words = 0
        min_reduce = 1
        vocab = defaultdict(int)
        for sentence_no, sentence in enumerate(sentences):
            if sentence_no % progress_per == 0:
                logger.info("PROGRESS: at sentence #%i, processed %i words, keeping %i word types",
                            sentence_no, sum(itervalues(vocab)) + total_words, len(vocab))
            if isinstance(sentence,tuple):
                sentence=sentence[0]
            for word in sentence:
                vocab[word] += 1

            if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                total_words += utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
                min_reduce += 1

        total_words += sum(itervalues(vocab))
        logger.info("collected %i word types from a corpus of %i raw words and %i sentences",
                    len(vocab), total_words, sentence_no + 1)
        self.corpus_count = sentence_no + 1
        self.raw_vocab = vocab

    def scale_vocab(self, min_count=None, sample=None, dry_run=False, keep_raw_vocab=False, trim_rule=None):
        """
        Apply vocabulary settings for `min_count` (discarding less-frequent words)
        and `sample` (controlling the downsampling of more-frequent words).

        Calling with `dry_run=True` will only simulate the provided settings and
        report the size of the retained vocabulary, effective corpus length, and
        estimated memory requirements. Results are both printed via logging and
        returned as a dict.

        Delete the raw vocabulary after the scaling is done to free up RAM,
        unless `keep_raw_vocab` is set.

        """
        min_count = min_count or self.min_count
        sample = sample or self.sample

        # Discard words less-frequent than min_count
        if not dry_run:
            self.index2word = []
            # make stored settings match these applied settings
            self.min_count = min_count
            self.sample = sample
            self.vocab = {}
        drop_unique, drop_total, retain_total, original_total = 0, 0, 0, 0
        retain_words = []
        for word, v in iteritems(self.raw_vocab):
            if keep_vocab_item(word, v, min_count, trim_rule=trim_rule):
                retain_words.append(word)
                retain_total += v
                original_total += v
                if not dry_run:
                    self.vocab[word] = Vocab(count=v, index=len(self.index2word))
                    self.index2word.append(word)
            else:
                drop_unique += 1
                drop_total += v
                original_total += v
        logger.info("min_count=%d retains %i unique words (drops %i)",
                    min_count, len(retain_words), drop_unique)
        logger.info("min_count leaves %i word corpus (%i%% of original %i)",
                    retain_total, retain_total * 100 / max(original_total, 1), original_total)

        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        elif sample < 1.0:
            # traditional meaning: set parameter as proportion of total
            threshold_count = sample * retain_total
        else:
            # new shorthand: sample >= 1 means downsample all words with higher count than sample
            threshold_count = int(sample * (3 + sqrt(5)) / 2)

        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = self.raw_vocab[w]
            word_probability = (sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                self.vocab[w].sample_int = int(round(word_probability * 2**32))

        if not dry_run and not keep_raw_vocab:
            logger.info("deleting the raw counts dictionary of %i items", len(self.raw_vocab))
            self.raw_vocab = defaultdict(int)

        logger.info("sample=%g downsamples %i most-common words", sample, downsample_unique)
        logger.info("downsampling leaves estimated %i word corpus (%.1f%% of prior %i)",
                    downsample_total, downsample_total * 100.0 / max(retain_total, 1), retain_total)

        # return from each step: words-affected, resulting-corpus-size
        report_values = {'drop_unique': drop_unique, 'retain_total': retain_total,
                         'downsample_unique': downsample_unique, 'downsample_total': int(downsample_total)}

        # print extra memory estimates
        report_values['memory'] = self.estimate_memory(vocab_size=len(retain_words))

        return report_values

    def finalize_vocab(self):
        """Build tables and model weights based on final vocabulary settings."""
        if not self.index2word:
            self.scale_vocab()
        if self.sorted_vocab:
            self.sort_vocab()
        if self.hs:
            # add info about each word's Huffman encoding
            self.create_binary_tree()
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table()
        if self.null_word:
            # create null pseudo-word for padding when using concatenative L1 (run-of-words)
            # this word is only ever input – never predicted – so count, huffman-point, etc doesn't matter
            word, v = '\0', Vocab(count=1, sample_int=0)
            v.index = len(self.vocab)
            self.index2word.append(word)
            self.vocab[word] = v
        # set initial input/projection and hidden weights
        self.reset_weights()

    def sort_vocab(self):
        """Sort the vocabulary so the most frequent words have the lowest indexes."""
        if hasattr(self, 'syn0'):
            raise RuntimeError("must sort before initializing vectors/weights")
        self.index2word.sort(key=lambda word: self.vocab[word].count, reverse=True)
        for i, word in enumerate(self.index2word):
            self.vocab[word].index = i

    def reset_from(self, other_model):
        """
        Borrow shareable pre-built structures (like vocab) from the other_model. Useful
        if testing multiple models in parallel on the same corpus.
        """
        self.vocab = other_model.vocab
        self.index2word = other_model.index2word
        self.cum_table = other_model.cum_table
        self.corpus_count = other_model.corpus_count
        self.reset_weights()

    def _do_train_job(self, sentences, alpha, inits):
        """
        Train a single batch of sentences. Return 2-tuple `(effective word count after
        ignoring unknown words and sentence length trimming, total word count)`.
        """
        work, neu1 = inits
        tally = 0
        if self.sg == 1:
            tally += train_batch_sg(self, sentences, alpha, work)
        elif self.sg == 2:
            tally += train_batch_sswe(self, sentences, alpha, work)
        elif self.sg == 3:
            tally += train_batch_sswe_bow(self, sentences, alpha, work)
        elif self.sg == 4:
            tally += train_batch_cbow_sentiment(self, sentences, alpha, self.obj_weight, work, neu1)
        else:
            tally += train_batch_cbow(self, sentences, alpha, work, neu1)
        return tally, self._raw_word_count(sentences)

    def _raw_word_count(self, job):
        """Return the number of words in a given job."""
        return sum(len(sentence) for sentence in job)

    def train(self, sentences, total_words=None, word_count=0,
              total_examples=None, queue_factor=2, report_delay=1.0):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        For Word2Vec, each sentence must be a list of unicode strings. (Subclasses may accept other examples.)

        To support linear learning-rate decay from (initial) alpha to min_alpha, either total_examples
        (count of sentences) or total_words (count of raw words in sentences) should be provided, unless the
        sentences are the same as those that were used to initially build the vocabulary.

        """
        
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("C extension not loaded for Word2Vec, training will be slow. "
                          "Install a C compiler and reinstall gensim for fast training.")
            self.neg_labels = []
            if self.negative > 0:
                # precompute negative labels optimization for pure-python training
                self.neg_labels = zeros(self.negative + 1)
                self.neg_labels[0] = 1.
            if self.sg == 2:
                self.layer1_size = self.layer1_size * self.window

        logger.info(
            "training model with %i workers on %i vocabulary and %i features, "
            "using sg=%s hs=%s sample=%s negative=%s",
            self.workers, len(self.vocab), self.layer1_size, self.sg,
            self.hs, self.sample, self.negative)

        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")
        if not hasattr(self, 'syn0'):
            raise RuntimeError("you must first finalize vocabulary before training the model")

        if total_words is None and total_examples is None:
            if self.corpus_count:
                total_examples = self.corpus_count
                logger.info("expecting %i sentences, matching count from corpus used for vocabulary survey", total_examples)
            else:
                raise ValueError("you must provide either total_words or total_examples, to enable alpha and progress calculations")

        job_tally = 0

        if self.iter > 1:
            sentences = utils.RepeatCorpusNTimes(sentences, self.iter)
            total_words = total_words and total_words * self.iter
            total_examples = total_examples and total_examples * self.iter

        def worker_loop():
            """Train the model, lifting lists of sentences from the job_queue."""
            if self.sg in [2,4]:
                work = matutils.zeros_aligned(self.vector_size * self.window, dtype=REAL)  # per-thread private work memory
                neu1 = matutils.zeros_aligned(self.vector_size * self.window, dtype=REAL)
            elif self.sg==3:
                work = matutils.zeros_aligned(self.vector_size, dtype=REAL)  # per-thread private work memory
                neu1 = matutils.zeros_aligned(self.vector_size, dtype=REAL) 
            else:
                work = matutils.zeros_aligned(self.layer1_size, dtype=REAL)  # per-thread private work memory
                neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
            jobs_processed = 0
            while True:
                job = job_queue.get()
                if job is None:
                    progress_queue.put(None)
                    break  # no more jobs => quit this worker
                sentences, alpha = job
                tally, raw_tally = self._do_train_job(sentences, alpha, (work, neu1))
                #self.do_train += 1
                #print("# of do_train_job?? ",self.do_train)
                progress_queue.put((len(sentences), tally, raw_tally))  # report back progress
                jobs_processed += 1
            logger.debug("worker exiting, processed %i jobs", jobs_processed)

        def job_producer():
            """Fill jobs queue using the input `sentences` iterator."""
            job_batch, batch_size = [], 0
            pushed_words, pushed_examples = 0, 0
            next_alpha = self.alpha
            job_no = 0

            for sent_idx, sentence in enumerate(sentences):
                sentence_length = self._raw_word_count([sentence])

                # can we fit this sentence into the existing job batch?
                if batch_size + sentence_length <= self.batch_words:
                    # yes => add it to the current job
                    job_batch.append(sentence)
                    batch_size += sentence_length
                else:
                    # no => submit the existing job
                    logger.debug(
                        "queueing job #%i (%i words, %i sentences) at alpha %.05f",
                        job_no, batch_size, len(job_batch), next_alpha)
                    job_no += 1
                    job_queue.put((job_batch, next_alpha))

                    # update the learning rate for the next job
                    if self.min_alpha < next_alpha:
                        if total_examples:
                            # examples-based decay
                            pushed_examples += len(job_batch)
                            progress = 1.0 * pushed_examples / total_examples
                        else:
                            # words-based decay
                            pushed_words += self._raw_word_count(job_batch)
                            progress = 1.0 * pushed_words / total_words
                        next_alpha = self.alpha - (self.alpha - self.min_alpha) * progress
                        next_alpha = max(self.min_alpha, next_alpha)

                    # add the sentence that didn't fit as the first item of a new job
                    job_batch, batch_size = [sentence], sentence_length

            # add the last job too (may be significantly smaller than batch_words)
            if job_batch:
                logger.debug(
                    "queueing job #%i (%i words, %i sentences) at alpha %.05f",
                    job_no, batch_size, len(job_batch), next_alpha)
                job_no += 1
                job_queue.put((job_batch, next_alpha))

            if job_no == 0 and self.train_count == 0:
                logger.warning(
                    "train() called with an empty iterator (if not intended, "
                    "be sure to provide a corpus that offers restartable "
                    "iteration = an iterable)."
                )

            # give the workers heads up that they can finish -- no more work!
            for _ in xrange(self.workers):
                job_queue.put(None)
            logger.debug("job loop exiting, total %i jobs", job_no)

        # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [threading.Thread(target=worker_loop) for _ in xrange(self.workers)]
        unfinished_worker_count = len(workers)
        workers.append(threading.Thread(target=job_producer))

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        example_count, trained_word_count, raw_word_count = 0, 0, word_count
        start, next_report = default_timer() - 0.00001, 1.0

        while unfinished_worker_count > 0:
            report = progress_queue.get()  # blocks if workers too slow
            if report is None:  # a thread reporting that it finished
                unfinished_worker_count -= 1
                logger.info("worker thread finished; awaiting finish of %i more threads", unfinished_worker_count)
                continue
            examples, trained_words, raw_words = report
            job_tally += 1

            # update progress stats
            example_count += examples
            trained_word_count += trained_words  # only words in vocab & sampled
            raw_word_count += raw_words

            # log progress once every report_delay seconds
            elapsed = default_timer() - start
            if elapsed >= next_report:
                if total_examples:
                    # examples-based progress %
                    logger.info(
                        "PROGRESS: at %.2f%% examples, %.0f words/s, in_qsize %i, out_qsize %i",
                        100.0 * example_count / total_examples, trained_word_count / elapsed,
                        utils.qsize(job_queue), utils.qsize(progress_queue))
                else:
                    # words-based progress %
                    logger.info(
                        "PROGRESS: at %.2f%% words, %.0f words/s, in_qsize %i, out_qsize %i",
                        100.0 * raw_word_count / total_words, trained_word_count / elapsed,
                        utils.qsize(job_queue), utils.qsize(progress_queue))
                next_report = elapsed + report_delay

        # all done; report the final stats
        elapsed = default_timer() - start
        logger.info(
            "training on %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
            raw_word_count, trained_word_count, elapsed, trained_word_count / elapsed)
        if job_tally < 10 * self.workers:
            logger.warning("under 10 jobs per worker: consider setting a smaller `batch_words' for smoother alpha decay")

        # check that the input corpus hasn't changed during iteration
        if total_examples and total_examples != example_count:
            logger.warning("supplied example count (%i) did not equal expected count (%i)", example_count, total_examples)
        if total_words and total_words != raw_word_count:
            logger.warning("supplied raw word count (%i) did not equal expected count (%i)", raw_word_count, total_words)

        self.train_count += 1  # number of times train() has been called
        self.total_train_time += elapsed
        self.clear_sims()
        return trained_word_count

    # basics copied from the train() function
    def score(self, sentences, total_sentences=int(1e6), chunksize=100, queue_factor=2, report_delay=1):
        """
        Score the log probability for a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.
        This does not change the fitted model in any way (see Word2Vec.train() for that)

        Note that you should specify total_sentences; we'll run into problems if you ask to
        score more than this number of sentences but it is inefficient to set the value too high.

        See the article by [taddy]_ and the gensim demo at [deepir]_ for examples of how to use such scores in document classification.

        .. [taddy] Taddy, Matt.  Document Classification by Inversion of Distributed Language Representations, in Proceedings of the 2015 Conference of the Association of Computational Linguistics.
        .. [deepir] https://github.com/TaddyLab/gensim/blob/deepir/docs/notebooks/deepir.ipynb

        """
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("C extension compilation failed, scoring will be slow. "
                          "Install a C compiler and reinstall gensim for fastness.")

        logger.info(
            "scoring sentences with %i workers on %i vocabulary and %i features, "
            "using sg=%s hs=%s sample=%s and negative=%s",
            self.workers, len(self.vocab), self.layer1_size, self.sg, self.hs, self.sample, self.negative)

        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before scoring new data")

        if not self.hs:
            raise RuntimeError("we have only implemented score for hs")

        def worker_loop():
            """Train the model, lifting lists of sentences from the jobs queue."""
            work = zeros(1, dtype=REAL)  # for sg hs, we actually only need one memory loc (running sum)
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
            while True:
                job = job_queue.get()
                if job is None:  # signal to finish
                    break
                ns = 0
                for sentence_id, sentence in job:
                    if sentence_id >= total_sentences:
                        break
                    if self.sg:
                        score = score_sentence_sg(self, sentence, work)
                    else:
                        score = score_sentence_cbow(self, sentence, work, neu1)
                    sentence_scores[sentence_id] = score
                    ns += 1
                progress_queue.put(ns)  # report progress

        start, next_report = default_timer(), 1.0
        # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [threading.Thread(target=worker_loop) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        sentence_count = 0
        sentence_scores = matutils.zeros_aligned(total_sentences, dtype=REAL)

        push_done = False
        done_jobs = 0
        jobs_source = enumerate(utils.grouper(enumerate(sentences), chunksize))

        # fill jobs queue with (id, sentence) job items
        while True:
            try:
                job_no, items = next(jobs_source)
                if (job_no - 1) * chunksize > total_sentences:
                    logger.warning(
                        "terminating after %i sentences (set higher total_sentences if you want more).",
                        total_sentences)
                    job_no -= 1
                    raise StopIteration()
                logger.debug("putting job #%i in the queue", job_no)
                job_queue.put(items)
            except StopIteration:
                logger.info(
                    "reached end of input; waiting to finish %i outstanding jobs",
                    job_no - done_jobs + 1)
                for _ in xrange(self.workers):
                    job_queue.put(None)  # give the workers heads up that they can finish -- no more work!
                push_done = True
            try:
                while done_jobs < (job_no + 1) or not push_done:
                    ns = progress_queue.get(push_done)  # only block after all jobs pushed
                    sentence_count += ns
                    done_jobs += 1
                    elapsed = default_timer() - start
                    if elapsed >= next_report:
                        logger.info(
                            "PROGRESS: at %.2f%% sentences, %.0f sentences/s",
                            100.0 * sentence_count, sentence_count / elapsed)
                        next_report = elapsed + report_delay  # don't flood log, wait report_delay seconds
                else:
                    # loop ended by job count; really done
                    break
            except Empty:
                pass  # already out of loop; continue to next push

        elapsed = default_timer() - start
        self.clear_sims()
        logger.info(
            "scoring %i sentences took %.1fs, %.0f sentences/s",
            sentence_count, elapsed, sentence_count / elapsed)
        return sentence_scores[:sentence_count]

    def clear_sims(self):
        self.syn0norm = None

    def reset_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        self.syn0 = empty((len(self.vocab), self.vector_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        for i in xrange(len(self.vocab)):
            # construct deterministic seed from word AND seed argument
            self.syn0[i] = self.seeded_vector(self.index2word[i] + str(self.seed))
        if self.hs:
            self.syn1 = zeros((len(self.vocab), self.layer1_size), dtype=REAL)
        if self.negative:
            self.syn1neg = zeros((len(self.vocab), self.layer1_size), dtype=REAL)
        self.syn0norm = None

        self.syn0_lockf = ones(len(self.vocab), dtype=REAL)  # zeros suppress learning

        if self.sg==2:
            self.nn_parms={}
            self.nn_parms['w1']=empty((self.layer1_size,self.vector_size*self.window), dtype=REAL)
            self.nn_parms['w2']=empty((self.nclass-1,self.layer1_size), dtype=REAL)
            for i in range(self.layer1_size):
                self.nn_parms['w1'][i]=self.seeded_vector('w1'+str(i) + str(self.seed),self.vector_size * self.window) 
            for i in range(self.nclass-1):
                self.nn_parms['w2'][i]=self.seeded_vector('w2'+str(i) + str(self.seed),self.layer1_size)
            
            self.nn_parms['b1']=empty((1,self.layer1_size), dtype=REAL)
            self.nn_parms['b1'][0]=self.seeded_vector('b1' + str(self.seed),self.layer1_size)
            self.nn_parms['b1']=self.nn_parms['b1'].flatten() 
            
            self.nn_parms['b2']=empty((1,self.nclass-1), dtype=REAL)
            self.nn_parms['b2'][0]=self.seeded_vector('b2' + str(self.seed),self.nclass-1)
            self.nn_parms['b2']=self.nn_parms['b2'].flatten() 
        elif self.sg==3:
            self.nn_parms={}
            self.nn_parms['w1']=empty((self.layer1_size,self.vector_size), dtype=REAL)
            self.nn_parms['w2']=empty((self.nclass-1,self.layer1_size), dtype=REAL)
            for i in range(self.layer1_size):
                self.nn_parms['w1'][i]=self.seeded_vector('w1'+str(i) + str(self.seed),self.vector_size)
            for i in range(self.nclass-1):
                self.nn_parms['w2'][i]=self.seeded_vector('w2'+str(i) + str(self.seed),self.layer1_size)
            
            self.nn_parms['b1']=empty((1,self.layer1_size), dtype=REAL)
            self.nn_parms['b1'][0]=self.seeded_vector('b1' + str(self.seed),self.layer1_size)
            self.nn_parms['b1']=self.nn_parms['b1'].flatten()
            
            self.nn_parms['b2']=empty((1,self.nclass-1), dtype=REAL)
            self.nn_parms['b2'][0]=self.seeded_vector('b2' + str(self.seed),self.nclass-1)
            self.nn_parms['b2']=self.nn_parms['b2'].flatten()
        elif self.sg==4:
            self.nn_parms={}
            
            # setence level(sentiment)         
            self.nn_parms['w1']=empty((self.layer1_size,self.vector_size*self.window), dtype=REAL)
            self.nn_parms['w2']=empty((self.nclass-1,self.layer1_size), dtype=REAL)
            for i in range(self.layer1_size):
                self.nn_parms['w1'][i]=self.seeded_vector('w1'+str(i) + str(self.seed),self.vector_size * self.window) 
            for i in range(self.nclass-1):
                self.nn_parms['w2'][i]=self.seeded_vector('w2'+str(i) + str(self.seed),self.layer1_size)
            
            self.nn_parms['b1']=empty((1,self.layer1_size), dtype=REAL)
            self.nn_parms['b1'][0]=self.seeded_vector('b1' + str(self.seed),self.layer1_size)
            self.nn_parms['b1']=self.nn_parms['b1'].flatten() 
            
            self.nn_parms['b2']=empty((1,self.nclass-1), dtype=REAL)
            self.nn_parms['b2'][0]=self.seeded_vector('b2' + str(self.seed),self.nclass-1)
            self.nn_parms['b2']=self.nn_parms['b2'].flatten()
            
            # transformation
            self.nn_parms['T']=empty((self.vector_size,self.vector_size), dtype=REAL)
            for i in range(self.vector_size):
                self.nn_parms['T'][i]=self.seeded_vector('T'+str(i) + str(self.seed),self.vector_size) 
            
    def seeded_vector(self, seed_string, vector_size=None):
        """Create one 'random' vector (but deterministic by seed_string)"""
        # Note: built-in hash() may vary by Python version or even (in Py3.x) per launch
        once = random.RandomState(self.hashfxn(seed_string) & 0xffffffff)
        if vector_size is None:
            vector_size = self.vector_size

        return (once.rand(vector_size) - 0.5) / vector_size

    def save_word2vec_format(self, fname, fvocab=None, binary=False):
        """
        Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        """
        if fvocab is not None:
            logger.info("storing vocabulary in %s" % (fvocab))
            with utils.smart_open(fvocab, 'wb') as vout:
                for word, vocab in sorted(iteritems(self.vocab), key=lambda item: -item[1].count):
                    vout.write(utils.to_utf8("%s %s\n" % (word, vocab.count)))
        logger.info("storing %sx%s projection weights into %s" % (len(self.vocab), self.vector_size, fname))
        assert (len(self.vocab), self.vector_size) == self.syn0.shape
        with utils.smart_open(fname, 'wb') as fout:
            fout.write(utils.to_utf8("%s %s\n" % self.syn0.shape))
            # store in sorted order: most frequent words at the top
            for word, vocab in sorted(iteritems(self.vocab), key=lambda item: -item[1].count):
                row = self.syn0[vocab.index]
                if binary:
                    fout.write(utils.to_utf8(word) + b" " + row.tostring())
                else:
                    fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join("%f" % val for val in row))))

    @classmethod
    def load_word2vec_format(cls, fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict'):
        """
        Load the input-hidden weight matrix from the original C word2vec-tool format.

        Note that the information stored in the file is incomplete (the binary tree is missing),
        so while you can query for word similarity etc., you cannot continue training
        with a model loaded this way.

        `binary` is a boolean indicating whether the data is in binary word2vec format.
        `norm_only` is a boolean indicating whether to only store normalised word2vec vectors in memory.
        Word counts are read from `fvocab` filename, if set (this is the file generated
        by `-save-vocab` flag of the original C tool).

        If you trained the C model using non-utf8 encoding for words, specify that
        encoding in `encoding`.

        """
        counts = None
        if fvocab is not None:
            logger.info("loading word counts from %s", fvocab)
            counts = {}
            with utils.smart_open(fvocab) as fin:
                for line in fin:
                    word, count = utils.to_unicode(line).strip().split()
                    counts[word] = int(count)

        logger.info("loading projection weights from %s", fname)
        with utils.smart_open(fname) as fin:
            header = utils.to_unicode(fin.readline(), encoding=encoding)
            vocab_size, vector_size = map(int, header.split())  # throws for invalid file format
            result = cls(size=vector_size)
            result.syn0 = zeros((vocab_size, vector_size), dtype=REAL)

            def add_word(word, weights):
                word_id = len(result.vocab)
                if word in result.vocab:
                    logger.warning("duplicate word '%s' in %s, ignoring all but first", word, fname)
                    return
                if counts is None:
                    # most common scenario: no vocab file given. just make up some bogus counts, in descending order
                    result.vocab[word] = Vocab(index=word_id, count=vocab_size - word_id)
                elif word in counts:
                    # use count from the vocab file
                    result.vocab[word] = Vocab(index=word_id, count=counts[word])
                else:
                    # vocab file given, but word is missing -- set count to None (TODO: or raise?)
                    logger.warning("vocabulary file is incomplete: '%s' is missing", word)
                    result.vocab[word] = Vocab(index=word_id, count=None)
                result.syn0[word_id] = weights
                result.index2word.append(word)

            if binary:
                binary_len = dtype(REAL).itemsize * vector_size
                for line_no in xrange(vocab_size):
                    # mixed text and binary: read text first, then binary
                    word = []
                    while True:
                        ch = fin.read(1)
                        if ch == b' ':
                            break
                        if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                            word.append(ch)
                    word = utils.to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                    weights = fromstring(fin.read(binary_len), dtype=REAL)
                    add_word(word, weights)
            else:
                for line_no, line in enumerate(fin):
                    parts = utils.to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
                    if len(parts) != vector_size + 1:
                        raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                    word, weights = parts[0], list(map(REAL, parts[1:]))
                    add_word(word, weights)
        if result.syn0.shape[0] != len(result.vocab):
            logger.info(
                "duplicate words detected, shrinking matrix size from %i to %i",
                result.syn0.shape[0], len(result.vocab)
            )
            result.syn0 = ascontiguousarray(result.syn0[: len(result.vocab)])
        assert (len(result.vocab), result.vector_size) == result.syn0.shape

        logger.info("loaded %s matrix from %s" % (result.syn0.shape, fname))
        return result

    def intersect_word2vec_format(self, fname, binary=False, encoding='utf8', unicode_errors='strict'):
        """
        Merge the input-hidden weight matrix from the original C word2vec-tool format
        given, where it intersects with the current vocabulary. (No words are added to the
        existing vocabulary, but intersecting words adopt the file's weights, and
        non-intersecting words are left alone.)

        `binary` is a boolean indicating whether the data is in binary word2vec format.
        """
        overlap_count = 0
        logger.info("loading projection weights from %s" % (fname))
        with utils.smart_open(fname) as fin:
            header = utils.to_unicode(fin.readline(), encoding=encoding)
            vocab_size, vector_size = map(int, header.split())  # throws for invalid file format
            if not vector_size == self.vector_size:
                raise ValueError("incompatible vector size %d in file %s" % (vector_size, fname))
                # TOCONSIDER: maybe mismatched vectors still useful enough to merge (truncating/padding)?
            if binary:
                binary_len = dtype(REAL).itemsize * vector_size
                for line_no in xrange(vocab_size):
                    # mixed text and binary: read text first, then binary
                    word = []
                    while True:
                        ch = fin.read(1)
                        if ch == b' ':
                            break
                        if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                            word.append(ch)
                    word = utils.to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                    weights = fromstring(fin.read(binary_len), dtype=REAL)
                    if word in self.vocab:
                        overlap_count += 1
                        self.syn0[self.vocab[word].index] = weights
                        self.syn0_lockf[self.vocab[word].index] = 0.0  # lock it
            else:
                for line_no, line in enumerate(fin):
                    parts = utils.to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
                    if len(parts) != vector_size + 1:
                        raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                    word, weights = parts[0], list(map(REAL, parts[1:]))
                    if word in self.vocab:
                        overlap_count += 1
                        self.syn0[self.vocab[word].index] = weights
        logger.info("merged %d vectors into %s matrix from %s" % (overlap_count, self.syn0.shape, fname))

    def most_similar(self, positive=[], negative=[], topn=10, restrict_vocab=None):
        """
        Find the top-N most similar words. Positive words contribute positively towards the
        similarity, negative words negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given words and the vectors for each word in the model.
        The method corresponds to the `word-analogy` and `distance` scripts in the original
        word2vec implementation.

        If topn is False, most_similar returns the vector of similarity scores.

        `restrict_vocab` is an optional integer which limits the range of vectors which
        are searched for most-similar values. For example, restrict_vocab=10000 would
        only check the first 10000 word vectors in the vocabulary order. (This may be
        meaningful if you've sorted the vocabulary by descending frequency.)

        Example::

          >>> trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
          [('queen', 0.50882536), ...]

        """
        self.init_sims()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [
            (word, 1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in positive
        ]
        negative = [
            (word, -1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in negative
        ]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, ndarray):
                mean.append(weight * word)
            elif word in self.vocab:
                mean.append(weight * self.syn0norm[self.vocab[word].index])
                all_words.add(self.vocab[word].index)
            else:
                raise KeyError("word '%s' not in vocabulary" % word)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        limited = self.syn0norm if restrict_vocab is None else self.syn0norm[:restrict_vocab]
        dists = dot(limited, mean)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]

    def most_similar_cosmul(self, positive=[], negative=[], topn=10):
        """
        Find the top-N most similar words, using the multiplicative combination objective
        proposed by Omer Levy and Yoav Goldberg in [4]_. Positive words still contribute
        positively towards the similarity, negative words negatively, but with less
        susceptibility to one large distance dominating the calculation.

        In the common analogy-solving case, of two positive and one negative examples,
        this method is equivalent to the "3CosMul" objective (equation (4)) of Levy and Goldberg.

        Additional positive or negative examples contribute to the numerator or denominator,
        respectively – a potentially sensible but untested extension of the method. (With
        a single positive example, rankings will be the same as in the default most_similar.)

        Example::

          >>> trained_model.most_similar_cosmul(positive=['baghdad', 'england'], negative=['london'])
          [(u'iraq', 0.8488819003105164), ...]

        .. [4] Omer Levy and Yoav Goldberg. Linguistic Regularities in Sparse and Explicit Word Representations, 2014.

        """
        self.init_sims()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar_cosmul('dog'), as a shorthand for most_similar_cosmul(['dog'])
            positive = [positive]

        all_words = set()

        def word_vec(word):
            if isinstance(word, ndarray):
                return word
            elif word in self.vocab:
                all_words.add(self.vocab[word].index)
                return self.syn0norm[self.vocab[word].index]
            else:
                raise KeyError("word '%s' not in vocabulary" % word)

        positive = [word_vec(word) for word in positive]
        negative = [word_vec(word) for word in negative]
        if not positive:
            raise ValueError("cannot compute similarity with no input")

        # equation (4) of Levy & Goldberg "Linguistic Regularities...",
        # with distances shifted to [0,1] per footnote (7)
        pos_dists = [((1 + dot(self.syn0norm, term)) / 2) for term in positive]
        neg_dists = [((1 + dot(self.syn0norm, term)) / 2) for term in negative]
        dists = prod(pos_dists, axis=0) / (prod(neg_dists, axis=0) + 0.000001)

        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]

    def doesnt_match(self, words):
        """
        Which word from the given list doesn't go with the others?

        Example::

          >>> trained_model.doesnt_match("breakfast cereal dinner lunch".split())
          'cereal'

        """
        self.init_sims()

        words = [word for word in words if word in self.vocab]  # filter out OOV words
        logger.debug("using words %s" % words)
        if not words:
            raise ValueError("cannot select a word from an empty list")
        vectors = vstack(self.syn0norm[self.vocab[word].index] for word in words).astype(REAL)
        mean = matutils.unitvec(vectors.mean(axis=0)).astype(REAL)
        dists = dot(vectors, mean)
        return sorted(zip(dists, words))[0][1]

    def __getitem__(self, words):

        """
        Accept a single word or a list of words as input.

        If a single word: returns the word's representations in vector space, as
        a 1D numpy array.

        Multiple words: return the words' representations in vector space, as a
        2d numpy array: #words x #vector_size. Matrix rows are in the same order
        as in input.

        Example::

          >>> trained_model['office']
          array([ -1.40128313e-02, ...])

          >>> trained_model[['office', 'products']]
          array([ -1.40128313e-02, ...]
                [ -1.70425311e-03, ...]
                 ...)

        """
        if isinstance(words, string_types):
            # allow calls like trained_model['office'], as a shorthand for trained_model[['office']]
            return self.syn0[self.vocab[words].index]

        return vstack([self.syn0[self.vocab[word].index] for word in words])

    def __contains__(self, word):
        return word in self.vocab

    def similarity(self, w1, w2):
        """
        Compute cosine similarity between two words.

        Example::

          >>> trained_model.similarity('woman', 'man')
          0.73723527

          >>> trained_model.similarity('woman', 'woman')
          1.0

        """
        return dot(matutils.unitvec(self[w1]), matutils.unitvec(self[w2]))

    def n_similarity(self, ws1, ws2):
        """
        Compute cosine similarity between two sets of words.

        Example::

          >>> trained_model.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])
          0.61540466561049689

          >>> trained_model.n_similarity(['restaurant', 'japanese'], ['japanese', 'restaurant'])
          1.0000000000000004

          >>> trained_model.n_similarity(['sushi'], ['restaurant']) == trained_model.similarity('sushi', 'restaurant')
          True

        """
        v1 = [self[word] for word in ws1]
        v2 = [self[word] for word in ws2]
        return dot(matutils.unitvec(array(v1).mean(axis=0)), matutils.unitvec(array(v2).mean(axis=0)))

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training** after doing a replace. The model becomes
        effectively read-only = you can call `most_similar`, `similarity` etc., but not `train`.

        """
        if getattr(self, 'syn0norm', None) is None or replace:
            logger.info("precomputing L2-norms of word weight vectors")
            if replace:
                for i in xrange(self.syn0.shape[0]):
                    self.syn0[i, :] /= sqrt((self.syn0[i, :] ** 2).sum(-1))
                self.syn0norm = self.syn0
                if hasattr(self, 'syn1'):
                    del self.syn1
            else:
                self.syn0norm = (self.syn0 / sqrt((self.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)

    def estimate_memory(self, vocab_size=None, report=None):
        """Estimate required memory for a model using current settings and provided vocabulary size."""
        vocab_size = vocab_size or len(self.vocab)
        report = report or {}
        report['vocab'] = vocab_size * (700 if self.hs else 500)
        report['syn0'] = vocab_size * self.vector_size * dtype(REAL).itemsize
        if self.hs:
            report['syn1'] = vocab_size * self.layer1_size * dtype(REAL).itemsize
        if self.negative:
            report['syn1neg'] = vocab_size * self.layer1_size * dtype(REAL).itemsize
        report['total'] = sum(report.values())
        logger.info("estimated required memory for %i words and %i dimensions: %i bytes",
                    vocab_size, self.vector_size, report['total'])
        return report

    @staticmethod
    def log_accuracy(section):
        correct, incorrect = len(section['correct']), len(section['incorrect'])
        if correct + incorrect > 0:
            logger.info("%s: %.1f%% (%i/%i)" %
                        (section['section'], 100.0 * correct / (correct + incorrect),
                         correct, correct + incorrect))

    def accuracy(self, questions, restrict_vocab=30000, most_similar=most_similar):
        """
        Compute accuracy of the model. `questions` is a filename where lines are
        4-tuples of words, split into sections by ": SECTION NAME" lines.
        See https://code.google.com/p/word2vec/source/browse/trunk/questions-words.txt for an example.

        The accuracy is reported (=printed to log and returned as a list) for each
        section separately, plus there's one aggregate summary at the end.

        Use `restrict_vocab` to ignore all questions containing a word whose frequency
        is not in the top-N most frequent words (default top 30,000).

        This method corresponds to the `compute-accuracy` script of the original C word2vec.

        """
        ok_vocab = dict(sorted(iteritems(self.vocab),
                               key=lambda item: -item[1].count)[:restrict_vocab])
        ok_index = set(v.index for v in itervalues(ok_vocab))

        sections, section = [], None
        for line_no, line in enumerate(utils.smart_open(questions)):
            # TODO: use level3 BLAS (=evaluate multiple questions at once), for speed
            line = utils.to_unicode(line)
            if line.startswith(': '):
                # a new section starts => store the old section
                if section:
                    sections.append(section)
                    self.log_accuracy(section)
                section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
            else:
                if not section:
                    raise ValueError("missing section header before line #%i in %s" % (line_no, questions))
                try:
                    a, b, c, expected = [word.lower() for word in line.split()]  # TODO assumes vocabulary preprocessing uses lowercase, too...
                except:
                    logger.info("skipping invalid line #%i in %s" % (line_no, questions))
                if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                    logger.debug("skipping line #%i with OOV words: %s" % (line_no, line.strip()))
                    continue

                ignore = set(self.vocab[v].index for v in [a, b, c])  # indexes of words to ignore
                predicted = None
                # find the most likely prediction, ignoring OOV words and input words
                sims = most_similar(self, positive=[b, c], negative=[a], topn=False, restrict_vocab=restrict_vocab)
                for index in matutils.argsort(sims, reverse=True):
                    if index in ok_index and index not in ignore:
                        predicted = self.index2word[index]
                        if predicted != expected:
                            logger.debug("%s: expected %s, predicted %s", line.strip(), expected, predicted)
                        break
                if predicted == expected:
                    section['correct'].append((a, b, c, expected))
                else:
                    section['incorrect'].append((a, b, c, expected))
        if section:
            # store the last section, too
            sections.append(section)
            self.log_accuracy(section)

        total = {
            'section': 'total',
            'correct': sum((s['correct'] for s in sections), []),
            'incorrect': sum((s['incorrect'] for s in sections), []),
        }
        self.log_accuracy(total)
        sections.append(total)
        return sections

    def __str__(self):
        return "%s(vocab=%s, size=%s, alpha=%s)" % (self.__class__.__name__, len(self.index2word), self.vector_size, self.alpha)

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors, recalculable table
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm', 'table', 'cum_table'])
        super(Word2Vec, self).save(*args, **kwargs)

    save.__doc__ = utils.SaveLoad.save.__doc__

    @classmethod
    def load(cls, *args, **kwargs):
        model = super(Word2Vec, cls).load(*args, **kwargs)
        # update older models
        if hasattr(model, 'table'):
            delattr(model, 'table')  # discard in favor of cum_table
        if model.negative and hasattr(model, 'index2word'):
            model.make_cum_table()  # rebuild cum_table from vocabulary
        if not hasattr(model, 'corpus_count'):
            model.corpus_count = None
        for v in model.vocab.values():
            if hasattr(v, 'sample_int'):
                break  # already 0.12.0+ style int probabilities
            elif hasattr(v, 'sample_probability'):
                v.sample_int = int(round(v.sample_probability * 2**32))
                del v.sample_probability
        if not hasattr(model, 'syn0_lockf') and hasattr(model, 'syn0'):
            model.syn0_lockf = ones(len(model.syn0), dtype=REAL)
        if not hasattr(model, 'random'):
            model.random = random.RandomState(model.seed)
        if not hasattr(model, 'train_count'):
            model.train_count = 0
            model.total_train_time = 0
        return model


class BrownCorpus(object):
    """Iterate over sentences from the Brown corpus (part of NLTK data)."""
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            for line in utils.smart_open(fname):
                line = utils.to_unicode(line)
                # each file line is a single sentence in the Brown corpus
                # each token is WORD/POS_TAG
                token_tags = [t.split('/') for t in line.split() if len(t.split('/')) == 2]
                # ignore words with non-alphabetic tags like ",", "!" etc (punctuation, weird stuff)
                words = ["%s/%s" % (token.lower(), tag[:2]) for token, tag in token_tags if tag[:2].isalpha()]
                if not words:  # don't bother sending out empty sentences
                    continue
                yield words


class Text8Corpus(object):
    """Iterate over sentences from the "text8" corpus, unzipped from http://mattmahoney.net/dc/text8.zip ."""
    def __init__(self, fname, max_sentence_length=MAX_WORDS_IN_BATCH):
        self.fname = fname
        self.max_sentence_length = max_sentence_length

    def __iter__(self):
        # the entire corpus is one gigantic line -- there are no sentence marks at all
        # so just split the sequence of tokens arbitrarily: 1 sentence = 1000 tokens
        sentence, rest = [], b''
        with utils.smart_open(self.fname) as fin:
            while True:
                text = rest + fin.read(8192)  # avoid loading the entire file (=1 line) into RAM
                if text == rest:  # EOF
                    sentence.extend(rest.split())  # return the last chunk of words, too (may be shorter/longer)
                    if sentence:
                        yield sentence
                    break
                last_token = text.rfind(b' ')  # last token may have been split in two... keep for next iteration
                words, rest = (utils.to_unicode(text[:last_token]).split(),
                               text[last_token:].strip()) if last_token >= 0 else ([], text)
                sentence.extend(words)
                while len(sentence) >= self.max_sentence_length:
                    yield sentence[:self.max_sentence_length]
                    sentence = sentence[self.max_sentence_length:]


class LineSentence(object):
    """
    Simple format: one sentence = one line; words already preprocessed and separated by whitespace.
    """

    def __init__(self, source, max_sentence_length=MAX_WORDS_IN_BATCH, limit=None):
        """
        `source` can be either a string or a file object. Clip the file to the first
        `limit` lines (or no clipped if limit is None, the default).

        Example::

            sentences = LineSentence('myfile.txt')

        Or for compressed files::

            sentences = LineSentence('compressed_text.txt.bz2')
            sentences = LineSentence('compressed_text.txt.gz')

        """
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for line in itertools.islice(self.source, self.limit):
                line = utils.to_unicode(line).split()
                i = 0
                while i < len(line):
                    yield line[i : i + self.max_sentence_length]
                    i += self.max_sentence_length
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.smart_open(self.source) as fin:
                for line in itertools.islice(fin, self.limit):
                    line = utils.to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i : i + self.max_sentence_length]
                        i += self.max_sentence_length

class LoadSentences(object):
    def __init__(self, dirname, max_sentence_length=MAX_WORDS_IN_BATCH, limit=None, stopwords = True):
        self.dirname = dirname
        self.max_sentence_length = max_sentence_length
        self.limit = limit
        self.stopwords = stopwords

    def __iter__(self):
        punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        if self.stopwords:
            stopwords='''i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such only own same so than too s t can will just don should now d ll m o re ve y'''
            stopwords=stopwords.split()
        else:
            stopwords=[]
        for fname in os.listdir(self.dirname):
            i = 0
            for line in open(os.path.join(self.dirname, fname), encoding="utf-8"):
                line = utils.to_unicode(line).lower()
                line = ''.join([ch for ch in line if ch not in punct])
                line = line.split()
                line = [x for x in line if x not in stopwords]
                i = 0
                while i < len(line):
                    yield line[i: i + self.max_sentence_length]
                    i += self.max_sentence_length

class SplitSentences(object):
    def __init__(self, data, max_sentence_length = MAX_WORDS_IN_BATCH, limit = None, stopwords = True):
        self.data = data
        self.max_sentence_length = max_sentence_length
        self.limit = limit
        self.stopwords = stopwords

    def __iter__(self):
        punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        if self.stopwords:
            stopwords='''i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such only own same so than too s t can will just don should now d ll m o re ve y'''
            stopwords=stopwords.split() 
        else:
            stopwords=[]
        for text in self.data:
            i = 0
            for line in text.split('\n'):
                line = utils.to_unicode(line).lower()
                line = ''.join([ch for ch in line if ch not in punct])
                line = line.split()
                line = [x for x in line if x not in stopwords]
                i = 0
                while i < len(line):
                    yield line[i: i + self.max_sentence_length]
                    i += self.max_sentence_length

class ClassSentences(object):
    def __init__(self, sentences, labels, max_sentence_length=MAX_WORDS_IN_BATCH, limit=None, stopwords = True):
        self.sentences = sentences
        self.labels = labels
        self.max_sentence_length = max_sentence_length
        self.limit = limit
        self.stopwords = stopwords

    def __iter__(self):
        punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        if self.stopwords:
            stopwords='''i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such only own same so than too s t can will just don should now d ll m o re ve y'''
            stopwords=stopwords.split()            
        else:
            stopwords=[]
        for ind, text in enumerate(self.sentences):
            for line in text.split('\n'):
                line = utils.to_unicode(line).lower()
                line = ''.join([ch for ch in line if ch not in punct])
                line = line.split()
                line = [x for x in line if x not in stopwords]
                i = 0
                while i < len(line):
                    yield line[i: i + self.max_sentence_length],self.labels[ind]
                    i += self.max_sentence_length

    def get_nclass(self):
        return len(set(self.labels))
        
def load_dir(dirname):
    corpus=[]
    for fname in os.listdir(dirname):
        f = open(os.path.join(dirname, fname), encoding="utf-8")
        corpus.append(f.read())
    return corpus


# Example: ./word2vec.py -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3
if __name__ == "__main__":
    import pandas as pd
    import time
    import numpy    
    import pickle
       
    book = load_dir(datapath)
    label = zeros(len(book))
    label[1000:] = 1

    book = pd.DataFrame(data={'target':label,'text':book})
    book_corpus = ClassSentences(book['text'].values,book['target'].values,stopwords=False)
    #book_corpus = LoadSentences(datapath)
    
    ###### parameters ######
    sg=4
    iter=10
    min_count=5
    window=5
    hs=100
    sample=0.01
    alpha=0.025
    corpus=book_corpus
    dataname='book'
    obj_weight=0.2    
    ###### parameters ######        
    
    filelist=os.listdir('result')
    filelist=[x for x in filelist if '.pkl' in x and time.strftime("%Y%m%d") in x and 'sg%d'%(sg) in x and dataname in x]
    version=len(filelist)+1
        
    print("start")
    t1=time.time()
    model_book = Word2Vec(corpus,min_count=min_count,sg=sg,window=window,hidden_size=hs, sample=sample, alpha=alpha, iter=iter, obj_weight=obj_weight)
    #model_book.save_word2vec_format(os.path.join(path,'model_book_sg3.bin'))
    output=open('result/model_%s_sg%d_%s_v%d.pkl'%(dataname,model_book.sg,time.strftime("%Y%m%d"),version),'wb')
    pickle.dump(model_book,output)
    output.close()
    print(time.time()-t1)
