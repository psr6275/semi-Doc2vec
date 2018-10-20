#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Deep learning via the distributed memory and distributed bag of words models from
[1]_, using either hierarchical softmax or negative sampling [2]_ [3]_.

**Make sure you have a C compiler before installing gensim, to use optimized (compiled)
doc2vec training** (70x speedup [blog]_).

Initialize a model with e.g.::

>>> model = Doc2Vec(documents, size=100, window=8, min_count=5, workers=4)

Persist a model to disk with::

>>> model.save(fname)
>>> model = Doc2Vec.load(fname)  # you can continue training with the loaded model!

The model can also be instantiated from an existing file on disk in the word2vec C format::

  >>> model = Doc2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format
  >>> model = Doc2Vec.load_word2vec_format('/tmp/vectors.bin', binary=True)  # C binary format

.. [1] Quoc Le and Tomas Mikolov. Distributed Representations of Sentences and Documents. http://arxiv.org/pdf/1405.4053v2.pdf
.. [2] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
.. [3] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality.
       In Proceedings of NIPS, 2013.
.. [blog] Optimizing word2vec in gensim, http://radimrehurek.com/2013/09/word2vec-in-python-part-two-optimizing/

"""
import sys,os
sys.path.append(os.getcwd())

import logging

import warnings
from six import iteritems

try:
    from queue import Queue
except ImportError:
    from Queue import Queue
import time
from collections import namedtuple, defaultdict
from timeit import default_timer

from numpy import zeros, random, sum as np_sum, add as np_add, concatenate, \
    repeat as np_repeat, array, float32 as REAL, empty, ones, memmap as np_memmap, \
    sqrt, newaxis, ndarray, dot, vstack, dtype, divide as np_divide
import copy
import numpy as np
import utils, matutils  # utility fnc for pickling, common scipy operations etc
import word2vec
from word2vec import Word2Vec, Vocab, train_cbow_pair, train_sg_pair, train_batch_sg
from six.moves import xrange, zip
from six import string_types, integer_types, itervalues
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray, sign, matmul, argsort, in1d, tile, kron, eye

from joblib import Parallel, delayed
import multiprocessing
logger = logging.getLogger(__name__)


def find_nn(i,syn0norm,topn=10):
    dists = dot(syn0norm,syn0norm[i])
    if not topn:
        return dists
    best = matutils.argsort(dists, topn = topn+1, reverse = True)
    result = [(sim,float(dists[sim])) for sim in best[1:]]
    return result

def find_nn_idx(i,syn0norm,topn=10):
    dists = dot(syn0norm,syn0norm[i])
    if not topn:
        return dists
    best = matutils.argsort(dists, topn = topn+1, reverse = True)
    result = best[1:]
    return result


try:

    from doc2vec_inner import train_document_dbow, train_document_dm, train_document_dm_concat,train_document_dbow_neighbor
    from word2vec_inner import FAST_VERSION  # blas-adaptation shared from word2vec
    print("Fast version for doc2vec import Success!")
    #a[1111]
except:
    print("# failed... fall back to plain numpy (20-80x slower training than the above)")
    FAST_VERSION = -1
    
    def train_document_dbow(model, doc_words, doctag_indexes, alpha, work=None,
                            train_words=False, learn_doctags=True, learn_words=True, learn_hidden=True,
                            word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
        """
        Update distributed bag of words model ("PV-DBOW") by training on a single document.
    
        Called internally from `Doc2Vec.train()` and `Doc2Vec.infer_vector()`.
    
        The document is provided as `doc_words`, a list of word tokens which are looked up
        in the model's vocab dictionary, and `doctag_indexes`, which provide indexes
        into the doctag_vectors array.
    
        If `train_words` is True, simultaneously train word-to-word (not just doc-to-word)
        examples, exactly as per Word2Vec skip-gram training. (Without this option,
        word vectors are neither consulted nor updated during DBOW doc vector training.)
    
        Any of `learn_doctags', `learn_words`, and `learn_hidden` may be set False to
        prevent learning-updates to those respective model weights, as if using the
        (partially-)frozen model to infer other compatible vectors.
    
        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from doc2vec_inner instead.
    
        """
        if doctag_vectors is None:
            doctag_vectors = model.docvecs.doctag_syn0
        if doctag_locks is None:
            doctag_locks = model.docvecs.doctag_syn0_lockf
    
        if train_words and learn_words:
            train_batch_sg(model, [doc_words], alpha, work)
        for doctag_index in doctag_indexes:
            for word in doc_words:
                train_sg_pair(model, word, doctag_index, alpha, learn_vectors=learn_doctags,
                              learn_hidden=learn_hidden, context_vectors=doctag_vectors,
                              context_locks=doctag_locks)
    
        return len(doc_words)
    
    def train_document_dbow_neighbor(model, doc_words, doctag_indexes, alpha, beta, work=None,
                            train_words=False, learn_doctags=True, learn_words=True, learn_hidden=True,
                            word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
        """
        train document vectors with reflecting tne labels of neighborhoods of a given document.
        
        We can choose one of two approaches.
        First, the document can move toward neighborhood documents which have the same labels, 
        at the same time, far way from neighborhood documents which have the opposite labels.
        Second, the document can move toward neighborhood documents which have the same labels, 
        at the same time, try to maintain the neighbors if the document do not have label information.
        
        
        """
        if doctag_vectors is None:
            doctag_vectors = model.docvecs.doctag_syn0
        if doctag_locks is None:
            doctag_locks = model.docvecs.doctag_syn0_lockf
    
        if train_words and learn_words:
            train_batch_sg(model, [doc_words], alpha, work)
        for doctag_index in doctag_indexes:
            for word in doc_words:
                train_sg_pair(model, word, doctag_index, alpha, learn_vectors=learn_doctags,
                              learn_hidden=learn_hidden, context_vectors=doctag_vectors,
                              context_locks=doctag_locks)
            
            train_doc_neighbors(model,doctag_index, beta, learn_vectors=learn_doctags, learn_hidden=learn_hidden)
            
                      
    
    
    def train_document_dm(model, doc_words, doctag_indexes, alpha, work=None, neu1=None,
                          learn_doctags=True, learn_words=True, learn_hidden=True,
                          word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
        """
        Update distributed memory model ("PV-DM") by training on a single document.
    
        Called internally from `Doc2Vec.train()` and `Doc2Vec.infer_vector()`. This
        method implements the DM model with a projection (input) layer that is
        either the sum or mean of the context vectors, depending on the model's
        `dm_mean` configuration field.  See `train_document_dm_concat()` for the DM
        model with a concatenated input layer.
    
        The document is provided as `doc_words`, a list of word tokens which are looked up
        in the model's vocab dictionary, and `doctag_indexes`, which provide indexes
        into the doctag_vectors array.
    
        Any of `learn_doctags', `learn_words`, and `learn_hidden` may be set False to
        prevent learning-updates to those respective model weights, as if using the
        (partially-)frozen model to infer other compatible vectors.
    
        This is the non-optimized, Python version. If you have a C compiler, gensim
        will use the optimized version from doc2vec_inner instead.
    
        """
        
        if word_vectors is None:
            word_vectors = model.syn0
        if word_locks is None:
            word_locks = model.syn0_lockf
        if doctag_vectors is None:
            doctag_vectors = model.docvecs.doctag_syn0
        if doctag_locks is None:
            doctag_locks = model.docvecs.doctag_syn0_lockf
    
        word_vocabs = [model.vocab[w] for w in doc_words if w in model.vocab and
                       model.vocab[w].sample_int > model.random.rand() * 2**32]
    
        for pos, word in enumerate(word_vocabs):
            reduced_window = model.random.randint(model.window)  # `b` in the original doc2vec code
            start = max(0, pos - model.window + reduced_window)
            window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
            word2_indexes = [word2.index for pos2, word2 in window_pos if pos2 != pos]
            l1 = np_sum(word_vectors[word2_indexes], axis=0) + np_sum(doctag_vectors[doctag_indexes], axis=0)
            count = len(word2_indexes) + len(doctag_indexes)
            if model.cbow_mean and count > 1 :
                l1 /= count
            neu1e = train_cbow_pair(model, word, word2_indexes, l1, alpha,
                                    learn_vectors=False, learn_hidden=learn_hidden)
            if not model.cbow_mean and count > 1:
                neu1e /= count
            if learn_doctags:
                for i in doctag_indexes:
                    doctag_vectors[i] += neu1e * doctag_locks[i]
            if learn_words:
                for i in word2_indexes:
                    word_vectors[i] += neu1e * word_locks[i]
    
        return len(word_vocabs)
    
    def train_document_dm_concat(model, doc_words, doctag_indexes, alpha, work=None, neu1=None,
                                 learn_doctags=True, learn_words=True, learn_hidden=True,
                                 word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
        """
        Update distributed memory model ("PV-DM") by training on a single document, using a
        concatenation of the context window word vectors (rather than a sum or average).
    
        Called internally from `Doc2Vec.train()` and `Doc2Vec.infer_vector()`.
    
        The document is provided as `doc_words`, a list of word tokens which are looked up
        in the model's vocab dictionary, and `doctag_indexes`, which provide indexes
        into the doctag_vectors array.
    
        Any of `learn_doctags', `learn_words`, and `learn_hidden` may be set False to
        prevent learning-updates to those respective model weights, as if using the
        (partially-)frozen model to infer other compatible vectors.
    
        This is the non-optimized, Python version. If you have a C compiler, gensim
        will use the optimized version from doc2vec_inner instead.
    
        """
        if word_vectors is None:
            word_vectors = model.syn0
        if word_locks is None:
            word_locks = model.syn0_lockf
        if doctag_vectors is None:
            doctag_vectors = model.docvecs.doctag_syn0
        if doctag_locks is None:
            doctag_locks = model.docvecs.doctag_syn0_lockf
    
        word_vocabs = [model.vocab[w] for w in doc_words if w in model.vocab and
                       model.vocab[w].sample_int > model.random.rand() * 2**32]
        doctag_len = len(doctag_indexes)
        if doctag_len != model.dm_tag_count:
            return 0  # skip doc without expected number of doctag(s) (TODO: warn/pad?)
    
        null_word = model.vocab['\0']
        pre_pad_count = model.window
        post_pad_count = model.window
        padded_document_indexes = (
            (pre_pad_count * [null_word.index])  # pre-padding
            + [word.index for word in word_vocabs if word is not None]  # elide out-of-Vocabulary words
            + (post_pad_count * [null_word.index])  # post-padding
        )
    
        for pos in range(pre_pad_count, len(padded_document_indexes) - post_pad_count):
            word_context_indexes = (
                padded_document_indexes[(pos - pre_pad_count): pos]  # preceding words
                + padded_document_indexes[(pos + 1):(pos + 1 + post_pad_count)]  # following words
            )
            word_context_len = len(word_context_indexes)
            predict_word = model.vocab[model.index2word[padded_document_indexes[pos]]]
            # numpy advanced-indexing copies; concatenate, flatten to 1d
            l1 = concatenate((doctag_vectors[doctag_indexes], word_vectors[word_context_indexes])).ravel()
            neu1e = train_cbow_pair(model, predict_word, None, l1, alpha,
                                    learn_hidden=learn_hidden, learn_vectors=False)
    
            # filter by locks and shape for addition to source vectors
            e_locks = concatenate((doctag_locks[doctag_indexes], word_locks[word_context_indexes]))
            neu1e_r = (neu1e.reshape(-1, model.vector_size)
                       * np_repeat(e_locks, model.vector_size).reshape(-1, model.vector_size))
    
            if learn_doctags:
                np_add.at(doctag_vectors, doctag_indexes, neu1e_r[:doctag_len])
            if learn_words:
                np_add.at(word_vectors, word_context_indexes, neu1e_r[doctag_len:])
    
        return len(padded_document_indexes) - pre_pad_count - post_pad_count


def train_doc_neighbors(model,doctag_indexes, beta, learn_vectors=True, learn_hidden=True,
              doctag_vectors = None, doctag_locks = None):
    """
    using this function, we want to train the document vector, reflecting the laebls of neighbor documents
    We can choose the learn the base document vectors using 'learn_vectors'
    """
   
    if doctag_vectors is None:
        doctag_vectors = model.docvecs.doctag_syn0
    
    if doctag_locks is None:
        doctag_locks = model.docvecs.doctag_syn0_lockf
    #model.init_sims()    
    base_doc = doctag_vectors[doctag_indexes]
    base_size =  sqrt((base_doc ** 2).sum(-1))
    
    base_label = model.doc_label[doctag_indexes]
    
    nbs = model.docvecs.most_similar(base_doc,topn = model.num_neighbor) # init_sim is runned!
    #doctag_norm = copy.deepcopy(model.docvecs.doctag_syn0norm)
    nbs_doc_tags = [i for i,j in nbs]
    #nbs_doc_sim = [j for i,j in nbs] 
    indexed_doctags = model.docvecs.indexed_doctags(nbs_doc_tags)
    nbs_indexes, nbs_vectors, nbs_locks, ignored = indexed_doctags    
        
    nbs_labels = model.doc_label[nbs_indexes]
    
    if model.learn_unlabel:
        lij = ((nbs_labels * base_label)+1)/2
    else:
        lij = nbs_labels * base_label
        
    neu1e = zeros(base_doc.shape)
    l1 = base_doc
    for nb,l in zip(nbs_vectors,lij):
        l2a = copy.deepcopy(nb)
        nb_size = sqrt((nb**2).sum(-1))
        fa = 1.0 / (1.0 + exp(-dot(l1, l2a.T)))  # propagate hidden -> output
        ga = (1 - fa)*beta
        l3a = (1/(nb_size*base_size))*(l2a - (dot(l1,l2a.T)/(base_size**2))*l1)
        neu1e += dot(ga, l3a)*l
    if learn_vectors:
        l1 += neu1e * doctag_locks
    
    return neu1e
    

class TaggedDocument(namedtuple('TaggedDocument', 'words tags')):
    """
    A single document, made up of `words` (a list of unicode string tokens)
    and `tags` (a list of tokens). Tags may be one or more unicode string
    tokens, but typical practice (which will also be most memory-efficient) is
    for the tags list to include a unique integer id as the only tag.
    
    Replaces "sentence as a list of words" from Word2Vec.
    
    """
    def __str__(self):
        return '%s(%s, %s)' % (self.__class__.__name__, self.words, self.tags)


# for compatibility
class LabeledSentence(TaggedDocument):
    def __init__(self, *args, **kwargs):
        warnings.warn('LabeledSentence has been replaced by TaggedDocument', DeprecationWarning)


class DocvecsArray(utils.SaveLoad):
    """
    Default storage of doc vectors during/after training, in a numpy array.
    
    As the 'docvecs' property of a Doc2Vec model, allows access and
    comparison of document vectors.
    
    >>> docvec = d2v_model.docvecs[99]
    >>> docvec = d2v_model.docvecs['SENT_99']  # if string tag used in training
    >>> sims = d2v_model.docvecs.most_similar(99)
    >>> sims = d2v_model.docvecs.most_similar('SENT_99')
    >>> sims = d2v_model.docvecs.most_similar(docvec)
    
    If only plain int tags are presented during training, the dict (of
    string tag -> index) and list (of index -> string tag) stay empty,
    saving memory.
    
    Supplying a mapfile_path (as by initializing a Doc2Vec model with a
    'docvecs_mapfile' value) will use a pair of memory-mapped
    files as the array backing for doctag_syn0/doctag_syn0_lockf values.
    
    The Doc2Vec model automatically uses this class, but a future alternative
    implementation, based on another persistence mechanism like LMDB, LevelDB,
    or SQLite, should also be possible.
    """
    def __init__(self, mapfile_path=None):
        self.doctags = {}  # string -> Doctag (only filled if necessary)
        self.max_rawint = -1  # highest rawint-indexed doctag
        self.offset2doctag = []  # int offset-past-(max_rawint+1) -> String (only filled if necessary)
        self.count = 0
        self.mapfile_path = mapfile_path
    
    def note_doctag(self, key, document_no, document_length):
        """Note a document tag during initial corpus scan, for structure sizing."""
        if isinstance(key, int):
            self.max_rawint = max(self.max_rawint, key)
        else:
            if key in self.doctags:
                self.doctags[key] = self.doctags[key].repeat(document_length)
            else:
                self.doctags[key] = Doctag(len(self.offset2doctag), document_length, 1)
                self.offset2doctag.append(key)
        self.count = self.max_rawint + 1 + len(self.offset2doctag)
    
    def indexed_doctags(self, doctag_tokens):
        """Return indexes and backing-arrays used in training examples."""
        return ([self._int_index(index) for index in doctag_tokens if index in self],
                self.doctag_syn0, self.doctag_syn0_lockf, doctag_tokens)
    
    def trained_item(self, indexed_tuple):
        """Persist any changes made to the given indexes (matching tuple previously
        returned by indexed_doctags()); a no-op for this implementation"""
        pass
    
    def _int_index(self, index):
        """Return int index for either string or int index"""
        if isinstance(index, int):
            return index
        else:
            return self.max_rawint + 1 + self.doctags[index].offset
    
    def _key_index(self, i_index, missing=None):
        """Return string index for given int index, if available"""
        warnings.warn("use DocvecsArray.index_to_doctag", DeprecationWarning)
        return self.index_to_doctag(i_index)
    
    def index_to_doctag(self, i_index):
        """Return string key for given i_index, if available. Otherwise return raw int doctag (same int)."""
        candidate_offset = i_index - self.max_rawint - 1
        if 0 <= candidate_offset < len(self.offset2doctag):
            return self.offset2doctag[candidate_offset]
        else:
            return i_index
    
    def __getitem__(self, index):
        """
        Accept a single key (int or string tag) or list of keys as input.
    
        If a single string or int, return designated tag's vector
        representation, as a 1D numpy array.
    
        If a list, return designated tags' vector representations as a
        2D numpy array: #tags x #vector_size.
        """
        if isinstance(index, string_types + (int,)):
            return self.doctag_syn0[self._int_index(index)]
    
        return vstack([self[i] for i in index])
    
    def __len__(self):
        return self.count
    
    def __contains__(self, index):
        if isinstance(index, int):
            return index < self.count
        else:
            return index in self.doctags
    
    def borrow_from(self, other_docvecs):
        self.count = other_docvecs.count
        self.doctags = other_docvecs.doctags
        self.offset2doctag = other_docvecs.offset2doctag
    
    def clear_sims(self):
        self.doctag_syn0norm = None
    
    def estimated_lookup_memory(self):
        """Estimated memory for tag lookup; 0 if using pure int tags."""
        return 60 * len(self.offset2doctag) + 140 * len(self.doctags)
    
    def reset_weights(self, model):
        length = max(len(self.doctags), self.count)
        if self.mapfile_path:
            self.doctag_syn0 = np_memmap(self.mapfile_path+'.doctag_syn0', dtype=REAL,
                                         mode='w+', shape=(length, model.vector_size))
            self.doctag_syn0_lockf = np_memmap(self.mapfile_path+'.doctag_syn0_lockf', dtype=REAL,
                                               mode='w+', shape=(length,))
            self.doctag_syn0_lockf.fill(1.0)
        else:
            self.doctag_syn0 = empty((length, model.vector_size), dtype=REAL)
            #self.doctag_syn0norm = empty((length,model.vector_size),dtype = REAL)
            self.doctag_syn0_lockf = ones((length,), dtype=REAL)  # zeros suppress learning
    
        for i in xrange(length):
            # construct deterministic seed from index AND model seed
            seed = "%d %s" % (model.seed, self.index_to_doctag(i))
            self.doctag_syn0[i] = model.seeded_vector(seed)
    
    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.
    
        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!
    
        Note that you **cannot continue training** after doing a replace. The model becomes
        effectively read-only = you can call `most_similar`, `similarity` etc., but not `train`.
    
        """
        if getattr(self, 'doctag_syn0norm', None) is None or replace:
            logger.info("precomputing L2-norms of doc weight vectors")
            if replace:
                for i in xrange(self.doctag_syn0.shape[0]):
                    self.doctag_syn0[i, :] /= sqrt((self.doctag_syn0[i, :] ** 2).sum(-1))
                self.doctag_syn0norm = self.doctag_syn0
            else:
                if self.mapfile_path:
                    self.doctag_syn0norm = np_memmap(
                        self.mapfile_path+'.doctag_syn0norm', dtype=REAL,
                        mode='w+', shape=self.doctag_syn0.shape)
                else:
                    self.doctag_syn0norm = empty(self.doctag_syn0.shape, dtype=REAL)
                np_divide(self.doctag_syn0, sqrt((self.doctag_syn0 ** 2).sum(-1))[..., newaxis], self.doctag_syn0norm)
    
    def most_similar(self, positive=[], negative=[], topn=10, clip_start=0, clip_end=None):
        """
        Find the top-N most similar docvecs known from training. Positive docs contribute
        positively towards the similarity, negative docs negatively.
    
        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given docs. Docs may be specified as vectors, integer indexes
        of trained docvecs, or if the documents were originally presented with string tags,
        by the corresponding tags.
    
        The 'clip_start' and 'clip_end' allow limiting results to a particular contiguous
        range of the underlying doctag_syn0norm vectors. (This may be useful if the ordering
        there was chosen to be significant, such as more popular tag IDs in lower indexes.)
        """
        self.init_sims()
        clip_end = clip_end or len(self.doctag_syn0norm)
    
        if isinstance(positive, string_types + integer_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]
    
        # add weights for each doc, if not already present; default to 1.0 for positive and -1.0 for negative docs
        positive = [
            (doc, 1.0) if isinstance(doc, string_types + (ndarray,) + integer_types)
            else doc for doc in positive
        ]
        negative = [
            (doc, -1.0) if isinstance(doc, string_types + (ndarray,) + integer_types)
            else doc for doc in negative
        ]
    
        # compute the weighted average of all docs
        all_docs, mean = set(), []
        for doc, weight in positive + negative:
            if isinstance(doc, ndarray):
                mean.append(weight * doc)
            elif doc in self.doctags or doc < self.count:
                mean.append(weight * self.doctag_syn0norm[self._int_index(doc)])
                all_docs.add(self._int_index(doc))
            else:
                raise KeyError("doc '%s' not in trained set" % doc)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)
    
        dists = dot(self.doctag_syn0norm[clip_start:clip_end], mean)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_docs), reverse=True)
        # ignore (don't return) docs from the input
        result = [(self.index_to_doctag(sim), float(dists[sim])) for sim in best if sim not in all_docs]
        return result[:topn]
    
    def doesnt_match(self, docs):
        """
        Which doc from the given list doesn't go with the others?
    
        (TODO: Accept vectors of out-of-training-set docs, as if from inference.)
    
        """
        self.init_sims()
    
        docs = [doc for doc in docs if doc in self.doctags or 0 <= doc < self.count]  # filter out unknowns
        logger.debug("using docs %s" % docs)
        if not docs:
            raise ValueError("cannot select a doc from an empty list")
        vectors = vstack(self.doctag_syn0norm[self._int_index(doc)] for doc in docs).astype(REAL)
        mean = matutils.unitvec(vectors.mean(axis=0)).astype(REAL)
        dists = dot(vectors, mean)
        return sorted(zip(dists, docs))[0][1]
    
    def similarity(self, d1, d2):
        """
        Compute cosine similarity between two docvecs in the trained set, specified by int index or
        string tag. (TODO: Accept vectors of out-of-training-set docs, as if from inference.)
    
        """
        return dot(matutils.unitvec(self[d1]), matutils.unitvec(self[d2]))
    
    def n_similarity(self, ds1, ds2):
        """
        Compute cosine similarity between two sets of docvecs from the trained set, specified by int
        index or string tag. (TODO: Accept vectors of out-of-training-set docs, as if from inference.)
    
        """
        v1 = [self[doc] for doc in ds1]
        v2 = [self[doc] for doc in ds2]
        return dot(matutils.unitvec(array(v1).mean(axis=0)), matutils.unitvec(array(v2).mean(axis=0)))


class Doctag(namedtuple('Doctag', 'offset, word_count, doc_count')):
    """A string document tag discovered during the initial vocabulary
    scan. (The document-vector equivalent of a Vocab object.)
    
    Will not be used if all presented document tags are ints.
    
    The offset is only the true index into the doctags_syn0/doctags_syn0_lockf
    if-and-only-if no raw-int tags were used. If any raw-int tags were used,
    string Doctag vectors begin at index (max_rawint + 1), so the true index is
    (rawint_index + 1 + offset). See also DocvecsArray.index_to_doctag().
    """
    __slots__ = ()
    
    def repeat(self, word_count):
        return self._replace(word_count=self.word_count + word_count, doc_count=self.doc_count + 1)


class Doc2Vec(Word2Vec):
    """Class for training, using and evaluating neural networks described in http://arxiv.org/pdf/1405.4053v2.pdf"""
    def __init__(self, documents=None, doc_label=None, size=300, alpha=0.025, window=8, min_count=5, beta = 0.010,
                 max_vocab_size=None, sample=0, seed=1, workers=1, min_alpha=0.0001, learn_unlabel = 0, dbow_neighbors = 0,
                 dm=1, hs=1, negative=0, dbow_words=0, dm_mean=0, dm_concat=0, dm_tag_count=1, num_neighbor = 10,nb_list = None,
                 docvecs=None, docvecs_mapfile=None, comment=None, trim_rule=None, learn_word=True, word_model=None, **kwargs):
        """
        Initialize the model from an iterable of `documents`. Each document is a
        TaggedDocument object that will be used for training.
    
        The `documents` iterable can be simply a list of TaggedDocument elements, but for larger corpora,
        consider an iterable that streams the documents directly from disk/network.
    
        If you don't supply `documents`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.
    
        `dm` defines the training algorithm. By default (`dm=1`), 'distributed memory' (PV-DM) is used.
        Otherwise, `distributed bag of words` (PV-DBOW) is employed.
    
        `size` is the dimensionality of the feature vectors.
    
        `window` is the maximum distance between the predicted word and context words used for prediction
        within a document.
    
        `alpha` is the initial learning rate (will linearly drop to zero as training progresses).
    
        'beta' is the control learning rate for documents similarities.
        
        `seed` = for the random number generator. Only runs with a single worker will be
        deterministically reproducible because of the ordering randomness in multi-threaded runs.
    
        `min_count` = ignore all words with total frequency lower than this.
    
        `max_vocab_size` = limit RAM during vocabulary building; if there are more unique
        words than this, then prune the infrequent ones. Every 10 million word types
        need about 1GB of RAM. Set to `None` for no limit (default).
    
        `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
                default is 0 (off), useful value is 1e-5.
    
        `workers` = use this many worker threads to train the model (=faster training with multicore machines).
    
        `hs` = if 1 (default), hierarchical sampling will be used for model training (else set to 0).
    
        `negative` = if > 0, negative sampling will be used, the int for negative
        specifies how many "noise words" should be drawn (usually between 5-20).
    
        `dm_mean` = if 0 (default), use the sum of the context word vectors. If 1, use the mean.
        Only applies when dm is used in non-concatenative mode.
    
        `dm_concat` = if 1, use concatenation of context vectors rather than sum/average;
        default is 0 (off). Note concatenation results in a much-larger model, as the input
        is no longer the size of one (sampled or arithmatically combined) word vector, but the
        size of the tag(s) and all words in the context strung together.
    
        `dm_tag_count` = expected constant number of document tags per document, when using
        dm_concat mode; default is 1.
    
        `dbow_words` if set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW
        doc-vector training; default is 0 (faster training of doc-vectors only).
    
        `trim_rule` = vocabulary trimming rule, specifies whether certain words should remain
         in the vocabulary, be trimmed away, or handled using the default (discard if word count < min_count).
         Can be None (min_count will be used), or a callable that accepts parameters (word, count, min_count) and
         returns either util.RULE_DISCARD, util.RULE_KEEP or util.RULE_DEFAULT.
         Note: The rule, if given, is only used prune vocabulary during build_vocab() and is not stored as part
          of the model.
         
        'learn_unlabel' = 0 if the unlabel document will not be updated when the neighborhoods of the document are considered.
        and 1 if the unlabeled document try to move to maintain the neighborhoods.
        
        'doc_label' is the sentiment label of the documents. It is needed for semi-supervised document representation. Also, this variable is 'array' type not 'list' type.

        'nb_list' is the list of indexes, which are neighbors of documents. It is needed for semi-supervised training.
        """
        
        self.learn_word = learn_word
        if learn_word:
            super(Doc2Vec, self).__init__(
                size=size, alpha=alpha, window=window, min_count=min_count, max_vocab_size=max_vocab_size,
                sample=sample, seed=seed, workers=workers, min_alpha=min_alpha,
                sg=(1+dm) % 2, hs=hs, negative=negative, cbow_mean=dm_mean,
                null_word=dm_concat, **kwargs)
        elif word_model is not None:
            self.window = word_model.window
            self.index2word = word_model.index2word
            self.sg = word_model.sg
            self.cum_table = word_model.cum_table
            self.vector_size = word_model.vector_size
            self.min_count = word_model.min_count
            self.max_vocab_size = word_model.max_vocab_size
            self.sample = word_model.sample
            self.workers = word_model.workers
            self.min_alpha = word_model.alpha
            self.hs = word_model.hs
            self.negative = word_model.negative
            self.cbow_mean = word_model.cbow_mean
            self.null_word = word_model.null_word
            self.train_count = word_model.train_count
            self.seed = word_model.seed        
            self.random = random.RandomState(self.seed)
            self.hashfxn = word_model.hashfxn
            self.iter = word_model.iter
            self.sorted_vocab = word_model.sorted_vocab
            self.batch_words = word_model.batch_words
            self.alpha = word_model.alpha
            self.total_train_time = 0
            #self.do_train = word_model.do_train
            # word vectors
            self.vocab = word_model.vocab
            self.syn0 = word_model.syn0
            self.syn0_lockf = zeros(len(self.vocab), dtype=REAL)      
            if self.hs:
                self.syn1 = word_model.syn1
            if self.negative:
                self.syn1neg = word_model.syn1neg            
            self.syn0norm = word_model.syn0norm            
            if self.sg==2:
                self.nn_parms = word_model.nn_parms
                self.nclass = word_model.nclass
                try:
                    self.hidden_size = word_model.hidden_size                           
                except:
                    self.hidden_size = len(word_model.nn_parms['b1'])
        
        self.num_neighbor = num_neighbor
        self.doc_label = np.array(doc_label)
        self.nb_list = nb_list
        self.beta = beta
        self.learn_unlabel = learn_unlabel  
        self.dbow_words = dbow_words
        self.dbow_neighbors = dbow_neighbors
        self.dm_concat = dm_concat
        self.do_train = 0
        self.dm_tag_count = dm_tag_count
        if self.dm and self.dm_concat:
            self.layer1_size = (self.dm_tag_count + (2 * self.window)) * self.vector_size
        else:
            self.layer1_size = size
        
        if word_model is not None:
            if word_model.sg!=2:
                self.layer1_size = word_model.layer1_size #dimension or concatenate!
            else:
                self.layer1_size = word_model.vector_size #vector dimension//
        
            
        self.docvecs = docvecs or DocvecsArray(docvecs_mapfile)
        self.comment = comment
        if documents is not None:
            self.build_vocab(documents, trim_rule=trim_rule)
            self.train(documents)
    
    @property
    def dm(self):
        return not self.sg  # opposite of SG
    
    @property
    def dbow(self):
        return self.sg  # same as SG
    
    def clear_sims(self):
        super(Doc2Vec, self).clear_sims()
        self.docvecs.clear_sims()
    
    def reset_weights(self):
        if self.dm and self.dm_concat:
            # expand l1 size to match concatenated tags+words length
            self.layer1_size = (self.dm_tag_count + (2 * self.window)) * self.vector_size
            logger.info("using concatenative %d-dimensional layer1" % (self.layer1_size))
        if self.learn_word:
            super(Doc2Vec, self).reset_weights()

        self.docvecs.reset_weights(self)
    
    def reset_from(self, other_model):
        """Reuse shareable structures from other_model."""
        self.docvecs.borrow_from(other_model.docvecs)
        super(Doc2Vec, self).reset_from(other_model)
    
    def build_vocab(self, documents, keep_raw_vocab=False, trim_rule=None):
        """
        Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """
        self.scan_vocab(documents, trim_rule=trim_rule)  # initial survey
        if self.learn_word:
            self.scale_vocab(keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule)  # trim by min_count & precalculate downsampling
            self.finalize_vocab()  # build tables & arrays
        self.reset_weights()
        
    def scan_vocab(self, documents, progress_per=10000, trim_rule=None):
        logger.info("collecting all words and their counts")
        document_no = -1
        if self.learn_word:
            total_words = 0
            min_reduce = 1
            vocab = defaultdict(int)
        else:
            vocab = self.vocab
            for word, voc in sorted(iteritems(self.vocab), key=lambda item: -item[1].count):
                self.syn0_lockf[voc.index] = 0 # we need to lock the pre-trained words!
                
        interval_start = default_timer() - 0.00001  # guard against next sample being identical
        interval_count = 0        
        for document_no, document in enumerate(documents):
            if document_no % progress_per == 0:      
                
                interval_rate = (total_words - interval_count) / (default_timer() - interval_start)
                logger.info("PROGRESS: at example #%i, processed %i words (%i/s), %i word types, %i tags",
                            document_no, total_words, interval_rate, len(vocab), len(self.docvecs))
                interval_start = default_timer()
                interval_count = total_words
            document_length = len(document.words)
    
            for tag in document.tags:
                self.docvecs.note_doctag(tag, document_no, document_length)
            
            
            for word in document.words:
                vocab[word] += 1
            total_words += len(document.words)
    
            if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
                min_reduce += 1
        if self.learn_word:
            logger.info("collected %i word types and %i unique tags from a corpus of %i examples and %i words",
                        len(vocab), len(self.docvecs), document_no + 1, total_words)
        self.corpus_count = document_no + 1
        
        
        self.raw_vocab = vocab
    """
    def set_neighbor_list(self):
        docvecs = self.docvecs.doctag_syn0
        m,n = docvecs.shape
        p = self.num_neighbor
        if self.nb_list is None:
            self.nb_list = zeros((m,p))
        for ii in range(m):
            #print(docvecs[ii])
            nbs = self.docvecs.most_similar([docvecs[ii]],topn = p)
            #print(nbs)
            nbs_doc_tags = [i for i,j in nbs]
            indexed_doctags = self.docvecs.indexed_doctags(nbs_doc_tags)
            nbs_indexes, nbs_vectors,nbs_locks, ignored = indexed_doctags
            #nbs_indexes = [self.docvecs.indexed_doctags(i) for i,j in nbs]
            #print("nbs_indexes",nbs_indexes)
            self.nb_list[ii] = nbs_indexes    
        #print(self.nb_list[:2])
    """
    def set_neighbor_list(self):

        docvecs = self.docvecs.doctag_syn0

        self.do_train +=1
        m,n = docvecs.shape
        p = self.num_neighbor
        if self.nb_list is None:
            self.nb_list = zeros((m,p))
        #print("set normalized vector empty!!")
        syn0norm = empty(self.docvecs.doctag_syn0.shape,dtype = REAL)
        print("start to get updated normalized vector!!")
        np_divide(self.docvecs.doctag_syn0, sqrt((self.docvecs.doctag_syn0**2).sum(-1))[..., newaxis], syn0norm)
        print("finished updating normalized vectors! start to search!")
        nn = [find_nn_idx(i,syn0norm) for i in range(m)]
        print("searched and copy the list to array!")
        self.nb_list = np.array(nn,copy = True)
        print(self.nb_list[:2])
        #print(type(self.nb_list))
        #print(self.nb_list.dtype) 
        #print(self.nb_list.shape)
                    
       
    def _do_train_job(self, job, alpha, inits):
        work, neu1 = inits
        tally = 0
        start_t = time.time()
        print("start do_train_job!")
        #print("worker?? ",inits)
        # one neighborhood update per job or 3000 document
        #if self.dbow_neighbors ==1 and self.do_train ==0:
        #    print("First, train job and update neighbor_list!")
        #    print("first_job = ",len(job))
        if self.do_train % 30 == 0:
            print("current_do_train =",self.do_train)
            self.set_neighbor_list()

            elapsed_t = time.time() -start_t
            
            print("first updated = ",elapsed_t)
            print("current_do_train_end =",self.do_train)
        #print("length of job = ",len(job))
        #print("count_do_train = ",self.do_train)
        #self.nb_list = zeros((len(self.docvecs),10))
        
        for doc in job:
            # print(doc)
            if tally ==0:
                print("start to get doctags!!!!!!!")
            #print("start to get doctags!")
            indexed_doctags = self.docvecs.indexed_doctags(doc.tags)
            doctag_indexes, doctag_vectors, doctag_locks, ignored = indexed_doctags
            """
            if self.dbow_neighbors==1 and self.do_train%3000 == 2999:
                print("do train job and update neighbor_list!")
                print("current do train for updating = ",self.do_train)
                start_t = time.time()
                self.set_neighbor_list()
                elapsed_t = time.time() - start_t
                print("neighborhood_undated time =",elapsed_t)
                print("after update, do_train_count = ",self.do_train)
            """
            if self.sg:
                tally += train_document_dbow(self, doc.words, doctag_indexes, alpha, work,
                                             train_words=self.dbow_words,
                                             doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            elif self.dbow_neighbors:
                #print("segmentation start?") 
                tally += train_document_dbow_neighbor(self, doc.words, doctag_indexes, alpha, beta = self.beta, work = work,
                                             train_words=self.dbow_words,
                                             doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
                # print('tally: ',tally)
                #print("segmentation end??")
            elif self.dm_concat:
                tally += train_document_dm_concat(self, doc.words, doctag_indexes, alpha, work, neu1,
                                                 doctag_vectors=doctag_vectors, doctag_locks=doctag_locks, learn_words=self.learn_word)
            
            else:
                tally += train_document_dm(self, doc.words, doctag_indexes, alpha, work, neu1,
                                           doctag_vectors=doctag_vectors, doctag_locks=doctag_locks, learn_words=self.learn_word)
            self.docvecs.trained_item(indexed_doctags)
        self.do_train += 1
        return tally, self._raw_word_count(job)
    
    def _raw_word_count(self, job):
        """Return the number of words in a given job."""
        return sum(len(sentence.words) for sentence in job)
    
    def infer_vector(self, doc_words, alpha=0.1, min_alpha=0.0001, steps=5):
        """
        Infer a vector for given post-bulk training document.
    
        Document should be a list of (word) tokens.
        """
        doctag_vectors = empty((1, self.vector_size), dtype=REAL)
        doctag_vectors[0] = self.seeded_vector(' '.join(doc_words))
        doctag_locks = ones(1, dtype=REAL)
        doctag_indexes = [0]
    
        work = zeros(self.layer1_size, dtype=REAL)
        if not self.sg:
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
    
        for i in range(steps):
            if self.sg:
                train_document_dbow(self, doc_words, doctag_indexes, alpha, work,
                                    learn_words=False, learn_hidden=False,
                                    doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            elif self.dm_concat:
                train_document_dm_concat(self, doc_words, doctag_indexes, alpha, work, neu1,
                                         learn_words=False, learn_hidden=False,
                                         doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            else:
                train_document_dm(self, doc_words, doctag_indexes, alpha, work, neu1,
                                  learn_words=False, learn_hidden=False,
                                  doctag_vectors=doctag_vectors, doctag_locks=doctag_locks)
            alpha = ((alpha - min_alpha) / (steps - i)) + min_alpha
    
        return doctag_vectors[0]
    
    def estimate_memory(self, vocab_size=None, report=None):
        """Estimate required memory for a model using current settings."""
        report = report or {}
        report['doctag_lookup'] = self.docvecs.estimated_lookup_memory()
        report['doctag_syn0'] = self.docvecs.count * self.vector_size * dtype(REAL).itemsize
        return super(Doc2Vec, self).estimate_memory(vocab_size, report=report)
    
    def __str__(self):
        """Abbreviated name reflecting major configuration paramaters."""
        segments = []
        if self.comment:
            segments.append('"%s"' % self.comment)
        if self.sg:
            if self.dbow_words:
                segments.append('dbow+w')  # also training words
            else:
                segments.append('dbow')  # PV-DBOW (skip-gram-style)
    
        else:  # PV-DM...
            if self.dm_concat:
                segments.append('dm/c')  # ...with concatenative context layer
            else:
                if self.cbow_mean:
                    segments.append('dm/m')
                else:
                    segments.append('dm/s')
        segments.append('d%d' % self.vector_size)  # dimensions
        if self.negative:
            segments.append('n%d' % self.negative)  # negative samples
        if self.hs:
            segments.append('hs')
        if not self.sg or (self.sg and self.dbow_words):
            segments.append('w%d' % self.window)  # window size, when relevant
        if self.min_count > 1:
            segments.append('mc%d' % self.min_count)
        if self.sample > 0:
            segments.append('s%g' % self.sample)
        if self.workers > 1:
            segments.append('t%d' % self.workers)
        return '%s(%s)' % (self.__class__.__name__, ','.join(segments))


class TaggedBrownCorpus(object):
    """Iterate over documents from the Brown corpus (part of NLTK data), yielding
    each document out as a TaggedDocument object."""
    def __init__(self, dirname):
        self.dirname = dirname
    
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            for item_no, line in enumerate(utils.smart_open(fname)):
                line = utils.to_unicode(line)
                # each file line is a single document in the Brown corpus
                # each token is WORD/POS_TAG
                token_tags = [t.split('/') for t in line.split() if len(t.split('/')) == 2]
                # ignore words with non-alphabetic tags like ",", "!" etc (punctuation, weird stuff)
                words = ["%s/%s" % (token.lower(), tag[:2]) for token, tag in token_tags if tag[:2].isalpha()]
                if not words:  # don't bother sending out empty documents
                    continue
                yield TaggedDocument(words, ['%s_SENT_%s' % (fname, item_no)])


class TaggedLineDocument(object):
    """Simple format: one document = one line = one TaggedDocument object.
    
    Words are expected to be already preprocessed and separated by whitespace,
    tags are constructed automatically from the document line number."""
    def __init__(self, source):
        """
        `source` can be either a string (filename) or a file object.
    
        Example::
    
            documents = TaggedLineDocument('myfile.txt')
    
        Or for compressed files::
    
            documents = TaggedLineDocument('compressed_text.txt.bz2')
            documents = TaggedLineDocument('compressed_text.txt.gz')
    
        """
        self.source = source
    
    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for item_no, line in enumerate(self.source):
                yield TaggedDocument(utils.to_unicode(line).split(), [item_no])
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.smart_open(self.source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [item_no])

class LoadTaggedDocument(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        for item_no, fname in enumerate(os.listdir(self.dirname)):            
            for line in open(os.path.join(self.dirname, fname), encoding="utf-8"):
                line = utils.to_unicode(line).lower()
                line = ''.join([ch for ch in line if ch not in punct])
                yield TaggedDocument(utils.to_unicode(line).split(), [item_no])

def model2array(model,case):
    data = []
    if case=='word':
        v = model.vocab
        words = list(v.keys())
        words.sort()
        for w in words:
            data.append(model[w])
    else:
        v = model.docvecs
        n = len(v)
        for i in range(n):
            data.append(v[i])
        
    return array(data)

if __name__ == "__main__":
    print(FAST_VERSION)
    tagged_sentences=LoadTaggedDocument(r'D:\Research\Journal\2015\1. Sentiment Analysis\Python\data\books\Sentences')
    
    word_model=word2vec.Word2Vec.load(r'D:\Research\Journal\2016\8. Doc2Vec\Python\Word2Vec\books_word_model_cbow')
    word_model_sswe=word2vec.Word2Vec.load(r'D:\Research\Journal\2016\8. Doc2Vec\Python\Word2Vec\books_word_sswe_fast')
    print(word_model.layer1_size)    
    doc_model_orig=Doc2Vec(tagged_sentences,size=100)
    print("doc_model_orig finish")
    doc_model=Doc2Vec(documents=tagged_sentences,learn_word=False,word_model=word_model)
    print("doc_model finish")
    doc_model_label=Doc2Vec(tagged_sentences,learn_word=False,word_model=word_model_sswe)
    print("doc_model_label finish")
