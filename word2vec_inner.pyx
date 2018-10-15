#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.math cimport log
from libc.string cimport memset, strcpy, strcat
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libc.stdio cimport FILE, fopen, fwrite, fscanf, fclose, fprintf
cimport libc.string

# scipy <= 0.15
try:
    from scipy.linalg.blas import fblas
except ImportError:
    # in scipy > 0.15, fblas function has been removed
    import scipy.linalg.blas as fblas

REAL = np.float32

DEF MAX_SENTENCE_LEN = 10000

cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x
cdef sgemv_ptr sgemv=<sgemv_ptr>PyCObject_AsVoidPtr(fblas.sgemv._cpointer) # float y= alpha * A * x + beta * y
cdef sgemm_ptr sgemm=<sgemm_ptr>PyCObject_AsVoidPtr(fblas.sgemm._cpointer) # float y= alpha * A * B + beta * C

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE
cdef REAL_t[EXP_TABLE_SIZE] LOG_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0
cdef REAL_t ONEF_M = <REAL_t> -1.0
cdef REAL_t ZEROF = <REAL_t>0.0

# for when fblas.sdot returns a double
cdef REAL_t our_dot_double(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>dsdot(N, X, incX, Y, incY)

# for when fblas.sdot returns a float
cdef REAL_t our_dot_float(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>sdot(N, X, incX, Y, incY)

# for when no blas available
cdef REAL_t our_dot_noblas(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    # not a true full dot()-implementation: just enough for our cases
    cdef int i
    cdef REAL_t a
    a = <REAL_t>0.0
    for i from 0 <= i < N[0] by 1:
        a += X[i] * Y[i]
    return a

# for when no blas available
cdef void our_saxpy_noblas(const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil:
    cdef int i
    for i from 0 <= i < N[0] by 1:
        Y[i * (incY[0])] = (alpha[0]) * X[i * (incX[0])] + Y[i * (incY[0])]
    
cdef void matrix_transpose(const int *M, const int *N, float *X, float *Y, const int *inc) nogil:
    cdef int i,j
    for i from 0 <= i < M[0] by 1:
        for j from 0 <= j < N[0] by 1:
            Y[j * (inc[0]) * (M[0]) + i * (inc[0])] = X[i * (inc[0]) * (M[0]) + j * (inc[0])]
    
cdef void our_multi_mat_vec(const char *T, const int *M, const int *N, const float *alpha, const float *A, const int *LDA, const float *X, const int *incX, const float *beta, const float *Y, const int *incY) nogil:
    sgemv(T,M,N,alpha,A,LDA,X,incX,beta,Y,incY)


cdef void our_multi_mat_mat(const char *T1, const char *T2, const int *M, const int *N, const int *K, const float *alpha, const float *A, const int *LDA, const float *B, const int *LDB, const float *beta, const float *C, const int *LDC) nogil:
    sgemm(T1,T2, M, N, K, alpha, A, LDA, B, LDB, beta, C, LDC)


cdef void hTanh(const int N, const REAL_t *V1, REAL_t *V2) nogil:
    cdef int i    

    for i from 0 <=i <N by 1:
        if V1[i]<-1.0:
            V2[i]=ONEF_M
        elif V1[i]>1.0:
            V2[i]=ONEF
        else:
            V2[i]=V1[i]

cdef void init_mat(const int M, const int N, REAL_t *A, const REAL_t value) nogil:
    cdef int i,j
    for i from 0 <=i < M by 1:
        for j from 0 <=j < N by 1:
            A[i+j*N]=value
    #free(A)    
    
cdef void fast_sentence_sg_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work, REAL_t *word_locks) nogil:

    cdef long long a, b
    cdef long long row1 = word2_index * size, row2
    cdef REAL_t f, g

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        row2 = word_point[b] * size
        f = our_dot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, &syn0[row1], &ONE, &syn1[row2], &ONE)
    our_saxpy(&size, &word_locks[word2_index], work, &ONE, &syn0[row1], &ONE)


# to support random draws from negative-sampling cum_table
cdef inline unsigned long long bisect_left(np.uint32_t *a, unsigned long long x, unsigned long long lo, unsigned long long hi) nogil:
    cdef unsigned long long mid
    while hi > lo:
        mid = (lo + hi) >> 1
        if a[mid] >= x:
            hi = mid
        else:
            lo = mid + 1
    return lo

# this quick & dirty RNG apparently matches Java's (non-Secure)Random
# note this function side-effects next_random to set up the next number
cdef inline unsigned long long random_int32(unsigned long long *next_random) nogil:
    cdef unsigned long long this_random = next_random[0] >> 16
    next_random[0] = (next_random[0] * <unsigned long long>25214903917ULL + 11) & 281474976710655ULL
    return this_random

cdef unsigned long long fast_sentence_sg_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *syn0, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random, REAL_t *word_locks) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f = our_dot(&size, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, &syn0[row1], &ONE, &syn1neg[row2], &ONE)

    our_saxpy(&size, &word_locks[word2_index], work, &ONE, &syn0[row1], &ONE)

    return next_random

cdef void fast_sentence_cbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, REAL_t *word_locks) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count = 1.0
    cdef int m

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        else:
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy?)

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelens[i]):
        row2 = word_point[b] * size
        f = our_dot(&size, neu1, &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)

    if not cbow_mean:  # divide error over summed window vectors
        sscal(&size, &inv_count, work, &ONE)  # (does this need BLAS-variants like saxpy?)

    for m in range(j, k):
        if m == i:
            continue
        else:
            our_saxpy(&size, &word_locks[indexes[m]], work, &ONE, &syn0[indexes[m] * size], &ONE)

cdef void fast_sentence_cbow_hs_sentiment(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, REAL_t *word_locks, const REAL_t *T) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count = 1.0
    cdef int m
    cdef REAL_t *inerr=<REAL_t *>malloc(size*cython.sizeof(REAL_t))
    cdef char trans = 'T'
    cdef char norm = 'N'
    #cdef REAL_t multiply
 
    # initialization
    init_mat(size,1,&inerr[0],ZEROF)
    
    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        else:
            count += ONEF
            our_multi_mat_vec(&trans,&size,&size,&ONEF,&T[0],&size,&syn0[indexes[m] * size],&ONE,&ONEF,neu1,&ONE) # neu1<-T*l0
            #our_saxpy(&size, &ONEF, &l0[0], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy?)

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelens[i]):
        row2 = word_point[b] * size        
        f = our_dot(&size, neu1, &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        # our_multi_mat_mat(&norm,&norm,&size,&size,&ONE,&multiply,&syn1[row2],&size,neu1,&ONE,&ONEF,&T[0],&size) #T=T+ obj_weight * alpha * matmul(g,l1)
        our_saxpy(&size, &g, &syn1[row2], &ONE, &inerr[0], &ONE) #inerr+=g*syn1[row2]
        # our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)
    
    # input gradient
    our_multi_mat_vec(&norm,&size,&size,&ONEF,&T[0],&size,&inerr[0],&ONE,&ONEF,work,&ONE) # work = work+obj_weight * matmul(T.T,inerr)
                
    if not cbow_mean:  # divide error over summed window vectors
        sscal(&size, &inv_count, work, &ONE)  # (does this need BLAS-variants like saxpy?)

    for m in range(j, k):
        if m == i:
            continue
        else:            
            our_saxpy(&size, &word_locks[indexes[m]], work, &ONE, &syn0[indexes[m] * size], &ONE)           

cdef unsigned long long fast_sentence_cbow_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1neg, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random, REAL_t *word_locks) nogil:

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count = 1.0, label
    cdef np.uint32_t target_index, word_index
    cdef int d, m

    word_index = indexes[i]

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        else:
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy?)

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    if not cbow_mean:  # divide error over summed window vectors
        sscal(&size, &inv_count, work, &ONE)  # (does this need BLAS-variants like saxpy?)

    for m in range(j,k):
        if m == i:
            continue
        else:
            our_saxpy(&size, &word_locks[indexes[m]], work, &ONE, &syn0[indexes[m]*size], &ONE)

    return next_random

cdef unsigned long long fast_sentence_cbow_neg_sentiment(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1neg, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random, REAL_t *word_locks, const REAL_t *T) nogil:

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count = 1.0, label
    cdef np.uint32_t target_index, word_index
    cdef int d, m
    cdef REAL_t *inerr=<REAL_t *>malloc(size*cython.sizeof(REAL_t))
    cdef char trans = 'T'
    cdef char norm = 'N'
    
    # initialization
    init_mat(size,1,&inerr[0],ZEROF)

    word_index = indexes[i]

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        else:
            count += ONEF
            our_multi_mat_vec(&trans,&size,&size,&ONEF,&T[0],&size,&syn0[indexes[m] * size],&ONE,&ONEF,neu1,&ONE) # neu1<-T*l0
            #our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy?)

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, &inerr[0], &ONE) #inerr+=g*syn1[row2]
        #our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    if not cbow_mean:  # divide error over summed window vectors
        sscal(&size, &inv_count, work, &ONE)  # (does this need BLAS-variants like saxpy?)
    
    # input gradient
    our_multi_mat_vec(&norm,&size,&size,&ONEF,&T[0],&size,&inerr[0],&ONE,&ONEF,work,&ONE) # work = work +  matmul(T.T,inerr)
    
    for m in range(j,k):
        if m == i:
            continue
        else:
            our_saxpy(&size, &word_locks[indexes[m]], work, &ONE, &syn0[indexes[m]*size], &ONE)

    return next_random 

cdef void derv_hTanh(const int n,const REAL_t *V1, const REAL_t *V2, REAL_t *V3) nogil:
    cdef int i    
    for i from 0<=i<n by 1:
        if V1[i]>1.0 or V1[i]<-1.0:
            V3[i]=ZEROF
        else:
            V3[i]=V2[i]
    
cdef void fast_sentence_sswe(const REAL_t label, REAL_t *syn0, const int layer1_size, const int vsize, const int window, const int nclass, const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work, const int i, const int j, REAL_t *word_locks, REAL_t *w1, REAL_t *b1, REAL_t *w2, REAL_t *b2) nogil:
    
    cdef int row, ind, pos1, pos2
    cdef int cv_size = vsize * window
    cdef int out_size = nclass-1
    cdef int tag = 1
    cdef REAL_t *l1=<REAL_t *>malloc(cv_size*cython.sizeof(REAL_t)) # input vector 
    cdef REAL_t *l2=<REAL_t *>malloc(layer1_size*cython.sizeof(REAL_t)) # hidden layer vector l2=w1*x+b1
    cdef REAL_t *l3=<REAL_t *>malloc(layer1_size*cython.sizeof(REAL_t)) # l3=hTanh(l2)
    cdef REAL_t *l4=<REAL_t *>malloc(out_size*cython.sizeof(REAL_t)) # l4=w2*l3+b2
    cdef REAL_t *l5=<REAL_t *>malloc(out_size*cython.sizeof(REAL_t)) # l5=logit(l4)
    
    # intermediate matrices
    cdef REAL_t *neuoe=<REAL_t *>malloc(out_size*cython.sizeof(REAL_t)) # error in output = label - l5
    cdef REAL_t *neu2e=<REAL_t *>malloc(layer1_size*cython.sizeof(REAL_t)) 
    cdef REAL_t *neuhte=<REAL_t *>malloc(layer1_size*cython.sizeof(REAL_t))
    
    cdef char trans='T'
    cdef char norm='N'
    cdef REAL_t alpha_M=-alpha # negatvie alpha
    cdef REAL_t lf_M 
    cdef REAL_t s=ZEROF
    
    # initialization of matrices
    init_mat(cv_size,1,&l1[0],ZEROF)
    init_mat(layer1_size,1,&l2[0],ZEROF)
    init_mat(layer1_size,1,&l3[0],ZEROF)
    init_mat(out_size,1,&l4[0],ZEROF)
    init_mat(layer1_size,1,&neu2e[0],ZEROF)
    init_mat(layer1_size,1,&neuhte[0],ZEROF)

    # errors on input vector
    memset(work, 0, cv_size * cython.sizeof(REAL_t))       
    
    ind = 0
    for pos1 from i<=pos1<j by 1:
        row = indexes[pos1] * vsize
        for pos2 from row<=pos2<row+vsize by 1:
            l1[ind] = syn0[pos2]
            ind += 1    
    
    our_multi_mat_vec(&trans,&cv_size,&layer1_size,&ONEF,&w1[0],&cv_size,&l1[0],&ONE,&ZEROF,&l2[0],&ONE) # l2<-w1*l1         
    our_saxpy(&layer1_size,&ONEF,&b1[0],&ONE,&l2[0],&ONE) #l2=l2+b1     
    hTanh(layer1_size,&l2[0],&l3[0]) # hTanh layer 
    our_multi_mat_vec(&trans,&layer1_size,&out_size,&ONEF,&w2[0],&layer1_size,&l3[0],&ONE,&ZEROF,&l4[0],&ONE) # l4=w2*l3
    our_saxpy(&out_size,&ONEF,&b2[0],&ONE,&l4[0],&ONE) #l4=l4+b2       
    
    for pos1 from 0<=pos1<out_size by 1:
        if l4[pos1] <= -MAX_EXP:
            l5[pos1] = ZEROF
        elif l4[pos1] >= MAX_EXP:
            l5[pos1] = ONEF
        else:
            l5[pos1] = EXP_TABLE[<int>((l4[pos1] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))] # logit(l4)
        if pos1==(label-1):
            neuoe[pos1] = l5[pos1]  - <REAL_t> 1.0 # error in output layer
        else:
            neuoe[pos1] = l5[pos1]
    #printf("target=%d, estimate=%f\n",label,l5[0])
    
    ## intermediate matrices for updating
    our_multi_mat_mat(&norm,&trans,&ONE,&layer1_size,&out_size,&ONEF,&neuoe[0],&ONE,&w2[0],&layer1_size,&ZEROF,&neu2e[0],&ONE) # neu2e=(w2^T*neuo)=layer1_size*1    
    derv_hTanh(layer1_size,&l2[0],&neu2e[0],&neuhte[0]) # neuhte=neu2e if -1.<=l2<=1. else 0
    our_multi_mat_mat(&norm,&trans,&ONE,&cv_size,&layer1_size,&ONEF,&neuhte[0],&ONE,&w1[0],&cv_size,&ZEROF,&work[0],&ONE) # work=(w1^T*neuhte)=cv_size*1
    
    ## update parameters
    our_multi_mat_mat(&norm,&norm,&layer1_size,&out_size,&ONE,&alpha_M,&l3[0],&layer1_size,&neuoe[0],&ONE,&ONEF,&w2[0],&layer1_size) #w2=w2-alpha*matmul(neuoe,l3)
    our_saxpy(&out_size,&alpha_M,&neuoe[0],&ONE,&b2[0],&ONE) #b2=b2-alpha*neuoe
    our_multi_mat_mat(&norm,&norm,&cv_size,&layer1_size,&ONE,&alpha_M,&l1[0],&cv_size,&neuhte[0],&ONE,&ONEF,&w1[0],&cv_size) #w1=w1-alpha*matmul(neuhte,l1)
    our_saxpy(&layer1_size,&alpha_M,&neuhte[0],&ONE,&b1[0],&ONE) #b2=b2-alpha*neuoe
    
    # update word vectors
    ind = 0
    for pos1 from i<=pos1<j by 1:
        row = indexes[pos1] * vsize
        lf_M = - word_locks[indexes[pos1]] * alpha
        our_saxpy(&vsize,&lf_M,&work[ind*vsize],&ONE,&syn0[row],&ONE)        
        ind+=1
    
cdef void fast_sentence_sswe_bow(const REAL_t label, REAL_t *syn0, const int layer1_size, const int vsize, const int window, const int nclass, const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work, const int i, const int j, REAL_t *word_locks, REAL_t *w1, REAL_t *b1, REAL_t *w2, REAL_t *b2) nogil:
    
    cdef int row, ind, pos1, pos2
    cdef int cv_size = vsize
    cdef int out_size = nclass-1
    cdef int tag = 1
    cdef REAL_t *l1=<REAL_t *>malloc(cv_size*cython.sizeof(REAL_t)) # input vector 
    cdef REAL_t *l2=<REAL_t *>malloc(layer1_size*cython.sizeof(REAL_t)) # hidden layer vector l2=w1*x+b1
    cdef REAL_t *l3=<REAL_t *>malloc(layer1_size*cython.sizeof(REAL_t)) # hTanh(l2)
    cdef REAL_t *l3_mask=<REAL_t *>malloc(layer1_size*cython.sizeof(REAL_t)) # derivatives of hTanh(l2)
    cdef REAL_t *l4=<REAL_t *>malloc(out_size*cython.sizeof(REAL_t)) # l4=w2*l3+b2
    cdef REAL_t *l5=<REAL_t *>malloc(out_size*cython.sizeof(REAL_t)) # l5=logit(l4)
    
    # intermediate matrices
    cdef REAL_t *neuoe=<REAL_t *>malloc(out_size*cython.sizeof(REAL_t)) # error in output = label - l5
    cdef REAL_t *neu2e=<REAL_t *>malloc(layer1_size*cython.sizeof(REAL_t)) 
    cdef REAL_t *neuhte=<REAL_t *>malloc(layer1_size*cython.sizeof(REAL_t))
    
    cdef char trans='T'
    cdef char norm='N'
    cdef REAL_t alpha_M=-alpha # negatvie alpha
    cdef REAL_t lf_M 
    cdef REAL_t s=ZEROF
    cdef REAL_t inv_count = 1.0
    
    # initialization of matrices
    init_mat(cv_size,1,&l1[0],ZEROF)
    init_mat(layer1_size,1,&l2[0],ZEROF)
    init_mat(layer1_size,1,&l3[0],ZEROF)
    init_mat(out_size,1,&l4[0],ZEROF)
    init_mat(layer1_size,1,&neu2e[0],ZEROF)
    init_mat(layer1_size,1,&neuhte[0],ZEROF)
       
    # errors on input vector
    memset(work, 0, cv_size * cython.sizeof(REAL_t))       
    
    for pos1 from i<=pos1<j by 1:
        row = indexes[pos1] * vsize
        our_saxpy(&cv_size,&ONEF,&syn0[row],&ONE,&l1[0],&ONE)        
    
    inv_count = ONEF/ window
    sscal(&cv_size, &inv_count, &l1[0], &ONE)    
    
    our_multi_mat_vec(&trans,&cv_size,&layer1_size,&ONEF,&w1[0],&cv_size,&l1[0],&ONE,&ZEROF,&l2[0],&ONE) # l2<-w1*l1         
    our_saxpy(&layer1_size,&ONEF,&b1[0],&ONE,&l2[0],&ONE) #l2=l2+b1     
    hTanh(layer1_size,&l2[0],&l3[0]) # hTanh layer 
    our_multi_mat_vec(&trans,&layer1_size,&out_size,&ONEF,&w2[0],&layer1_size,&l3[0],&ONE,&ZEROF,&l4[0],&ONE) # l4=w2*l3
    our_saxpy(&out_size,&ONEF,&b2[0],&ONE,&l4[0],&ONE) #l4=l4+b2       
    
    for pos1 from 0<=pos1<out_size by 1:
        if l4[pos1] <= -MAX_EXP:
            l5[pos1] = ZEROF
        elif l4[pos1] >= MAX_EXP:
            l5[pos1] = ONEF
        else:
            l5[pos1] = EXP_TABLE[<int>((l4[pos1] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))] # logit(l4)
        if pos1==(label-1):
            neuoe[pos1] = l5[pos1]  - <REAL_t> 1.0 # error in output layer
        else:
            neuoe[pos1] = l5[pos1]            
        
    ## intermediate matrices for updating
    our_multi_mat_mat(&norm,&trans,&ONE,&layer1_size,&out_size,&ONEF,&neuoe[0],&ONE,&w2[0],&layer1_size,&ZEROF,&neu2e[0],&ONE) # neu2e=(w2^T*neuo)=layer1_size*1 
    derv_hTanh(layer1_size,&l2[0],&neu2e[0],&neuhte[0]) # neuhte=neu2e if -1.<=l2<=1. else 0
    our_multi_mat_mat(&norm,&trans,&ONE,&cv_size,&layer1_size,&ONEF,&neuhte[0],&ONE,&w1[0],&cv_size,&ZEROF,&work[0],&ONE) # work=(w1^T*neuhte)=cv_size*1

    ## update parameters
    our_multi_mat_mat(&norm,&norm,&layer1_size,&out_size,&ONE,&alpha_M,&l3[0],&layer1_size,&neuoe[0],&ONE,&ONEF,&w2[0],&layer1_size) #w2=w2-alpha*matmul(neuoe,l3)
    our_saxpy(&out_size,&alpha_M,&neuoe[0],&ONE,&b2[0],&ONE) #b2=b2-alpha*neuoe
    our_multi_mat_mat(&norm,&norm,&cv_size,&layer1_size,&ONE,&alpha_M,&l1[0],&cv_size,&neuhte[0],&ONE,&ONEF,&w1[0],&cv_size) #w1=w1-alpha*matmul(neuhte,l1)
    our_saxpy(&layer1_size,&alpha_M,&neuhte[0],&ONE,&b1[0],&ONE) #b2=b2-alpha*neuoe
    
    # update word vectors
    for pos1 from i<=pos1<j by 1:
        row = indexes[pos1] * vsize
        lf_M = - word_locks[indexes[pos1]] * alpha
        our_saxpy(&vsize,&lf_M,&work[0],&ONE,&syn0[row],&ONE)

def train_batch_sswe(model, sentences, alpha, _work):
    cdef int sample = (model.sample != 0)
    
    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.syn0_lockf))
    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    
    cdef int layer1_size = model.layer1_size
    cdef int vector_size = model.vector_size
    cdef int nclass = model.nclass
    
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef REAL_t sentence_label[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window
    
    cdef int i, j
    cdef int effective_words = 0, effective_sentences = 0, sent_words = 0
    cdef int sent_idx, idx_start, idx_end
    
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random, this_random
    next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)
    
    cdef REAL_t *w1 = <REAL_t *>(np.PyArray_DATA(model.nn_parms['w1']))        
    cdef REAL_t *b1 = <REAL_t *>(np.PyArray_DATA(model.nn_parms['b1']))
    cdef REAL_t *w2 = <REAL_t *>(np.PyArray_DATA(model.nn_parms['w2']))
    cdef REAL_t *b2 = <REAL_t *>(np.PyArray_DATA(model.nn_parms['w2']))
    
    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)  
       
    vlookup = model.vocab
    sentence_idx[0] = 0    
    for sent, label in sentences:
        if not sent : # ignore empty sentences; leave effective_sentences unchanged
            continue
        for token in sent:
            word = vlookup[token] if token in vlookup else None
            this_random = random_int32(&next_random)
            if word is None:
                continue
            if sample and word.sample_int < this_random:
                continue
            indexes[effective_words] = word.index
            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break
        sentence_label[effective_sentences] = <REAL_t> label
        effective_sentences += 1
        #printf("label=%f ",sentence_label[effective_sentences-1])
        sentence_idx[effective_sentences] = effective_words
        #print("label:",label)
        if effective_words == MAX_SENTENCE_LEN:
            break
    
    with nogil:
        for sent_idx in range(effective_sentences):
            idx_start = sentence_idx[sent_idx]
            idx_end = sentence_idx[sent_idx + 1]
            if (idx_end-idx_start)>=window:
                for i in range(idx_start,idx_end - window + 1):
                    j = i + window
                    fast_sentence_sswe(sentence_label[sent_idx],syn0,layer1_size, vector_size, window, nclass, indexes, _alpha, work, i, j, word_locks, w1, b1, w2, b2)    
    return effective_words
    
def train_batch_sswe_bow(model, sentences, alpha, _work):
    cdef int sample = (model.sample != 0)
    
    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.syn0_lockf))
    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    
    cdef int layer1_size = model.layer1_size
    cdef int vector_size = model.vector_size
    cdef int nclass = model.nclass
    
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef REAL_t sentence_label[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window
    
    cdef int i, j
    cdef int effective_words = 0, effective_sentences = 0, sent_words = 0
    cdef int sent_idx, idx_start, idx_end
    
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random, this_random
    next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)
       
    cdef REAL_t *w1 = <REAL_t *>(np.PyArray_DATA(model.nn_parms['w1']))        
    cdef REAL_t *b1 = <REAL_t *>(np.PyArray_DATA(model.nn_parms['b1']))
    cdef REAL_t *w2 = <REAL_t *>(np.PyArray_DATA(model.nn_parms['w2']))
    cdef REAL_t *b2 = <REAL_t *>(np.PyArray_DATA(model.nn_parms['w2']))
    
    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)  
    
    vlookup = model.vocab
    sentence_idx[0] = 0    
    for sent, label in sentences:
        if not sent : # ignore empty sentences; leave effective_sentences unchanged
            continue
        for token in sent:
            word = vlookup[token] if token in vlookup else None
            this_random = random_int32(&next_random)
            if word is None:
                continue
            if sample and word.sample_int < this_random:
                continue
            indexes[effective_words] = word.index
            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break
        sentence_label[effective_sentences] = <REAL_t> label
        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words
        #print("label:",label)
        if effective_words == MAX_SENTENCE_LEN:
            break

    with nogil:
        for sent_idx in range(effective_sentences):
            idx_start = sentence_idx[sent_idx]
            idx_end = sentence_idx[sent_idx + 1]
            if (idx_end-idx_start)>=window:
                for i in range(idx_start,idx_end - window + 1):
                    j = i + window
                    fast_sentence_sswe_bow(sentence_label[sent_idx],syn0, layer1_size, vector_size, window, nclass, indexes, _alpha, work, i, j, word_locks, w1, b1, w2, b2)
    return effective_words
    
def train_batch_sg(model, sentences, alpha, _work):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.sample != 0)

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.syn0_lockf))
    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.cum_table))
        cum_table_len = len(model.cum_table)
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)

    # prepare C structures so we can go "full C" and release the Python GIL
    vlookup = model.vocab
    sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word = vlookup[token] if token in vlookup else None
            if word is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if sample and word.sample_int < random_int32(&next_random):
                continue
            indexes[effective_words] = word.index
            if hs:
                codelens[effective_words] = <int>len(word.code)
                codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[effective_words] = <np.uint32_t *>np.PyArray_DATA(word.point)
            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?

    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, window, effective_words)):
        reduced_windows[i] = item

    # release GIL & train on all sentences
    with nogil:
        for sent_idx in range(effective_sentences):
            idx_start = sentence_idx[sent_idx]
            idx_end = sentence_idx[sent_idx + 1]
            for i in range(idx_start, idx_end):
                j = i - window + reduced_windows[i]
                if j < idx_start:
                    j = idx_start
                k = i + window + 1 - reduced_windows[i]
                if k > idx_end:
                    k = idx_end
                for j in range(j, k):
                    if j == i:
                        continue
                    if hs:
                        fast_sentence_sg_hs(points[i], codes[i], codelens[i], syn0, syn1, size, indexes[j], _alpha, work, word_locks)
                    if negative:
                        next_random = fast_sentence_sg_neg(negative, cum_table, cum_table_len, syn0, syn1neg, size, indexes[i], indexes[j], _alpha, work, next_random, word_locks)

    return effective_words

def train_batch_cbow(model, sentences, alpha, _work, _neu1):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.sample != 0)
    cdef int cbow_mean = model.cbow_mean

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.syn0_lockf))
    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.cum_table))
        cum_table_len = len(model.cum_table)
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)

    # prepare C structures so we can go "full C" and release the Python GIL
    vlookup = model.vocab
    sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word = vlookup[token] if token in vlookup else None
            if word is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if sample and word.sample_int < random_int32(&next_random):
                continue
            indexes[effective_words] = word.index
            if hs:
                codelens[effective_words] = <int>len(word.code)
                codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[effective_words] = <np.uint32_t *>np.PyArray_DATA(word.point)
            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?

    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, window, effective_words)):
        reduced_windows[i] = item

    # release GIL & train on all sentences
    with nogil:
        for sent_idx in range(effective_sentences):
            idx_start = sentence_idx[sent_idx]
            idx_end = sentence_idx[sent_idx + 1]
            for i in range(idx_start, idx_end):
                j = i - window + reduced_windows[i]
                if j < idx_start:
                    j = idx_start
                k = i + window + 1 - reduced_windows[i]
                if k > idx_end:
                    k = idx_end
                if hs:
                    fast_sentence_cbow_hs(points[i], codes[i], codelens, neu1, syn0, syn1, size, indexes, _alpha, work, i, j, k, cbow_mean, word_locks)
                if negative:
                    next_random = fast_sentence_cbow_neg(negative, cum_table, cum_table_len, codelens, neu1, syn0, syn1neg, size, indexes, _alpha, work, i, j, k, cbow_mean, next_random, word_locks)

    return effective_words

def train_batch_cbow_sentiment(model, sentences, alpha, obj_weight, _work, _neu1):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.sample != 0)
    cdef int cbow_mean = model.cbow_mean

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.syn0_lockf))
    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    cdef REAL_t _obj_weight = obj_weight
    cdef REAL_t _weight_sswe = _alpha * (1.0-_obj_weight)
    cdef REAL_t _weight_cbow = _alpha * _obj_weight
    
    cdef int layer1_size = model.layer1_size
    cdef int vector_size = model.vector_size
    cdef int nclass = model.nclass

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef REAL_t sentence_label[MAX_SENTENCE_LEN + 1]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window
    cdef int slide = window/2

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.cum_table))
        cum_table_len = len(model.cum_table)
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)
    
    # SSWE
    cdef REAL_t *w1 = <REAL_t *>(np.PyArray_DATA(model.nn_parms['w1']))        
    cdef REAL_t *b1 = <REAL_t *>(np.PyArray_DATA(model.nn_parms['b1']))
    cdef REAL_t *w2 = <REAL_t *>(np.PyArray_DATA(model.nn_parms['w2']))
    cdef REAL_t *b2 = <REAL_t *>(np.PyArray_DATA(model.nn_parms['w2']))
    
    # transformation matrix
    cdef REAL_t *T = <REAL_t *>(np.PyArray_DATA(model.nn_parms['T']))    
            
    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)

    # prepare C structures so we can go "full C" and release the Python GIL
    vlookup = model.vocab
    sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent, label in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word = vlookup[token] if token in vlookup else None
            if word is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if sample and word.sample_int < random_int32(&next_random):
                continue
            indexes[effective_words] = word.index
            if hs:
                codelens[effective_words] = <int>len(word.code)
                codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[effective_words] = <np.uint32_t *>np.PyArray_DATA(word.point)
            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?
            
        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        sentence_label[effective_sentences] = <REAL_t> label
        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?
   
    # release GIL & train on all sentences
    with nogil:
        for sent_idx in range(effective_sentences):
            idx_start = sentence_idx[sent_idx]
            idx_end = sentence_idx[sent_idx + 1]
            if (idx_end-idx_start)>=window:
                for j in range(idx_start,idx_end - window + 1):
                    k = j + window
                    i = j + slide
                    if hs:
                        fast_sentence_cbow_hs_sentiment(points[i], codes[i], codelens, neu1, syn0, syn1, vector_size, indexes, _weight_cbow, work, i, j, k, cbow_mean, word_locks, T)
                        #fast_sentence_cbow_hs(points[i], codes[i], codelens, neu1, syn0, syn1, vector_size, indexes, _weight_cbow, work, i, j, k, cbow_mean, word_locks)
                    if negative:
                        next_random = fast_sentence_cbow_neg_sentiment(negative, cum_table, cum_table_len, codelens, neu1, syn0, syn1neg, vector_size, indexes, _weight_cbow, work, i, j, k, cbow_mean, next_random, word_locks, T)
                        #next_random = fast_sentence_cbow_neg(negative, cum_table, cum_table_len, codelens, neu1, syn0, syn1neg, vector_size, indexes, _weight_cbow, work, i, j, k, cbow_mean, next_random, word_locks)
                    fast_sentence_sswe(sentence_label[sent_idx],syn0,layer1_size,vector_size, window, nclass, indexes, _weight_sswe, work, i, j, word_locks, w1, b1, w2, b2)    

    return effective_words
    
# Score is only implemented for hierarchical softmax
def score_sentence_sg(model, sentence, _work):

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *work
    cdef int size = model.layer1_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)

    vlookup = model.vocab
    i = 0
    for token in sentence:
        word = vlookup[token] if token in vlookup else None
        if word is None:
            continue  # should drop the
        indexes[i] = word.index
        codelens[i] = <int>len(word.code)
        codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
        points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
        result += 1
        i += 1
        if i == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?
    sentence_len = i

    # release GIL & train on the sentence
    work[0] = 0.0

    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window
            if j < 0:
                j = 0
            k = i + window + 1
            if k > sentence_len:
                k = sentence_len
            for j in range(j, k):
                if j == i or codelens[j] == 0:
                    continue
                score_pair_sg_hs(points[i], codes[i], codelens[i], syn0, syn1, size, indexes[j], work)

    return work[0]

cdef void score_pair_sg_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, REAL_t *work) nogil:

    cdef long long b
    cdef long long row1 = word2_index * size, row2, sgn
    cdef REAL_t f

    for b in range(codelen):
        row2 = word_point[b] * size
        f = our_dot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        sgn = (-1)**word_code[b] # ch function: 0-> 1, 1 -> -1
        f = sgn*f
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = LOG_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        work[0] += f

def score_sentence_cbow(model, sentence, _work, _neu1):
    cdef int cbow_mean = model.cbow_mean

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *work
    cdef REAL_t *neu1
    cdef int size = model.layer1_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)

    vlookup = model.vocab
    i = 0
    for token in sentence:
        word = vlookup[token] if token in vlookup else None
        if word is None:
            continue  # for score, should this be a default negative value?
        indexes[i] = word.index
        codelens[i] = <int>len(word.code)
        codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
        points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
        result += 1
        i += 1
        if i == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?
    sentence_len = i

    # release GIL & train on the sentence
    work[0] = 0.0
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window
            if j < 0:
                j = 0
            k = i + window + 1
            if k > sentence_len:
                k = sentence_len
            score_pair_cbow_hs(points[i], codes[i], codelens, neu1, syn0, syn1, size, indexes, work, i, j, k, cbow_mean)

    return work[0]

cdef void score_pair_cbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], REAL_t *work,
    int i, int j, int k, int cbow_mean) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count, sgn
    cdef int m

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)

    for b in range(codelens[i]):
        row2 = word_point[b] * size
        f = our_dot(&size, neu1, &ONE, &syn1[row2], &ONE)
        sgn = (-1)**word_code[b] # ch function: 0-> 1, 1 -> -1
        f = sgn*f
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = LOG_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        work[0] += f


def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.  Also calculate log(sigmoid(x)) into LOG_TABLE.

    """
    global our_dot
    global our_saxpy

    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))
        LOG_TABLE[i] = <REAL_t>log( EXP_TABLE[i] )

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res
    if (abs(d_res - expected) < 0.0001):
        our_dot = our_dot_double
        our_saxpy = saxpy
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
        our_dot = our_dot_float
        our_saxpy = saxpy
        return 1  # float
    else:
        # neither => use cython loops, no BLAS
        # actually, the BLAS is so messed up we'll probably have segfaulted above and never even reach here
        our_dot = our_dot_noblas
        our_saxpy = our_saxpy_noblas
        return 2

FAST_VERSION = init()  # initialize the module
MAX_WORDS_IN_BATCH = MAX_SENTENCE_LEN
