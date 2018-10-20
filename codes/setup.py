# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 09:00:09 2016

@author: Administrator
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext  =  [Extension( "word2vec_inner", sources=["word2vec_inner.pyx"] )]

setup(cmdclass={'build_ext' : build_ext}, include_dirs = [numpy.get_include()],ext_modules=ext
)