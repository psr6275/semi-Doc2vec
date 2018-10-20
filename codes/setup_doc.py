
from distutils.core import setup
from Cython.Build import cythonize
import numpy
#ext_modules = [Extension()]
setup(ext_modules = cythonize("doc2vec_inner.pyx"),include_dirs = [numpy.get_include()])

