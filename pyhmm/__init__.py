# Simple HMM package in Python (PyHMM)
#
# Author: Zhiyong You <zhiyongy0@outlook.com>
#


'''
Simple HMM package in Python (PyHMM). 

PyHMM -- Simple HMM package in Python -- is a simple HMM package writed in
Python Programming Language. It implements the forward-backward algorithm
and Viterbi algorithm. It also provides some scripts to demonstrate how to
solve the three problems -- 'Problem 1', 'Problem 2' and 'Problem 3' ( which
are come from the book SLP*, more about the book, see notes bellow) -- in
HMM, these scripts are also writed in Python.

This pakcage is created just for learning purpose. If you want to do some
real jobs, you may want to implement the algorithms in C/C++ and call them
in Python.

The algorithms it has implemented include:
    Forward           (problem 1 according to SLP*)
    ForwardWithScale  (scale method borrows from UMDHMM* package)
    Backward
    BackwardWithScale (scale method borrows from UMDHMM* package)
    Viterbi           (problem 2 according to SLP*)
    Baumwelch         (problem 3 according to SLP*, train an HMM model)
    RandomInitHMM
    ReadHMM           (file format borrows from UMDHMM* package)
    ReadSeq
    PrintHMM

It also implements some utility functions:
    lmap
    vector
    matrix
    matrix3
    getopt
    
Notes:
    SLP: The book, Speech and Language Processing: An introduction to
	natural language processing, computational linguistics, and speech
	recognition, draft edition of June 25, 2007, by Daniel Jurafsky &
	James H. Martin.
    UMDHMM: The UMDHMM package, implemented in C, Version 1.02 (19 February,
	1998), by Tapas Kanungo(kanungo@cfar.umd.edu), the web is 
	http://www.cfar.umd.edu/~kanungo

Contribution:
    Feedback is always welcome ;-)

'''

__version__ = '0.1.0'
__author__ = 'Zhiyong You <zhiyongy0@outlook.com>'
__all__ = ['HMM',
           'Forward', 'ForwardWithScale',
           'Backward', 'BackwardWithScale',
           'Viterbi', 'Baumwelch',
           'RandomInitHMM', 'ReadHMM',
           'ReadSeq', 'PrintHMM',
           'lmap', 'vector',
           'matrix', 'matrix3',
           'getopt']

from pyhmm.hmm import *
from pyhmm.utils import *

