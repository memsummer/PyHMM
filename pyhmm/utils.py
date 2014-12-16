# Simple HMM package in Python (PyHMM)
#
# Author: Zhiyong You <zhiyongy0@outlook.com>
#

'''
Some utility functions for PyHMM package.
'''

__all__ = ['lmap', 'vector',
           'matrix', 'matrix3',
           'getopt']

def lmap(fun, listdata):
    '''
    map function for list data, it is just like system's map functions except 
    that it will NOT create new object but change data in it's original
    location.
    '''

    for i in range(len(listdata)):
        listdata[i] = fun(listdata[i])
    return listdata

def vector(n):
    '''
    create a vector(which is list).
    '''

    return [0 for i in range(n)]

def matrix(n, m):
    '''
    create a matrix(which is list).
    '''

    return [[0 for j in range(m)] for i in range(n)]

def matrix3(a, b, c):
    '''
    create a matrix(which is list).
    '''

    return [[[0 for k in range(c)] for j in range(b)] for i in range(a)]

def getopt():
    '''
    simple solution for processing command line arguments, just like getopt()
    call in C++/C. For example, a program is called by
    '-v -N 3 -M 2 file.seq', then call the getopt():

        import pyhmm
        opt = pyhmm.getopt()

    and the returned dictionary 'opt' should be:
        opt ==> {
            'argc' : 7,
            'argv' : '-v -N 3 -M 2 file.seq',
            '-v' : None,
            '-N' : '3',
            '-M' : '2',
            'entity' : ['file.seq']}.
    '''
    import sys

    opt = {}
    opt['argc'] = len(sys.argv)
    opt['argv'] = list(sys.argv)
    opt['entity'] = []

    N = opt['argc']
    i = 1
    while i < N:
        arg = sys.argv[i]
        if arg.startswith('-'):
            if i+1<N:
                nextarg = sys.argv[i+1]
            else:
                nextarg = None
            if nextarg and nextarg.startswith('-'):
                opt[arg] = None
                i += 1
            else:
                opt[arg] = nextarg
                i += 2
        else:
            opt['entity'].append(arg)
            i += 1
    return opt

