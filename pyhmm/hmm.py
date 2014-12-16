# Simple HMM package in Python (PyHMM)
#
# Author: Zhiyong You <zhiyongy0@outlook.com>
# 

'''
Simple HMM package in Python (PyHMM). 

Implementing the forward-backward algorithm and viterbi algorithm for HMM.

'''

import math
import random
from pyhmm.utils import lmap, matrix, vector, matrix3

__all__ = ['HMM',
           'Forward', 'ForwardWithScale',
           'Backward', 'BackwardWithScale',
           'Viterbi', 'Baumwelch',
           'RandomInitHMM', 'ReadHMM',
           'ReadSeq', 'PrintHMM']

class HMM():
    '''
    HMM = (N, M, pi, A, B).

    N:  number of states
    M:  number of observations
    pi: initial probability distribution
    A:  transition probability matrix
    B:  emission probability matrix
    
    States are encoded in [0..N-1], and observations are encoded in [0..M-1].
    '''

    N = 0
    M = 0
    pi = None
    A = None
    B = None

def Forward(hmm, O, alpha):
    '''
    forward algorithm for HMM, alpha matrix will contain forward temporary
    teril values, which alpha[t][i] = Prob(o0, o1, ..., ot, qt=i | model).
    '''

    N = hmm.N
    pi = hmm.pi
    A = hmm.A
    B = hmm.B
    T = len(O)

    for i in range(N):
        alpha[0][i] = pi[i] * B[i][O[0]]

    for t in range(1, T):
        for j in range(N):
            alpha[t][j] = 0
            for i in range(N):
                alpha[t][j] += alpha[t-1][i] * A[i][j] * B[j][O[t]]

    probf = sum(alpha[-1])

    return math.log(probf)

def ForwardWithScale(hmm, O, alpha, scale):
    '''
    forward algorithm for HMM with scaling.
    '''

    N = hmm.N
    pi = hmm.pi
    A = hmm.A
    B = hmm.B
    T = len(O)

    scale[0] = 0
    for i in range(N):
        alpha[0][i] = pi[i] * B[i][O[0]]
        scale[0] += alpha[0][i]

    for i in range(N):
        alpha[0][i] /= scale[0]

    for t in range(1, T):
        scale[t] = 0
        for j in range(N):
            alpha[t][j] = 0
            for i in range(N):
                alpha[t][j] += alpha[t-1][i] * A[i][j] * B[j][O[t]]
            scale[t] += alpha[t][j]

        for j in range(N):
            alpha[t][j] /= scale[t]

    probf = 0
    for t in range(T):
        probf += math.log(scale[t])

    return probf

def Backward(hmm, O, beta):
    '''
    backward algorithm for HMM, beta matrix will contain backward temporary
    teril values, which beta[t][i] = Prob(ot+1, ot+2, ..., oT-1, qt=i | model).
    '''

    N = hmm.N
    pi = hmm.pi
    A = hmm.A
    B = hmm.B
    T = len(O)

    for j in range(N):
        beta[T-1][j] = 1

    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t][i] = 0
            for j in range(N):
                beta[t][i] += A[i][j] * B[j][O[t+1]] * beta[t+1][j]

    probb = 0
    for j in range(N):
        probb += pi[j] * B[j][O[0]] * beta[0][j]

    return math.log(probb)

def BackwardWithScale(hmm, O, beta, scale):
    '''
    backward algorithm for HMM with scaling, scale is computed by
    ForwardWithScale.
    '''

    N = hmm.N
    A = hmm.A
    B = hmm.B
    T = len(O)

    for j in range(N):
        beta[T-1][j] = 1/scale[T-1]

    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t][i] = 0
            for j in range(N):
                beta[t][i] += A[i][j] * B[j][O[t+1]] * beta[t+1][j]
            beta[t][i] /= scale[t]

def Viterbi(hmm, O):
    '''
    Viterbi algorithm for HMM. Finding the optimal state sequences for an 
    observations O, given an HMM model, which
    Q = arg_{q0, q1, ..., qT-1} max { Prob(O| q0, q1, ..., qT-1, model) }.
    '''

    N = hmm.N
    pi = hmm.pi
    A = hmm.A
    B = hmm.B
    T = len(O)

    v = matrix(T, N)
    jumpfrom = matrix(T, N)

    for i in range(N):
        v[0][i] = pi[i] * B[i][O[0]]

    for t in range(1, T):
        for j in range(N):
            for i in range(N):
                if v[t-1][i] * A[i][j] * B[j][O[t]] > v[t][j]:
                    v[t][j] = v[t-1][i] * A[i][j] * B[j][O[t]]
                    jumpfrom[t][j] = i

    # trace the state sequence
    seq = []
    prob = max(v[-1])
    j = v[-1].index(prob)
    for t in range(T-1, -1, -1):
        seq.insert(0, j)
        j = jumpfrom[t][j]

    return (math.log(prob), seq)

def RandomInitHMM(N, M):
    '''
    Initializing HMM parameters -- pi, A, B -- randomly, should be used for 
    Baum-Welch algorithm for training an HMM.
    '''

    hmm = HMM()
    hmm.N = N
    hmm.M = M

    pi = [random.random() for i in range(N)]
    dsum = sum(pi)
    lmap(lambda x: x/dsum, pi)
    hmm.pi = pi

    A = [[random.random() for j in range(N)] for i in range(N)]
    for i in range(N):
        dsum = sum(A[i])
        lmap(lambda x: x/dsum, A[i])
    hmm.A = A

    B = [[random.random() for j in range(M)] for i in range(N)]
    for i in range(N):
        dsum = sum(B[i])
        lmap(lambda x: x/dsum, B[i])
    hmm.B = B

    return hmm

def Computexi(hmm, O, alpha, deta, xi):
    '''
    Compute xi matrix according to alpha and deta matrix.
    '''

    N = hmm.N
    A = hmm.A
    B = hmm.B
    T = len(O)

    for t in range(T-1):
        dsum = 0
        for i in range(N):
            for j in range(N):
                xi[t][i][j] = alpha[t][i] * A[i][j] * \
                        B[j][O[t+1]] * deta[t+1][j]
                dsum += xi[t][i][j]
        for i in range(N):
            lmap(lambda x: x/dsum, xi[t][i])

def Computegamma(hmm, O, alpha, deta, gamma):
    '''
    Compute gamma matrix according to alpha and deta matrix.
    '''

    N = hmm.N
    A = hmm.A
    B = hmm.B
    T = len(O)

    for t in range(T):
        dsum = 0
        for j in range(N):
            gamma[t][j] = alpha[t][j] * deta[t][j]
            dsum += gamma[t][j]
        lmap(lambda x: x/dsum, gamma[t])

def Baumwelch(hmm, O):
    '''
    Baum-Welch algorithm for HMM. Using maximum likelihood estimate (MLE) 
    method to find optimal model, which
    model = arg_{model} max { Prob(O| model) }.
    '''
    
    N = hmm.N
    M = hmm.M
    A = hmm.A
    B = hmm.B
    pi = hmm.pi
    T = len(O)
    DELTA = 0.001

    alpha = matrix(T, N)
    scale = vector(T)
    beta = matrix(T, N)
    xi = matrix3(T, N, N)
    gamma = matrix(T, N)

    logprobf = ForwardWithScale(hmm, O, alpha, scale)
    BackwardWithScale(hmm, O, beta, scale)
    Computexi(hmm, O, alpha, beta, xi)
    Computegamma(hmm, O, alpha, beta, gamma)

    logprobprev = -DELTA
    logprobold = logprobf
    iiter = 0

    while abs(logprobf-logprobprev) > DELTA:

        for i in range(N):
            pi[i] = .001 + .999 * gamma[0][i]

        for i in range(N):
            for j in range(N):
                # It should be noted that the item
                # 'sum(xi[t][i][j] for t in range(T-1) for j in range(N))'
                # has the same value as the item
                # 'sum(gamma[t][i] for t in range(T-1))',
                # but the later is more computational cheap.
                #A[i][j] = .001 + .999 * \
                #        sum(xi[t][i][j] for t in range(T-1)) / \
                #        sum(xi[t][i][j] for t in range(T-1) for j in range(N))
                A[i][j] = .001 + .999 * \
                        sum(xi[t][i][j] for t in range(T-1)) / \
                        sum(gamma[t][i] for t in range(T-1))

        for j in range(N):
            for k in range(M):
                B[j][k] = .001 + .999 * \
                        sum(gamma[t][j] for t in range(T) if O[t] == k) / \
                        sum(gamma[t][j] for t in range(T))

        logprobprev = logprobf

        logprobf = ForwardWithScale(hmm, O, alpha, scale)
        BackwardWithScale(hmm, O, beta, scale)
        Computexi(hmm, O, alpha, beta, xi)
        Computegamma(hmm, O, alpha, beta, gamma)
        iiter += 1

    logprobnew = logprobf

    return (iiter, logprobold, logprobnew)


def ReadHMM(filename):
    '''
    Read HMM from file (use the UMDHMM package's data format).
    '''

    hmm = HMM()

    with open(filename) as fhmm:
        M = int(fhmm.readline().split()[1])
        N = int(fhmm.readline().split()[1])
        fhmm.readline()
        A = [lmap(float, fhmm.readline().split()) for i in range(N)]
        fhmm.readline()
        B = [lmap(float, fhmm.readline().split()) for i in range(N)]
        fhmm.readline()
        pi = lmap(float, fhmm.readline().split())

        hmm.N = N
        hmm.M = M
        hmm.A = A
        hmm.B = B
        hmm.pi = pi

    return hmm

def ReadSeq(filename):
    '''
    Read observation sequence from file (use the UMDHMM package's data
    format).
    '''
    with open(filename) as fseq:
    	T = int(fseq.readline().split()[1])
    	O = lmap(int, fseq.readline().split())
    return (T, O)

def PrintHMM(hmm):
    '''
    Translate HMM into string, and can be writed to file(use the UMDHMM
    packages's data format).
    '''

    N = hmm.N
    M = hmm.M
    A = hmm.A
    B = hmm.B
    pi = hmm.pi

    strhmm = 'M= %d\nN= %d\n' % (M, N)
    strhmm += 'A:\n'
    for i in range(N):
        strhmm += ' '.join(lmap(str, A[i])) + '\n'
    strhmm += 'B:\n'
    for i in range(N):
        strhmm += ' '.join(lmap(str, B[i])) + '\n'
    strhmm += 'pi:\n'
    strhmm += ' '.join(lmap(str, pi)) + '\n'
    return strhmm

