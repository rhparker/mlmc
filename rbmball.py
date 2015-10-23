

import numpy as np
import math
from scipy.stats import norm
from mlmc_test import mlmc_test
import ctypes

#
# level 1 estimator, uses ctypes to call the c-version of estimator 
# (recommended for speed)
# 

_rbmball = ctypes.CDLL('rbmball.so')
# void mcqmc06_l(int l, int N, int option, double *sums) 
rbm_ball = _rbmball.rbm_ball
# rbm_ball.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double * 6) )
rbm_ball.argtypes = (ctypes.c_int, ctypes.c_int, ndpointer(np.float, flags="C_CONTIGUOUS") )


def rbm_ball(l, N):
    # array_type = ctypes.c_double * 6
    # sums = array_type()
    sums = np.zeros(6)
    _rbmball.rbm_ball(l, N, sums )
    return sums  


# main program

N = 10000   # samples for each level
M  = 2      # refinement cost factor
N0 = 100    # initial samples on each level
Lmin = 2    # minimum refinement level
Lmax = 5    # maximum refinement level
L = 5
Eps = [0.0002, 0.0005, 0.001, 0.002, 0.005]


mlmc_test( rbm_ball, M, N, L, N0, Eps, Lmin, Lmax, save="rbmball")



