"""
% These are similar to the MLMC tests for the MCQMC06 paper
% using a Milstein discretisation with 2^l timesteps on level l
%
% The figures are slightly different due to
% -- change in MSE split
% -- change in cost calculation
% -- different random number generation
% -- switch to S_0=100
"""

import numpy as np
import math
from scipy.stats import norm
from mlmc_test import mlmc_test
import ctypes

#
# level 1 estimator, uses ctypes to call the c-version of estimator 
# (recommended for speed)
# 

_mcql = ctypes.CDLL('mcq.so')
# void mcqmc06_l(int l, int N, int option, double *sums) 
mcqmc06_l = _mcql.mcqmc06_l
mcqmc06_l.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double * 6) )

def mcqmc06_l_c(l, N, option):
    array_type = ctypes.c_double * 6
    sums = array_type()
    _mcql.mcqmc06_l(l, N, option, sums )
    return sums

#
# level 1 estimator in Python
# (works in principle, but will not use since very slow)
# 

def mcqmc06_l_python(l, N, option):
  sums = np.zeros(6)
  dWf = np.zeros(2)
  dIf = np.zeros(2)
  Lf = np.zeros(2)

  K   = 100.0
  T   = 1.0
  r   = 0.05
  sig = 0.2
  B   = 0.85*K

  # nf = 1<<l;
  nf = 2.0**l
  nc = int( nf/2 );

  hf = T / nf
  hc = np.float64(T) / nc

  for counter in range(0, N):
    X0 = K

    Xf = X0
    Xc = Xf

    Af  = 0.5*hf*Xf
    Ac  = 0.5*hc*Xc

    Mf  = Xf
    Mc  = Xc

    Bf  = 1.0
    Bc  = 1.0

    if l == 0:
      dWf[0] = math.sqrt(hf)*np.random.randn()
      Lf[0] = math.log(np.random.rand())
      dIf[0] = math.sqrt(hf/12.0)*hf*np.random.randn()

      Xf0 = Xf
      Xf  = Xf + r*Xf*hf + sig*Xf*dWf[0] + 0.5*sig*sig*Xf*(dWf[0]*dWf[0]-hf)
      vf  = sig*Xf0
      Af  = Af + 0.5*hf*Xf + vf*dIf[0]
      Mf  = min(Mf,0.5*(Xf0+Xf-math.sqrt((Xf-Xf0)*(Xf-Xf0)-2.0*hf*vf*vf*Lf[0])))
      Bf  = Bf*(1.0-math.exp(-2.0*max(0.0,(Xf0-B)*(Xf-B)/(hf*vf*vf))))

    else:
      for n in range(0, nc):
        dWf[0] = math.sqrt(hf)*np.random.randn()
        dWf[1] = math.sqrt(hf)*np.random.randn()
        Lf[0] = np.log(np.random.rand())
        Lf[1] = np.log(np.random.rand())
        dIf[0] = math.sqrt(hf/12.0)*hf*np.random.randn()
        dIf[1] = math.sqrt(hf/12.0)*hf*np.random.randn()

        for m in range(0, 2):
          Xf0 = Xf
          Xf  = Xf + r*Xf*hf + sig*Xf*dWf[m]+0.5*sig*sig*Xf*(dWf[m]*dWf[m]-hf)
          vf  = sig*Xf0
          Af  = Af + hf*Xf + vf*dIf[m]
          Mf  = min(Mf, 0.5*(Xf0+Xf-math.sqrt((Xf-Xf0)*(Xf-Xf0)-2.0*hf*vf*vf*Lf[m])))
          Bf  = Bf*(1.0-math.exp(-2.0*max(0.0,(Xf0-B)*(Xf-B)/(hf*vf*vf))))

        dWc = dWf[0] + dWf[1]
        ddW = dWf[0] - dWf[1]

        Xc0 = Xc
        Xc  = Xc + r*Xc*hc + sig*Xc*dWc + 0.5*sig*sig*Xc*(dWc*dWc-hc)

        vc  = sig*Xc0
        Ac  = Ac + hc*Xc + vc*(dIf[0]+dIf[1] + 0.25*hc*ddW)
        Xc1 = 0.5*(Xc0 + Xc + vc*ddW)
        Mc  = min(Mc, 0.5*(Xc0+Xc1-math.sqrt((Xc1-Xc0)*(Xc1-Xc0)-2.0*hf*vc*vc*Lf[0])))
        Mc  = min(Mc, 0.5*(Xc1+Xc -math.sqrt((Xc -Xc1)*(Xc -Xc1)-2.0*hf*vc*vc*Lf[1])))
        Bc  = Bc *(1.0-math.exp(-2.0*max(0.0,(Xc0-B)*(Xc1-B)/(hf*vc*vc))))
        Bc  = Bc *(1.0-math.exp(-2.0*max(0.0,(Xc1-B)*(Xc -B)/(hf*vc*vc))))
      
      Af = Af - 0.5*hf*Xf;
      Ac = Ac - 0.5*hc*Xc;

    if option==1:
      Pf  = max(0.0,Xf-K)
      Pc  = max(0.0,Xc-K)
  
    elif option==2:
      Pf  = max(0.0,Af-K)
      Pc  = max(0.0,Ac-K)
  
    elif option==3:
      Pf  = Xf - Mf
      Pc  = Xc - Mc
    
    elif option==4:
      Pf  = K*norm.cdf((Xf0+r*Xf0*hf-K)/(sig*Xf0*sqrt(hf)))
      if l==0:
        Pc = Pf
      else:
        Pc = K*norm.cdf((Xc0+r*Xc0*hc+sig*Xc0*dWf[0]-K)/(sig*Xc0*sqrt(hf)))
    
    elif option==5:
      Pf  = Bf*max(0.0,Xf-K)
      Pc  = Bc*max(0.0,Xc-K)

    dP  = math.exp(-r*T)*(Pf-Pc)
    Pf  = math.exp(-r*T)*Pf

    if l==0:
      dP = Pf

    sums[0] += dP
    sums[1] += dP*dP
    sums[2] += dP*dP*dP
    sums[3] += dP*dP*dP*dP
    sums[4] += Pf
    sums[5] += Pf*Pf
  return sums

# main program

M  = 2     # refinement cost factor
N0 = 200   # initial samples on each level
Lmin = 2   # minimum refinement level
Lmax = 10  # maximum refinement level

# runs test
#   option = option to choose
#   title = title of this option (for display purposes)
#   N = samples for comvergence tests
#   L = levels for convergence tests
#   Eps = values of epsilon to use for tests
def run_test(option, title, N, L, Eps):
    print "\n ---- option {}: {} ----\n".format(option, title)
    mlmc_test( lambda l, N: mcqmc06_l_c(l, N, option), M, N, L, N0, Eps, Lmin, Lmax, save="euro")


# option 1: European call
run_test(1, "European call", 20000, 8, [0.005, 0.01, 0.02, 0.05, 0.1] )

# # option 2: Asian call
# run_test(2, "Asian call", 20000, 8, [0.005, 0.01, 0.02, 0.05, 0.1] )

# # option 3: lookback call
# run_test(3, "lookback call", 20000, 10, [0.005, 0.01, 0.02, 0.05, 0.1] )

# # option 4: digital call
# run_test(4, "digital call", 200000, 8, [0.01, 0.02, 0.05, 0.1, 0.2] )

# # option 5: barrier call
# run_test(5, "barrier call", 200000, 8, [0.005, 0.01, 0.02, 0.05, 0.1] )



