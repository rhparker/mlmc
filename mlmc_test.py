#
# mlmc_test.py
#

"""
  mlmc_test(mlmc_l, M, N,L, N0,Eps,Lmin,Lmax, fp)

  multilevel Monte Carlo test routine

   mlmc_l(l,N,sums)     low-level routine

   inputs:  l = level
            N = number of paths

   output: sums[0] = sum(Pf-Pc)
           sums[1] = sum((Pf-Pc).^2)
           sums[2] = sum((Pf-Pc).^3)
           sums[3] = sum((Pf-Pc).^4)
           sums[4] = sum(Pf)
           sums[5] = sum(Pf.^2)

   M      = refinement cost factor

   N      = number of samples for convergence tests
   L      = number of levels for convergence tests

   N0     = initial number of samples
   Eps    = desired accuracy array
   Lmin   = minimum level of refinement
   Lmax   = maximum level of refinement
"""

import numpy as np
from mlmc import mlmc, regression
from time import time, strftime

# analogs of tic/toc in Matlab
_tstart_stack = []

def tic():
    _tstart_stack.append(time())

def toc(fmt="Elapsed: %s s"):
    return time() - _tstart_stack.pop()
    # print fmt % (time() - _tstart_stack.pop())

def mlmc_test(mlmc_l, M, N, L, N0, Eps, Lmin, Lmax, **kwargs):
    # first, convergence test

    print "**********************************************************"
    print "*** Convergence tests, kurtosis, telescoping sum check ***"
    print "**********************************************************"
    print " l   ave(Pf-Pc)    ave(Pf)   var(Pf-Pc)    var(Pf)    kurtosis      check "
    print "----------------------------------------------------------------------------"

    del1 = np.zeros(L+1)
    del2 = np.zeros(L+1)
    var1 = np.zeros(L+1)
    var2 = np.zeros(L+1)
    chk1 = np.zeros(L+1)
    kur1 = np.zeros(L+1)
    cost = []

    for l in xrange(0, L+1):
        # generate mlmc samples, use actualy computation time for cost
        tic()
        sums = np.array( mlmc_l( l, N ) ) / N
        cost.append( toc() )

        del1[l] = sums[0];
        del2[l] = sums[4];
        var1[l] = np.maximum(sums[1]-sums[0]*sums[0], 1e-10);
        var2[l] = np.maximum(sums[5]-sums[4]*sums[4], 1e-10);

        kur1[l]  = ( sums[3] - 4.0*sums[2]*sums[0] +
                6.0 * sums[1] * ( sums[0]**2 ) -
                3.0 * ( sums[0]**4 ) ) / ( var1[l]**2 )

        if l == 0:
            chk1[l] = 0.0
        else:
            chk1[l] = np.sqrt(N) * np.absolute( del1[l] + del2[l-1] - del2[l] ) / ( 3.0 *
                    ( np.sqrt(var1[l]) + np.sqrt(var2[l-1]) + np.sqrt(var2[l]) ) )
        print "{}   {:10.4f}  {:10.4f}  {:10.4f}  {:10.4f}  {:10.4f}  {:10.4f}  ".format(
                l, del1[l], del2[l], var1[l], var2[l], kur1[l], chk1[l] )

    # print out a warning if kurtosis or consistency check looks bad

    if kur1[L] > 100.0:
        print "WARNING: kurtosis on finest level = ", kur1[L]
        print " indicates MLMC correction dominated by a few rare paths;"
        print " for information on the connection to variance of sample variances"
        print" see http://mathworld.wolfram.com/SampleVarianceDistribution.html"

    if np.max(chk1) > 1.0:
        print "WARNING: maximum consistency error = ", np.max(chk1)
        print " indicates identity E[Pf-Pc] = E[Pf] - E[Pc] not satisfied"

    # use linear regression to estimate alpha, beta

    x = np.zeros(L+1)
    y = np.zeros(L+1)

    for l in xrange(1, L+1):
        x[l-1] = l;
        y[l-1] = -np.log2( np.absolute( del1[l]) )
    alpha, sumsq = regression(L, x, y)

    for l in xrange(1, L+1):
        x[l-1] = l;
        y[l-1] = -np.log2( np.absolute( var1[l]) )
    beta, sumsq = regression(L, x, y)

    gamma = np.log2( cost[-1] / cost[-2] )
    # gamma = np.log2(M)

    print "******************************************************"
    print "*** Linear regression estimates of MLMC parameters ***"
    print "******************************************************"
    print " alpha = {}  (exponent for MLMC weak convergence)".format(alpha)
    print " beta  = {}  (exponent for MLMC variance)".format(beta)
    print " gamma = {}  (exponent for MLMC cost)".format(gamma)

    # second, mlmc complexity tests

    print
    print "*****************************"
    print "*** MLMC complexity tests ***"
    print "*****************************"
    print 
    print "  eps    mlmc_cost   std_cost    savings    N_l"
    print "------------------------------------------------"
 
    cost_data = []
    level_data = []
    for ep in Eps:
        P, Nl = mlmc(Lmin, Lmax, N0, ep, mlmc_l, gamma, alpha_0=alpha, beta_0=beta)

        std_cost = 0.0
        mlmc_cost = 0.0
        theta = 0.25

        for l in xrange(0, Lmax+1):
            if Nl[l] > 0:
                mlmc_cost += (1.0 + 1.0/M) * Nl[l] * (M**l)
                std_cost += var2[min(l, L)] * (M**l)

        std_cost = std_cost / ( ( 1.0 -theta)*(ep**2) )
        print "{:4.4f}  {:.4e}  {:.4e}  {:8.2f}".format(
                ep, mlmc_cost, std_cost, std_cost/mlmc_cost), "  ", filter(None, Nl)

        cost_data.append( [ep, mlmc_cost, std_cost] )
        level_data.append( Nl )
    print

    # if we want to save the data
    if 'save' in kwargs:
        np.savez(kwargs['save'], del1=del1, del2=del2, var1=var1, var2=var2, 
            kur1=kur1, chk1=chk1, cost_data=np.array(cost_data), level_data=np.array(level_data) )
