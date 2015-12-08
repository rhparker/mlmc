#
# mlmc.py
#

import math
import numpy as np

"""
P = mlmc(Lmin, Lmax, N0, eps, mlmc_l, alpha, beta, gamma, Nl)

multilevel Monte Carlo control routine

Lmin  = minimum level of refinement       >= 2
Lmax  = maximum level of refinement       >= Lmin
N0    = initial number of samples         > 0
eps   = desired accuracy (rms error)      > 0 

alpha -> weak error is  O(2^{-alpha*l})
beta  -> variance is    O(2^{-beta*l})
gamma -> sample cost is O(2^{gamma*l})    > 0

if alpha, beta are not positive then they will be estimated

mlmc_l( l,N )   low-level function
     l       = level
     N       = number of paths
  mlmc_l returns array sums:
     sums[0] = sum(Y)
     sums[1] = sum(Y.^2)
  where Y are iid samples with expected value:
     E[P_0]           on level 0
     E[P_l - P_{l-1}] on level l>0

mlmc returns:
    P     = value
    Nl    = number of samples at each level
"""

def mlmc(Lmin, Lmax, N0, eps, mlmc_l, gamma, **kwargs):

    # check input parameters
    assert Lmin >= 2, "error: need Lmin >= 2"
    assert Lmax >= Lmin, "error: need Lmax >= Lmin"
    assert N0 > 0, "error: need N0 > 0"
    assert eps > 0, "error: need eps > 0"
    assert gamma > 0, "error: need gamma > 0"   
    
    # initialization, use zero if alpha_0 and/or beta_0 not specified
    # if they are not specified, they will be estimated later
    if "alpha_0" in kwargs:
        alpha = max(0.0, kwargs["alpha_0"])
    else:
        alpha = 0 
    if "beta_0" in kwargs:
        beta = max(0.0, kwargs["beta_0"])
    else:
        beta = 0  
    
    theta = 0.25                              # MSE split between bias^2 and variance   
    L = Lmin;                                 # current level we are up to, starts at Lmin
    converged = False;  

    # initialize arrays
    Nl = [0 for i in range(Lmax + 1)] 
    Cl = np.array( [ 2.0 ** (l * gamma) for l in xrange(Lmax + 1) ] )
    suml = np.zeros( (4, Lmax + 1) )           # stores sums on various levels
    dNl = [0 for i in range(Lmax + 1)]         # number of additional samples to generate on level l
    ml = np.zeros(Lmax + 1)                    # absolute average
    Vl = np.zeros(Lmax + 1)                    # variance
    x = np.zeros(Lmax + 1)
    y = np.zeros(Lmax + 1)

    # initialize dNl (how many new samples we need on each level)
    for l in range(Lmin + 1):
        dNl[l] = N0

    # main loop

    while converged == False:
        # update sample sums
        # iterate over all the levels 
        for l in range(L+1):
            # if we need to generate additional samples on level l
            if dNl[l] > 0:
                # low-level function returns sum and sum of squares for 
                # dNl[l] samples on level l
                sums = mlmc_l( l, dNl[l] )
                suml[0][l] += dNl[l]            # number of samples on level l
                suml[1][l] += sums[0]           # sum of samples on level l
                suml[2][l] += sums[1]           # sum of squares of samples on level l

        # compute absolute average and variance, correct for possible under-sampling,
        # and set optimal number of new samples

        csum = 0.0                               # cumulative cost

        for l in range(L+1):
            # computes absolute average and variance
            # for number of samples on level 1, take max with 1 so sure we are not dividing by zero
            num_l = max(suml[0][l], 1)
            ml[l] = abs( suml[1][l] / num_l )
            Vl[l] = max( 0.0, suml[2][l]/num_l - ml[l]**2 )
            # correct for possible under-sampling in situations where there are few samples
            if l > 1:
                ml[l] = max( ml[l], 0.5 * ml[l-1] / ( 2.0 ** alpha) )
                Vl[l] = max( Vl[l], 0.5 * Vl[l-1] / ( 2.0 ** beta) )
            # cumulative sum of costs at each level
            csum += math.sqrt( Vl[l] * Cl[l] )

        # set optimal number of additional samples needed at each level
        # we make our estimate then subtract how many samples we currently
        # have at each level
        # for l in range(L+1):
        #     Ns = np.ceil( math.sqrt( Vl[l]/Cl[l] ) * csum / ( (1.0-theta)*(eps**2) ) )
        #     dNl[l] = int( max( 0, Ns - suml[0][l] ) )

        for l in range(L+1):
            Ns = max(0, math.sqrt( Vl[l]/Cl[l] ) * csum / ( (1.0-theta)*(eps**2) ) - suml[0][l] )
            dNl[l] = int( np.ceil( Ns ) )

        # use linear regression to estimate alpha, beta if not given

        if ("alpha_0" not in kwargs):
            for l in range(1, L + 1):
                x[l-1] = l;
                y[l-1] = -np.log2( ml[l] )
            alpha, sumsq = regression(L, x, y)

        if ("beta_0" not in kwargs):
            for l in range(1, L + 1):
                x[l-1] = l
                y[l-1] = -np.log2( Vl[l] )
            beta, sumsq = regression(L, x, y)

        # if (almost) converged, estimate remaining error and decide 
        # whether a new level is required

        csum = 0.0
        # checks to see if we need more than 1% more samples than we currently have
        for l in range(0, L+1):
            csum += max( 0.0, dNl[l] - 0.01*suml[0][l] )

        # if we don't, then either we are done, or we need a new level
        if csum == 0:
            converged = True;
            # check to see if we've actually converged
            rem = ml[L] / ( (2.0**alpha) - 1.0 )

            # if we haven't then add a new level, if allowed
            if rem > math.sqrt(theta)*eps:
                # if we are at the maximum level
                if L == Lmax:
                    print("*** failed to achieve weak convergence ***")
                else:
                    converged = False;
                    # otherwise go up to the next level and recompute how many new
                    # samples we need for each level
                    L += 1
                    csum = 0.0;
                    for l in range(0, L+1):
                        csum += math.sqrt( Vl[l]*Cl[l] )
                    for l in range(0, L+1):
                        Ns = max(0, math.sqrt( Vl[l]/Cl[l] ) * csum / ( (1.0-theta)*(eps**2) ) - suml[0][l] )
                        dNl[l] = int( np.ceil( Ns ) )

    # finally, evaluate multilevel estimator

    P = 0.0;
    for l in range(0, L+1):
        P += suml[1][l]/suml[0][l]
        Nl[l] = int(suml[0][l])

    return P, Nl

# regression function
def regression(N, x, y):
    csum = np.zeros(3)
    csum_y = np.zeros(2)

    for i in xrange(1, N):
        csum[0] += 1.0
        csum[1] += x[i]
        csum[2] += x[i]**2
        csum_y[0] += y[i]
        csum_y[1] += y[i]*x[i]

    a = (csum[0]*csum_y[1] - csum[1]*csum_y[0]) / (csum[0]*csum[2] - csum[1]*csum[1])
    b = (csum[2]*csum_y[0] - csum[1]*csum_y[1]) / (csum[0]*csum[2] - csum[1]*csum[1])
    return a, b
