/*
% These are similar to the MLMC tests for the MCQMC06 paper
% using a Milstein discretisation with 2^l timesteps on level l
%
% The figures are slightly different due to
% -- change in MSE split
% -- change in cost calculation
% -- different random number generation
% -- switch to S_0=100
*/

# include <complex.h>
# include <stdlib.h>
# include <stdio.h>
# include <time.h>
# include <math.h>
#include "normal.h" // normal and uniform random distributions

// normal CDF function, uses erfc from math.h
double normalCDF(double value) {
   return 0.5 * erfc(-value * M_SQRT1_2);
}


/*-------------------------------------------------------
%
% level l estimator
%
*/


void mcqmc06_l(int l, int N, int option, double *sums) {

  int   M, nf, nc;
  float T, r, sig, B, hf, hc, X0, Xf, Xc, Af, Ac, Mf, Mc, Bf, Bc,
        Xf0, Xc0, Xc1, vf, vc, dWc, ddW, Pf, Pc, dP, K;

  float dWf[2], dIf[2], Lf[2];

  // initialise seeds

  int seed;
  // seed = 123456789;
  seed = time(NULL);

  K   = 100.0f;
  T   = 1.0f;
  r   = 0.05f;
  sig = 0.2f;
  B   = 0.85f*K;

  nf = 1<<l;
  nc = nf/2;

  hf = T / ((float) nf);
  hc = T / ((float) nc);

  for (int k=0; k<6; k++) sums[k] = 0.0;

  for (int np = 0; np<N; np++) {
    X0 = K;

    Xf = X0;
    Xc = Xf;

    Af  = 0.5f*hf*Xf;
    Ac  = 0.5f*hc*Xc;

    Mf  = Xf;
    Mc  = Xc;

    Bf  = 1.0f;
    Bc  = 1.0f;

    if (l==0) {
      dWf[0] = sqrt(hf)*r4_normal_01(&seed);
      Lf[0] = logf( r4_uniform_01(&seed) );
      dIf[0] = sqrt(hf/12.0f)*hf*r4_normal_01(&seed);

      Xf0 = Xf;
      Xf  = Xf + r*Xf*hf + sig*Xf*dWf[0]
               + 0.5f*sig*sig*Xf*(dWf[0]*dWf[0]-hf);
      vf  = sig*Xf0;
      Af  = Af + 0.5f*hf*Xf + vf*dIf[0];
      Mf  = fminf(Mf,
            0.5f*(Xf0+Xf-sqrtf((Xf-Xf0)*(Xf-Xf0)-2.0f*hf*vf*vf*Lf[0])));
      Bf  = Bf*(1.0f-expf(-2.0f*fmaxf(0.0f,(Xf0-B)*(Xf-B)/(hf*vf*vf))));
    }

    else {
      for (int n=0; n<nc; n++) {
        dWf[0] = sqrt(hf)*r4_normal_01(&seed);
        dWf[1] = sqrt(hf)*r4_normal_01(&seed);

        Lf[0] = logf( r4_uniform_01(&seed) );
        Lf[1] = logf( r4_uniform_01(&seed) );

        dIf[0] = sqrt(hf/12.0f)*hf*r4_normal_01(&seed);
        dIf[1] = sqrt(hf/12.0f)*hf*r4_normal_01(&seed);

        for (int m=0; m<2; m++) {
          Xf0 = Xf;
          Xf  = Xf + r*Xf*hf + sig*Xf*dWf[m]
                   + 0.5f*sig*sig*Xf*(dWf[m]*dWf[m]-hf);
          vf  = sig*Xf0;
          Af  = Af + hf*Xf + vf*dIf[m];
          Mf  = fminf(Mf,
                0.5f*(Xf0+Xf-sqrtf((Xf-Xf0)*(Xf-Xf0)-2.0f*hf*vf*vf*Lf[m])));
          Bf  = Bf*(1.0f-expf(-2.0f*fmaxf(0.0f,(Xf0-B)*(Xf-B)/(hf*vf*vf))));
        }

        dWc = dWf[0] + dWf[1];
        ddW = dWf[0] - dWf[1];

        Xc0 = Xc;
        Xc  = Xc + r*Xc*hc + sig*Xc*dWc + 0.5f*sig*sig*Xc*(dWc*dWc-hc);

        vc  = sig*Xc0;
        Ac  = Ac + hc*Xc + vc*(dIf[0]+dIf[1] + 0.25f*hc*ddW);
        Xc1 = 0.5f*(Xc0 + Xc + vc*ddW);
        Mc  = fminf(Mc,
            0.5f*(Xc0+Xc1-sqrtf((Xc1-Xc0)*(Xc1-Xc0)-2.0f*hf*vc*vc*Lf[0])));
        Mc  = fminf(Mc,
            0.5f*(Xc1+Xc -sqrtf((Xc -Xc1)*(Xc -Xc1)-2.0f*hf*vc*vc*Lf[1])));
        Bc  = Bc *(1.0f-expf(-2.0f*fmaxf(0.0f,(Xc0-B)*(Xc1-B)/(hf*vc*vc))));
        Bc  = Bc *(1.0f-expf(-2.0f*fmaxf(0.0f,(Xc1-B)*(Xc -B)/(hf*vc*vc))));
      }
      Af = Af - 0.5f*hf*Xf;
      Ac = Ac - 0.5f*hc*Xc;
    }

    if (option==1) {
      Pf  = fmaxf(0.0f,Xf-K);
      Pc  = fmaxf(0.0f,Xc-K);
    }
    else if (option==2) {
      Pf  = fmaxf(0.0f,Af-K);
      Pc  = fmaxf(0.0f,Ac-K);
    }
    else if (option==3) {
      Pf  = Xf - Mf;
      Pc  = Xc - Mc;
    }
    else if (option==4) {
      Pf  = K * normalCDF((Xf0+r*Xf0*hf-K)/(sig*Xf0*sqrt(hf)));
      if (l==0)
        Pc = Pf;
      else
        Pc = K * normalCDF((Xc0+r*Xc0*hc+sig*Xc0*dWf[0]-K)/(sig*Xc0*sqrt(hf)));
    }
    else if (option==5) {
      Pf  = Bf*fmaxf(0.0f,Xf-K);
      Pc  = Bc*fmaxf(0.0f,Xc-K);
    }

    dP  = exp(-r*T)*(Pf-Pc);
    Pf  = exp(-r*T)*Pf;

    if (l==0) dP = Pf;

    sums[0] += dP;
    sums[1] += dP*dP;
    sums[2] += dP*dP*dP;
    sums[3] += dP*dP*dP*dP;
    sums[4] += Pf;
    sums[5] += Pf*Pf;
  }

}

