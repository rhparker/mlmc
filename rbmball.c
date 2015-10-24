//
// rbmball.c
//

#include <complex.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "normal.h" // normal and uniform random distributions

// returns sum of squares
float sum_squares(float *x, int dim) {
	float s = 0.0;
	for (int i=0; i<dim; i++) {
		s += x[i] * x[i];
	}
	return s;
}

float step_size(int l, float *x, int dim) {
	 return pow(2, -(l + 4) );
}

float adaptive_step(int l, float *x, int dim) {
	float d;
	float coarse, fine, d_factor;
	d = 1 - sqrt( sum_squares(x, dim) );
	coarse = pow(0.5, l + 4);
	fine = pow(0.25, l + 4);
	d_factor = (d/4)*(d/4);
	return fmin( coarse, fmax(fine, d_factor) );

}

// reflect vector in unit ball
int reflect(float *x, int dim) {
	float sumsq;
	float R;
	float ratio;
	sumsq = sum_squares(x, dim);
	if (sumsq > 1) {
		R = sqrt(sumsq) - 1;
		ratio = (1 - R) / (1 + R);
		for (int i=0; i<dim; i++) {
			x[i] = x[i] * ratio;
		}
		return(1);
	}
	else {
		return(0);
	}
}

void zeros(float *x, int dim) {
	for (int d = 0; d < dim; d++) {
		x[d] = 0;
	}
}

void add_to(float *x, float *y, int dim) {
	for (int d = 0; d < dim; d++) {
		x[d] += y[d];
	}
}

void rand_norm(float *x, int dim, float variance, int *seed) {
	float std_dev = sqrt(variance);
	for (int d = 0; d < dim; d++) {
		x[d] = r4_normal_ab(0, std_dev, seed);
	}
}

int rbm_ball_single(int l, float end_time, float *x, int dim, int *seed) {
	float t = 0;
	int steps = 0;
	int reflections = 0;
	float rands[dim];
	float h;

	while (t < end_time) {
		// h = step_size(l, x, dim);
		h = adaptive_step(l, x, dim);
		if (t + h >= end_time) {
			h = end_time - t;
		}
		rand_norm(rands, dim, h, seed);
		add_to(x, rands, dim);
		reflections += reflect(x, dim);
		t += h;
		steps++;
	}
	return 0;
}

int rbm_ball_diff(int l, float end_time, float *xc, float *xf, int dim, int *seed) {
	float std_dev;
	float t_old;
	float t = 0;
	float tc = 0;
	float tf = 0;
	float hc, hf;
	float rands[dim];
	float uc[dim];
	float uf[dim];

	if (l == 0) {
		return -1;
	}
	zeros(uc, dim);
	zeros(uf, dim);

	while (t < end_time) {
		t_old = t;
		t = fmin(tc, tf);
		rand_norm(rands, dim, t - t_old, seed);
		add_to(uf, rands, dim);
		add_to(uc, rands, dim);

		if (t == tf) {
			add_to(xf, uf, dim);
			reflect(xf, dim);
			// hf = step_size(l, xf, dim);
			hf = fmin( adaptive_step(l, xf, dim), end_time - tf );
			tf += hf;
			zeros(uf, dim);
		}
		if (t == tc) {
			add_to(xc, uc, dim);
			reflect(xc, dim);
			// hc = step_size(l - 1, xc, dim);
			hc = fmin( adaptive_step(l - 1, xc, dim), end_time - tc);
			tc += hc;
			zeros(uc, dim);
		}
		// printf("t_o %f  t %f  t_c %f  t_f %f\n", t_old, t, tc, tf);
	}
	return 0;
}

void rbm_ball(int l, int N, double *sums) {

	int seed;
	seed = 123456789;
	// seed = time(NULL);

	int dim = 3;
	int block_size = 1000;

	float T = 1.0;
	int steps;
	float step_size;
	float Pf[20], Pc[20];
	float Df;
	float Dc;
	float dP;

	for (int k=0; k<6; k++) {
		sums[k] = 0.0;
	}

	for (int n = 0; n < N; n++) {
		// initialize
		for (int i = 0; i < dim; i++) {
			Pf[i] = 0;
			Pc[i] = 0;
		}
		if (l == 0) {
			rbm_ball_single(l, T, Pf, dim, &seed);
			Df = sum_squares(Pf, dim);
			Dc = 0;
		}
		else {
			// rbm_ball_single(l, T, Pc, dim, &seed);
			// rbm_ball_single(l, T, Pf, dim, &seed);
			rbm_ball_diff(l, T, Pc, Pf, dim, &seed); 
			Dc = sum_squares(Pc, dim);
			Df = sum_squares(Pf, dim);
			// printf("%f %f \n", Df, Dc);
		}
		dP = Df - Dc;

		// add to running total of sum of squares
		sums[0] += dP;
		sums[1] += dP*dP;
		sums[2] += dP*dP*dP;
		sums[3] += dP*dP*dP*dP;
		sums[4] += Df;
		sums[5] += Df*Df;
	}


}

int main() {
	double sums[6];
	int N = 100;
	for (int i = 0; i <= 5; i++) {
		rbm_ball(i, N, sums);
		printf("Level %d, %f %f\n", i, sums[0], sums[4]);
	}
}

