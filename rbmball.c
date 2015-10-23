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

// reflect vector in unit ball
void reflect(float *x, int dim) {
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
	}
}

void rbm_ball_single(float step_size, float steps, float *x, float dim, int *seed) {
	float std_dev;
	std_dev = sqrt(step_size);

	for (int i = 0; i < steps; i++) {
		for (int d = 0; d < dim; d++) {
			x[d] += r4_normal_ab(0, std_dev, seed);
		}
		reflect(x, dim);
	}
}

void rbm_ball_diff(float step_size, float steps, float *y, float *z, float dim, int *seed) {
	float std_dev;
	float randoms[20][2];
	std_dev = sqrt(step_size / 2);
	float large_step = sqrt(step_size);
	float small_step = sqrt( step_size / 2 );

	// for (int d = 0; d < dim; d++) {
	// 	// generate random normals, uses same randoms for each path
	// 	randoms[d][0] = r4_normal_ab(0, std_dev, seed);
	// }

	for (int i = 0; i < steps; i++) {
		// y takes one big step
		for (int d = 0; d < dim; d++) {
			// y[d] += ( randoms[d][0] + randoms[d][1] );
			y[d] += r4_normal_ab(0, large_step, seed);
		}
		reflect(y, dim);
		// z takes two small steps
		for (int j = 0; j < 2; j++) {
			for (int d = 0; d < dim; d++) {
				// z[d] += randoms[d][j];
				z[d] += r4_normal_ab(0, small_step, seed);
			}
			reflect(z, dim);
		}
	}
}


void rbm_ball(int l, int N, double *sums) {

	int seed;
	seed = 123456789;
	// seed = time(NULL);

	int dim = 3;

	float T   = 1.0;
	int initial_steps = 100;
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

		steps = initial_steps * pow(2, l);
		step_size = T / steps;

		if (l == 0) {
			rbm_ball_single(step_size, steps, Pf, dim, &seed);
			Df = sum_squares(Pf, dim);
			Dc = 0;
		}
		else {
			rbm_ball_diff(step_size, steps, Pc, Pf, dim, &seed);
			Df = sum_squares(Pf, dim);
			Dc = sum_squares(Pc, dim);
			// printf("%f %f \n", Df, Dc);
		}
		dP = Df - Dc;

		// add to running total of sum of squares
		sums[0] += dP;
		sums[1] += dP*dP;
		sums[2] += dP*dP*dP;
		sums[3] += dP*dP*dP*dP;
		sums[4] += Df;
		sums[5] += Df;
	}


}

int main() {
	double sums[6];
	int N = 1000;
	for (int i = 0; i <= 5; i++) {
		rbm_ball(i, N, sums);
		printf("Level %d, %f %f\n", i, sums[0], sums[4]);
	}
}

