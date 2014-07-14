/* 
   Source file that provides methods to
   * perform LU decomposition on a matrix
   * perform backward substitution
*/

#include <iostream>
#include <math.h>
#define NRANSI
#define TINY 1.0e-20;

#include "lu.h"

void ludcmp(float a[9][9], int n, int indx[], float *d)
{
	int i, imax, j, k;
	float big, dum, sum, temp;
	float *vv;
	
	vv = new float[n];
	*d = 1.0;
	
	for (i = 1 ; i <= n ; i++) {
		big = 0.0;
		for ( j = 1 ; j <= n ; j++)
			if ( (temp = fabs(a[i][j])) > big )
				big = temp;
		if (big == 0.0)
			std::cerr<<"Singular matrix in routine ludcmp";
		vv[i] = 1.0 / big;
	}
	
	for (j = 1 ; j <= n ; j++) {
		for (i = 1 ; i < j ; i++) {
			sum = a[i][j];
			for (k = 1 ; k < i ; k++)
				sum -= a[i][k] * a[k][j];
			a[i][j] = sum;
		}
		big = 0.0;
		for (i = j ; i <= n ; i++) {
			sum = a[i][j];
			for (k = 1 ; k < j ; k++)
				sum -= a[i][k] * a[k][j];
			a[i][j] = sum;
			if ( (dum = vv[i]*fabs(sum)) >= big ) {
				big = dum;
				imax = i;
			}
		}
		if (j != imax) {
			for (k = 1 ; k <= n ; k++) {
				dum = a[imax][k];
				a[imax][k] = a[j][k];
				a[j][k] = dum;
			}
			*d = -(*d);
			vv[imax] = vv[j];
		}
		indx[j] = imax;
		if (a[j][j] == 0.0)
			a[j][j] = TINY;
		if (j != n) {
			dum = 1.0 / a[j][j];
			for (i = j+1 ; i <= n ; i++)
				a[i][j] *= dum;
		}
	}
	
	delete vv;
}

void lubksb(float a[9][9], int n, int indx[], float b[])
{
	int i, ii=0, ip, j;
	float sum;

	for (i = 1 ; i<=n ; i++) {
		ip = indx[i];
		sum = b[ip];
		b[ip] = b[i];
		if (ii)
			for (j = ii ; j <= i-1 ; j++)
				sum -= a[i][j]*b[j];
		else if (sum)
			ii = i;
		b[i] = sum;
	}
	
	for (i = n ; i >= 1 ; i--) {
		sum = b[i];
		for (j = i+1 ; j <= n ; j++)
			sum -= a[i][j] * b[j];
		b[i] = sum / a[i][i];
	}
}

#undef NRANSI
#undef TINY
