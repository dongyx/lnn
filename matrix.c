#include "matrix.h"

void mxv(double *out, double *in, double **mtrx, int n, int m)
{
	int i, j;

	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			out[i] += mtrx[i][j] * in[j];
}

void mtxv(double *out, double *in, double **mtrx, int n, int m)
{
	int i, j;

	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			out[i] += mtrx[j][i] * in[j];
}

void vxvt(double **out, double *u, double *v, int m, int n)
{
	int i, j;

	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			out[i][j] += u[i]*v[j];
}

void nxm(double **mtrx, double c, int m, int n)
{
	int i, j;

	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			mtrx[i][j] *= c;
}

void mam(double **a, double **b, int m, int n)
{
	int i, j;

	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			a[i][j] += b[i][j];
}

void entmul(double *dst, double *src, int n)
{
	while (n-- > 0)
		*dst++ *= *src++;
}

void entadd(double *dst, double *src, int n)
{
	while (n-- > 0)
		*dst++ += *src++;
}

void nummul(double *dst, double c, int n)
{
	while (n-- > 0)
		*dst++ *= c;
}
