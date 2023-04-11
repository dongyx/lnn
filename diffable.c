#include <string.h>
#include <math.h>

void ident(double *y, double *x, int n)
{
	memcpy(y, x, n * sizeof *y);
}

void dident(double *d, double *y, int n)
{
	while (n-- > 0)
		*d++ = 1;
}

void sigm(double *y, double *x, int n)
{
	while (n-- > 0)
		*y++ = 1 / (1 + exp(-*x++));
}

void dsigm(double *d, double *y, int n)
{
	for (; n-- > 0; y++)
		*d++ = *y * (1 - *y);
}

void htan(double *y, double *x, int n)
{
	double h;

	while (n-- > 0) {
		h = exp(2 * *x++);
		*y++ = (h-1)/(h+1);
	}
}

void dhtan(double *d, double *y, int n)
{
	for (; n-- > 0; y++)
		*d++ = 1 - (*y)*(*y);
}

void relu(double *y, double *x, int n)
{
	for (; n-- > 0; x++)
		*y++ = *x > 0 ? *x : 0;
}

void drelu(double *d, double *y, int n)
{
	while (n-- > 0)
		*d++ = *y++ > 0;
}

void smax(double *y, double *x, int n)
{
	double s;
	int i;

	for (s = i = 0; i < n; i++)
		s += (y[i] = exp(x[i]));
	while (n-- > 0)
		*y++ /= s;
}

void dsmax(double **d, double *y, int n)
{
	int i, j;

	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			if (i == j)
				d[i][j] = y[i]*(1-y[i]);
			else
				d[i][j] = -y[i]*y[j];
}

double quade(double *ov, double *tv, int n)
{
	double s, d;

	for (s = 0; n-- > 0; ov++, tv++) {
		d = *ov - *tv;
		s += d*d / 2;
	}
	return s;
}

void dquade(double *dv, double *ov, double *tv, int n)
{
	while (n-- > 0)
		*dv++ = *ov++ - *tv++;
}

double binxe(double *ov, double *tv, int n)
{
	double s;

	for (s = 0; n-- > 0; ov++, tv++)
		s -= *tv*log(*ov) + (1-*tv)*log(1-*ov);
	return s;
}

void dbinxe(double *dv, double *ov, double *tv, int n)
{
	for (; n-- > 0; ov++, tv++)
		*dv++ = (*ov-*tv) / (*ov*(1-*ov));
}

double xentr(double *ov, double *tv, int n)
{
	double s;

	while (n-- > 0)
		s -= *tv++ * log(*ov++);
	return s;
}

void dxentr(double *dv, double *ov, double *tv, int n)
{
	while (n-- > 0)
		*dv++ = -*tv++ / *ov++;
}
