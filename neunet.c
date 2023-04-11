#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.h"
#include "matrix.h"
#include "neunet.h"

struct layer net[MAXLAYER];
int nlayer;
double (*lossf)(double *ov, double *tv, int n);
void (*dlossf)(double *dv, double *ov, double *tv, int n);

static double *vbuf;	/* temporary vector buffer */

static void propback(double *tv);
static void setdi(struct layer *l, double *dout);
static void update(double lr, double l2, int n);
static double randlf(void);
static void check(struct layer *l);

void addlayer(int n, vfield act, dvfield dact, int xact)
{
	struct layer *l;

	l = &net[nlayer++];
	l->n = n;
	l->act = act;
	l->dact = dact;
	l->xact = xact;
}

void netinit(int train)
{
	int i, j, k, max;

	max = 0;
	for (k = 0; k < nlayer; k++) {
		if (net[k].n > max)
			max = net[k].n;
		CALLOC(net[k].ov, net[k].n);
		if (k == 0)
			continue;
		ALLOC2(net[k].wm, net[k].n, net[k-1].n);
		CALLOC(net[k].iv, net[k].n);
		CALLOC(net[k].bv, net[k].n);
		if (train) {
			CALLOC(net[k].di, net[k].n);
			CALLOC(net[k].db, net[k].n);
			ALLOC2(net[k].dw, net[k].n, net[k-1].n);
			if (net[k].xact)
				ALLOC2(net[k].jact, net[k].n, net[k].n);
		}
	}
	if (!train)
		return;
	CALLOC(vbuf, max);
	for (k = 1; k < nlayer; k++)
		for (i = 0; i < net[k].n; i++) {
			for (j = 0; j < net[k-1].n; j++)
				net[k].wm[i][j] = randlf();
			net[k].bv[i] = randlf();
		}
}

void feedfwd(double *iv)
{
	int i;

	net->act(net->ov, iv, net->n);
	for (i = 1; i < nlayer; i++) {
		MEMCPY(net[i].iv, net[i].bv, net[i].n);
		mxv(net[i].iv, net[i-1].ov, net[i].wm, net[i-1].n, net[i].n);
		net[i].act(net[i].ov, net[i].iv, net[i].n);
	}
}

void learn(double **iv, double **tv, int n, double lr, double l2)
{
	int i;

	for (i = 1; i < nlayer; i++) {
		MEMSET(net[i].dw[0], 0, net[i-1].n*net[i].n);
		MEMSET(net[i].db, 0, net[i].n);
	}
	for (i = 0; i < n; i++) {
		feedfwd(iv[i]);
		propback(tv[i]);
	}
	update(lr, l2, n);
}

static void propback(double *tv)
{
	struct layer *ol;
	int i, j;

	ol = &net[nlayer-1];
	dlossf(vbuf, ol->ov, tv, ol->n);
	setdi(ol, vbuf);
	for (i = nlayer-2; i > 0; i--) {
		MEMSET(vbuf, 0, net[i].n);
		mtxv(vbuf, net[i+1].di, net[i+1].wm, net[i+1].n, net[i].n);
		setdi(&net[i], vbuf);
	}
	for (i = 1; i < nlayer; i++) {
		vxvt(net[i].dw, net[i].di, net[i-1].ov, net[i].n, net[i-1].n);
		entadd(net[i].db, net[i].di, net[i].n);
	}
}

static void setdi(struct layer *l, double *dout)
{
	if (l->xact) {
		MEMSET(l->di, 0, l->n);
		l->dact(l->jact, l->ov, l->n);
		mtxv(l->di, dout, l->jact, l->n, l->n);
	} else {
		l->dact(l->di, l->ov, l->n);
		entmul(l->di, dout, l->n);
	}
}

static void update(double lr, double l2, int n)
{
	int i;

	for (i = 1; i < nlayer; i++) {
		nxm(net[i].wm, 1-lr*l2, net[i].n, net[i-1].n);
		nxm(net[i].dw, -lr/n, net[i].n, net[i-1].n);
		mam(net[i].wm, net[i].dw, net[i].n, net[i-1].n);
		nummul(net[i].db, -lr/n, net[i].n);
		entadd(net[i].bv, net[i].db, net[i].n);
		check(&net[i]);
	}
}

static double randlf(void)
{
	return (double)rand()/RAND_MAX*2-1;
}

static void check(struct layer *l)
{
	struct layer *p;
	int i, j;

	p = l - 1;
	for (i = 0; i < l->n; i++)
		for (j = 0; j < p->n; j++)
			if (!isfinite(l->wm[i][j]))
				err("Weight float overflow\n");
	for (i = 0; i < l->n; i++)
		if (!isfinite(l->bv[i]))
			err("Bias float overflow\n");
}
