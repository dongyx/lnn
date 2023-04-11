#define MAXLAYER 64

/* R^n -> R^n function (vector field) */
typedef void (*vfield)(double *y, double *x, int n);

/* Derivative of vfield */
typedef void (*dvfield)(void *y, double *x, int n);

struct layer {
	int n;
	double **wm;
	double *bv;
	double *iv;
	double *ov;
	double *di;
	double **dw;
	double *db;
	double **jact;
	vfield act;
	dvfield dact;
	int xact;
};

extern struct layer net[MAXLAYER];
extern int nlayer;
extern double (*lossf)(double *ov, double *tv, int n);
extern void (*dlossf)(double *dv, double *ov, double *tv, int n);

void addlayer(int n, vfield act, dvfield dact, int xact);
void netinit(int train);
void learn(double **iv, double **tv, int n, double lr, double l2);
void feedfwd(double *iv);
