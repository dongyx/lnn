/* differentiable functions and their derivatives */

extern void ident(double *y, double *x, int n);
extern void dident(double *d, double *y, int n);
extern void sigm(double *y, double *x, int n);
extern void dsigm(double *d, double *y, int n);
extern void htan(double *y, double *x, int n);
extern void dhtan(double *d, double *y, int n);
extern void relu(double *y, double *x, int n);
extern void drelu(double *d, double *y, int n);
extern void smax(double *y, double *x, int n);
extern void dsmax(double **d, double *y, int n);

extern double quade(double *ov, double *tv, int n);
extern void dquade(double *dv, double *ov, double *tv, int n);
extern double binxe(double *ov, double *tv, int n);
extern void dbinxe(double *dv, double *ov, double *tv, int n);
extern double xentr(double *ov, double *tv, int n);
extern void dxentr(double *dv, double *ov, double *tv, int n);
