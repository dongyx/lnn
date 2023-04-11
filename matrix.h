/* out += mtrx*in
 * in: n-d vector
 * out: m-d vector
 * mtrx: m*n matrix
 */
void mxv(double *out, double *in, double **mtrx, int n, int m);

/* out += traspose(mtrx)*in
 * in: n-d vector
 * out: m-d vector
 * mtrx: n*m matrix
 */
void mtxv(double *out, double *in, double **mtrx, int n, int m);

/* vector multiplies vector trasposed
 * out += transpose(u).v
 * u: m-d vector
 * v: n-d vector
 */
void vxvt(double **out, double *u, double *v, int m, int n);

/* number multiplies matrix
 * mtrx *= c
 * mtrx: m*n matrix
 */
void nxm(double **mtrx, double c, int m, int n);

/* matrix adds matrix
 * a += b
 * a,b: m*n matricex
 */
void mam(double **a, double **b, int m, int n);

/* entrywise product: dst *= src */
void entmul(double *dst, double *src, int n);

/* entrywise sum: dst += scr */
void entadd(double *dst, double *src, int n);

/* numerical product: dst *= c */
void nummul(double *dst, double c, int n);
