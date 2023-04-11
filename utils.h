/* deps: stdlib.h, string.h */

#define CALLOC(x, n) do { \
	if (!((x) = calloc((n), sizeof *(x)))) \
		syserr(); \
} while(0)

#define REALLOC(x, n) do { \
	if (!((x) = realloc((x), (n) * sizeof *(x)))) \
		syserr(); \
} while(0)

#define ALLOC2(x, m, n) ((x) = alloc2((m), (n), sizeof **(x)))

#define MEMSET(x, v, n) memset((x), (v), (n)*sizeof *(x))

#define MEMCPY(d, s, n) memcpy((d), (s), (n)*sizeof *(d))

#define SWAP(x, y) do { \
	char _SWAP_MACRO_BUF[sizeof *(x)]; \
	memcpy(_SWAP_MACRO_BUF, (x), sizeof (*x)); \
	memcpy((x), (y), sizeof (*x)); \
	memcpy((y), _SWAP_MACRO_BUF, sizeof (*x)); \
} while(0)

void err(char *fmt, ...);
void syserr(void);
int intparse(char *s, char **next);
double lfparse(char *s);
void *alloc2(int m, int n, int w);
