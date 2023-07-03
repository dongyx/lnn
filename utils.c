#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <limits.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include "utils.h"

extern char *progname;
static void verr(char *fmt, va_list ap);

void err(char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	verr(fmt, ap);
}

void syserr(void)
{
	perror(progname);
	exit(-1);
}

int intparse(char *s, char **next)
{
	long buf;
	char *e;

	errno = 0;
	buf = strtol(s, &e, 10);
	if (errno)
		syserr();
	if (e == s || !next && *e)
		err("Invalid integer\n");
	if (next)
		*next = e;
#if LONG_MAX > INT_MAX
	if (buf < INT_MIN || buf > INT_MAX)
		err("Integer out of range: %ld\n", buf);
#endif
	return buf;
}

double lfparse(char *s)
{
	double buf;
	char *e;

	errno = 0;
	buf = strtod(s, &e);
	if (errno)
		syserr();
	if (e == s || *e || !isfinite(buf))
		err("Invalid number: %s\n", s);
	return buf;
}

void *alloc2(int m, int n, int w)
{
	char **p;
	int i;

	CALLOC(p, m);
	if (!(p[0] = calloc(m*n, w)))
		syserr();
	for (i = 1; i < m; i++)
		p[i] = p[i-1] + w*n;
	return p;
}

static void verr(char *fmt, va_list ap)
{
	fprintf(stderr, "%s: ", progname);
	vfprintf(stderr, fmt, ap);
	exit(-1);
}
