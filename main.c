#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <limits.h>
#include <stdio.h>
#include <errno.h>
#include <time.h>
#include "utils.h"
#include "neunet.h"
#include "diffable.h"
#ifndef SIZE_MAX
#define SIZE_MAX ((size_t)-1)	/* C89 doesn't define SIZE_MAX */
#endif
#if SIZE_MAX < INT_MAX
#error Unexpected small SIZE_MAX
#endif
#if RAND_MAX < INT_MAX
#error Unexpected small RAND_MAX
#endif
#define HEADER "LNN Model 1"
#define MAXSPEC 256

char *progname = "lnn";
static char netspec[MAXSPEC];

static void run(int argc, char **argv);
static void train(int argc, char **argv);
static void test(int argc, char **argv);
static void help(FILE *fp, int es);
static void version(void);
static void creatmodel(int train);
static void loadmodel(char *fn, int train);
static void setnet(int train);
static void dumpout(FILE *fp);
static void dumpmodel(FILE *fp);
static int loadtrainv(FILE *fp, double ***ivp, double ***tvp);

int main(int argc, char **argv)
{
	progname = argv[0];
	if (argc <= 1)
		help(stderr, -1);
	argv++;
	argc--;
	if (!strcmp(*argv, "--help"))
		help(stdout, 0);
	else if (!strcmp(*argv, "--version"))
		version();
	else if (!strcmp(*argv, "run"))
		run(argc, argv);
	else if (!strcmp(*argv, "train"))
		train(argc, argv);
	else if (!strcmp(*argv, "test"))
		test(argc, argv);
	else
		help(stderr, -1);
	return 0;
}

static void run(int argc, char **argv)
{
	FILE *fp;
	double *iv, *ip;
	int i;

	if (getopt(argc, argv, "m:") != 'm')
		help(stderr, -1);
	argc -= optind;
	argv += optind;
	if (argc > 1)
		help(stderr, -1);
	loadmodel(optarg, 0);
	if (!(fp = argc ? fopen(*argv, "r") : stdin))
		syserr();
	CALLOC(iv, net->n);
	ip = iv;
	while (1 == fscanf(fp, "%lf", ip))
		if (++ip == iv + net->n) {
			feedfwd(ip=iv);
			dumpout(stdout);
		}
	if (!feof(fp) || ip != iv)
		err("Invalid input vectors\n");
}

static void train(int argc, char **argv)
{
	static char mfile[BUFSIZ];
	int ch, iters, bs, tot, i, j, create;
	double lr, l2, **iv, **tv;
	FILE *fp;

	srand(clock()^getpid());
	create = -1;
	l2 = -1;
	iters = 32;
	lr = 1;
	bs = -1;
	l2 = 0;
	while ((ch = getopt(argc, argv, "m:C:i:r:b:R:s:")) != -1)
		switch (ch) {
		case 'm':
			create = 0;
			strncpy(mfile, optarg, sizeof mfile);
			if (mfile[sizeof mfile - 1])
				err("File path too long: %s\n", optarg);
			break;
		case 'C':
			create = 1;
			strncpy(netspec, optarg, sizeof netspec);
			if (netspec[sizeof netspec - 1])
				err("Network spec too long: %s\n", optarg);
			break;
		case 'i':
			if ((iters = intparse(optarg, NULL)) < 0)
				err("Invalid iteration count\n");
			break;
		case 'r':
			if ((lr = lfparse(optarg)) < 0)
				err("Invalid learning rate\n");
			break;
		case 'b':
			if ((bs = intparse(optarg, NULL)) < 0)
				err("Invalid batch size\n");
			break;
		case 'R':
			if ((l2 = lfparse(optarg)) < 0)
				err("Invalid L2-regularization parameter\n");
			break;
		case 's':
			srand(intparse(optarg, NULL));
			break;
		default:
			help(stderr, -1);
		}
	switch (create) {
	case 0:
		loadmodel(mfile, 1);
		break;
	case 1:
		creatmodel(1);
		break;
	default:
		err("Expect -m or -C\n");
	}
	argc -= optind;
	argv += optind;
	if (argc > 1)
		help(stderr, -1);
	if (!(fp = argc ? fopen(*argv, "r") : stdin))
		syserr();
	if ((tot = loadtrainv(fp, &iv, &tv)) > 0) {
		if (bs < 0 || bs > tot)
			bs = tot;
		while (iters-- > 0) {
			for (i = 0; i < bs; i++) {
				j = i + rand()%(tot-i);
				SWAP(&iv[i], &iv[j]);
				SWAP(&tv[i], &tv[j]);
			}
			learn(iv, tv, bs, lr, l2);
		}
	}
	dumpmodel(stdout);
}

void test(int argc, char **argv)
{
	FILE *fp;
	double *iv, *tv, loss;
	int n, m, k, i;

	if (getopt(argc, argv, "m:") != 'm')
		help(stderr, -1);
	argc -= optind;
	argv += optind;
	if (argc > 1)
		help(stderr, -1);
	loadmodel(optarg, 0);
	if (!(fp = argc ? fopen(*argv, "r") : stdin))
		syserr();
	n = net[0].n;
	m = net[nlayer-1].n;
	CALLOC(iv, n);
	CALLOC(tv, m);
	loss = 0;
	k = 0;
	fscanf(fp, " ");
	while (!feof(fp)) {
		for (i = 0; i < n; i++)
			if (1 != fscanf(fp, "%lf", &iv[i]))
				goto efmt;
		for (i = 0; i < m; i++)
			if (1 != fscanf(fp, "%lf", &tv[i]))
				goto efmt;
		fscanf(fp, " ");
		if (k == INT_MAX)
			goto emax;
		feedfwd(iv);
		loss += lossf(net[nlayer-1].ov, tv, m);
		k++;
	}
	printf("%f\n", loss/k);
	return;
efmt:	err("Invalid test data\n");
emax:	err("Too many test data\n");
}

void help(FILE *fp, int es)
{
	fputs(
		"Usage:\n"
		"	lnn run [OPTIONS] [FILE]\n"
		"	lnn train [OPTIONS] [FILE]\n"
		"	lnn test [OPTIONS] [FILE]\n"
		"	lnn --help\n"
		"	lnn --version\n"
		"Options for Running:\n"
		"	-m FILE\n"
		"		Specify the model\n"
		"Options for Training:\n"
		"	-C STR\n"
		"		Create a new model with the specific structure\n"
		"	-m FILE\n"
		"		Specify an existed model\n"
		"	-R NUM\n"
		"		Set the parameter of the L2 regularization\n"
		"		(default: regularization disabled)\n"
		"	-i INT\n"
		"		Set the number of iterations\n"
		"		(default: 32)\n"
		"	-r NUM\n"
		"		Set the learning rate\n"
		"		(default: 1)\n"
		"	-b INT\n"
		"		Set the batch size\n"
		"		(default: mini-batch training disabled)\n"
		"	-s INT\n"
		"		Set the random seed\n"
		"Options for Testing:\n"
		"	-m FILE\n"
		"		Specify the model\n"
		"Other Options:\n"
		"	--help\n"
		"		Print this brief usage\n"
		"	--version\n"
		"		Print the version information\n"
		"Visit <https://github.com/dongyx/lnn> for more information.\n"
		,fp
	);
	exit(es);
}

void version(void)
{
	puts("LNN 0.0.2");
	puts("Copyright(c) 2023 DONG Yuxuan <https://www.dyx.name>");
	exit(0);
}

static void creatmodel(int train)
{
	setnet(train);
}

static void loadmodel(char *fn, int train)
{
	static char fh[MAXSPEC];	/* file header */
	FILE *fp;
	int i, j, k;

	if (!(fp = fopen(fn, "r")))
		syserr();
	if (
		!fgets(fh, sizeof fh, fp) ||
		!fgets(netspec, sizeof netspec, fp)
	)
		err("Invalid mode file: lack of meta data\n");
	if (fh[i=strlen(fh)-1] != '\n')
		err("Invalid mode file: header too long\n");
	fh[i] = 0;
	if (netspec[i=strlen(netspec)-1] != '\n')
		err("Invalid mode file: network spec too long\n");
	netspec[i] = 0;
	if (strcmp(fh, HEADER))
		err("Invalid mode file: unknown header\n");
	setnet(train);
	for (k = 1; k < nlayer; k++) {
		for (i = 0; i < net[k].n; i++)
			for (j = 0; j < net[k-1].n; j++)
				if (1 != fscanf(fp, "%lf", &net[k].wm[i][j]))
					goto efmt;
		for (i = 0; i < net[k].n; i++)
			if (1 != fscanf(fp, "%lf", &net[k].bv[i]))
				goto efmt;
	}
	return;
efmt:	err("Invalid model file\n");
}

static void setnet(int train)
{
	char *ns, *act;
	int n, dim, cons, units;

	ns = netspec;
	cons = units = 0;
	switch (*ns) {
	case 'q':
		lossf = quade;
		dlossf = dquade;
		break;
	case 'b':
		lossf = binxe;
		dlossf = dbinxe;
		break;
	case 'x':
		lossf = xentr;
		dlossf = dxentr;
		break;
	case 0:
		err("Invalid network spec\n");
	default:
		err("Unknown loss function: %c\n", *ns);
	}
	ns++;
	while (*ns) {
		dim = intparse(ns, &act);
		if (dim < 0)
			err("Invalid dimension: %d\n", dim);
		if (dim > INT_MAX - units)
			err("Too many units\n");
		units += dim;
		if (nlayer > 0) {
			if (
				dim > INT_MAX/net[nlayer-1].n ||
				net[nlayer-1].n*dim > INT_MAX-cons
			)
				err("Too many connections\n");
			cons += net[nlayer-1].n * dim;
		}
		switch (*act) {
		case 'i':
			addlayer(dim, ident, (dvfield)dident, 0);
			break;
		case 's':
			addlayer(dim, sigm, (dvfield)dsigm, 0);
			break;
		case 't':
			addlayer(dim, htan, (dvfield)dhtan, 0);
			break;
		case 'r':
			addlayer(dim, relu, (dvfield)drelu, 0);
			break;
		case 'm':
			addlayer(dim, smax, (dvfield)dsmax, 1);
			break;
		default:
			err("Unknown activation function: %c\n", *act);
		}
		ns = act + 1;
	}
	if (nlayer < 2)
		err("Too less layers\n");
	netinit(train);
}

static void dumpout(FILE *fp)
{
	struct layer *l;
	int i;

	l = &net[nlayer-1];
	fprintf(fp, "%f", l->ov[0]);
	for (i = 1; i < l->n; i++)
		fprintf(fp, " %f", l->ov[i]);
	fputc('\n', fp);
}

static void dumpmodel(FILE *fp)
{
	int i, j, k;

	fprintf(fp, "%s\n%s\n", HEADER, netspec);
	for (k = 1; k < nlayer; k++) {
		fputc('\n', fp);
		for (i = 0; i < net[k].n; i++) {
			fprintf(fp, "%f", net[k].wm[i][0]);
			for (j = 1; j < net[k-1].n; j++)
				fprintf(fp, " %f", net[k].wm[i][j]);
			fputc('\n', fp);
		}
		fputc('\n', fp);
		fprintf(fp, "%f", net[k].bv[0]);
		for (i = 1; i < net[k].n; i++)
			fprintf(fp, " %f", net[k].bv[i]);
		fputc('\n', fp);
	}
}

static int loadtrainv(FILE *fp, double ***ivp, double ***tvp)
{
	double buf, **iv, **tv;
	int cap, sz, n, m, i;

	n = net[0].n;
	m = net[nlayer-1].n;
	sz = 0, cap = 64;
	CALLOC(iv, cap);
	CALLOC(tv, cap);
	fscanf(fp, " ");
	while (!feof(fp)) {
		if (sz >= cap) {
			if (cap > INT_MAX/2 || (cap*=2) > SIZE_MAX/sizeof *iv)
				err("Too many training data\n");
			REALLOC(iv, cap);
			REALLOC(tv, cap);
		}
		CALLOC(iv[sz], n);
		CALLOC(tv[sz], m);
		for (i = 0; i < n; i++)
			if (1 != fscanf(fp, "%lf", &iv[sz][i]))
				err(
					"Invalid training data: "
					"expecting input number\n"
				);
		for (i = 0; i < m; i++)
			if (1 != fscanf(fp, "%lf", &tv[sz][i]))
				err(
					"Invalid training data: "
					"expecting target number\n"
				);
		fscanf(fp, " ");
		sz++;
	}
	*ivp = iv;
	*tvp = tv;
	return sz;
}
