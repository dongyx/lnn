.PHONY: test install clean

CC=cc
LIBS=-lm
INSTALL=install
prefix=/usr/local
bindir=$(prefix)/bin

all: lnn

lnn: main.o utils.o matrix.o neunet.o diffable.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

main.o: main.c utils.h neunet.h diffable.h
	$(CC) $(CFLAGS) -c -o $@ $<

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c -o $@ $<

matrix.o: matrix.c matrix.h
	$(CC) $(CFLAGS) -c -o $@ $<

neunet.o: neunet.c neunet.h matrix.h utils.h
	$(CC) $(CFLAGS) -c -o $@ $<

diffable.o: diffable.c diffable.h
	$(CC) $(CFLAGS) -c -o $@ $<

test: lnn
	./runtest

install: lnn
	$(INSTALL) -d $(bindir)
	$(INSTALL) $< $(bindir)

clean:
	rm -rf lnn *.o *.tmp
