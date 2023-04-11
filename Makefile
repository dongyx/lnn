.PHONY: test install clean

CC=cc
INSTALL=install
prefix=/usr/local
bindir=$(prefix)/bin

all: lnn

lnn: main.o utils.o matrix.o neunet.o diffable.o
	$(CC) -o $@ $^

main.o: main.c utils.h neunet.h diffable.h
	$(CC) -c -o $@ $<

utils.o: utils.c utils.h
	$(CC) -c -o $@ $<

matrix.o: matrix.c matrix.h
	$(CC) -c -o $@ $<

neunet.o: neunet.c neunet.h matrix.h utils.h
	$(CC) -c -o $@ $<

diffable.o: diffable.c diffable.h
	$(CC) -c -o $@ $<

test: lnn
	./runtest

install: lnn
	$(INSTALL) -d $(bindir)
	$(INSTALL) $< $(bindir)

clean:
	rm -rf lnn testenv *.o *.tmp
