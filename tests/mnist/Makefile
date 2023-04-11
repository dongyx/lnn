.PHONY: clean
all: data.nosync data.nosync/train.vec data.nosync/test.in data.nosync/test.out data.nosync/test.lab data.nosync/test.vec

data.nosync:
	mkdir -p $@

data.nosync/train.vec: data.nosync/train.in data.nosync/train.out
	paste $^ >$@

data.nosync/train.in: train.bimg.gz
	gzip -d <$< | ./bimg2vec >$@

data.nosync/train.out: train.blab.gz
	gzip -d <$< | ./blab2vec >$@

data.nosync/test.vec: data.nosync/test.in data.nosync/test.out
	paste $^ >$@

data.nosync/test.in: test.bimg.gz
	gzip -d <$< | ./bimg2vec >$@

data.nosync/test.out: test.blab.gz
	gzip -d <$< | ./blab2vec >$@

data.nosync/test.lab: data.nosync/test.out
	./vec2lab <$< >$@

clean:
	rm -rf data.nosync
