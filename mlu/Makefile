CNCC=cncc
OPTIONS=--bang-mlu-arch=mtp_372 --bang-mlu-arch=mtp_592 

TARGETS=$(patsubst %.mlu, %.so, $(wildcard *.mlu))

all: $(TARGETS)

%.so: %.mlu
	$(CNCC) $^ $(OPTIONS) -shared -O3 -Wall -Werror -fPIC -std=c++11 -pthread -o $@ 

.PHONY : clean, copy
clean:
	rm $(TARGETS)

