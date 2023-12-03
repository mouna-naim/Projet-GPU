PROG = stencil
CFLAGS = -Wall -g -mavx  -O3 
LDLIBS = -lm 

.phony: all clean

all: $(PROG)

clean:
	rm -fv $(PROG)
