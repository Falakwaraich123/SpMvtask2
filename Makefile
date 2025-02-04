TARGETS = gsl mkl
CC = gcc
ICC = icc
OFLAGS = -O0
WARNFLAGS = -Wall -Wextra
CFLAGS = $(OFLAGS) $(WARNFLAGS) -D_GSL_
IFLAGS = $(OFLAGS) $(WARNFLAGS) -D_MKL_ -I"${MKLROOT}/include" -diag-disable=10441
LDLIBS = -lm -lgsl -lgslcblas
ILDLIBS = -lm -qmkl

SRCS = timer.c my_dense.c my_csr.c my_coo.c my_csc.c spmv.c
IOBJS = $(SRCS:.c=.o)
OBJS = $(SRCS:.c=_g.o)

all: $(TARGETS)

gsl: $(OBJS) 	
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@n
mkl: $(IOBJS)
	$(ICC) $(IFLAGS) $^ $(ILDLIBS) -o $@

$(IOBJS): $(SRCS)
	$(ICC) $(IFLAGS) $^ -c

%_g.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

cleanall:
	$(RM) $(OBJS) $(IOBJS) $(TARGETS) *~

