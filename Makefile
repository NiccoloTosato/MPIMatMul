# Define required macros here
SHELL = /bin/sh

CFLAG = -Wall -03 -march=native -Wextra  -DDGEMM -DDEBUG
CC = mpicc
INCLUDES = -I/usr/include/openblas/
LIBS = -lopenblas -fopenmp

all: matrix debug dgemm
matrix:
	${CC} main.c  ${CFLAGS} ${INCLUDES} -o ${@}.x  ${LIBS}

debug:
	${CC} main.c  ${CFLAGS} ${INCLUDES} -o matrix_${@}.x  ${LIBS}  -DTEST

dgemm:
	${CC} main.c  ${CFLAGS} ${INCLUDES} -o matrix_${@}.x  ${LIBS}  -DDGEMM

clean:
	-rm -f *.x
