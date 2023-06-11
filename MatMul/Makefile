# Define required macros here
SHELL = /bin/sh

CFLAGS = -Wall -O3 -mcpu=native -Wextra
CC = mpicc
INCLUDES = -I${OPENBLAS_INC}   -I ${CUDA_INC} 
LIBS = -lopenblas -fopenmp -lcublas -lcudart  -L${OPENBLAS_LIB}  -L${CUDA_LIB} 

all: matrix test dgemm cuda
matrix: auxiliary.o
	${CC} main.c auxiliary.o  ${CFLAGS} ${INCLUDES} -o ${@}.x  ${LIBS}

test: auxiliary.o
	${CC} main.c auxiliary.o  ${CFLAGS} ${INCLUDES} -o matrix_${@}.x  ${LIBS}  -DTEST

dgemm: auxiliary.o
	${CC} main.c auxiliary.o  ${CFLAGS} ${INCLUDES} -o matrix_${@}.x  ${LIBS}  -DDGEMM

cuda: auxiliary.o
	${CC} main.c auxiliary.o  ${CFLAGS} ${INCLUDES} -o matrix_${@}.x  ${LIBS}  -DCUBLAS

auxiliary.o:
	${CC} auxiliary.c -c  ${CFLAGS} -o auxiliary.o  -fopenmp
clean:
	-rm -f *.x
