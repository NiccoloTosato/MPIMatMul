# Define required macros here
SHELL = /bin/sh

CFLAG = -Wall -03 -march=native -Wextra 
CC = mpicc
INCLUDES = -I/usr/include/openblas/
LIBS = -lopenblas

matrix.x:
	${CC} main.c ${CFLAGS} ${INCLUDES} -o $@  ${LIBS}

clean:
	-rm -f main.x
