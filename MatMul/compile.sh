OPENBLAS_LIB=/u/area/ntosato/lecture2/MPIMatMul/cublas_new/oblas/lib/
OPENBLAS_INC=/u/area/ntosato/lecture2/MPIMatMul/cublas_new/oblas/include/
gcc auxiliary.c -o auxiliary.o -fopenmp -c -O3 -Wall -Wextra

mpicc main.c auxiliary.o -O3 -Wall -Wextra -o matrix.x
mpicc main.c auxiliary.o -O3 -Wall -Wextra -o matrix_dgemm.x  -lopenblas -L$OPENBLAS_LIB -I$OPENBLAS_INC -DDGEMM
mpicc main.c auxiliary.o -O3 -Wall -Wextra -o matrix_test.x  -lopenblas -L$OPENBLAS_LIB -I$OPENBLAS_INC -DTEST 
mpicc main.c auxiliary.o -O3 -Wall -Wextra -o matrix_debug.x  -lopenblas -L$OPENBLAS_LIB -I$OPENBLAS_INC -DDEBUG
mpicc main.c auxiliary.o -O3 -Wall -Wextra   -lopenblas  -lcublas -lcudart  -L$OPENBLAS_LIB -I$OPENBLAS_INC  -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/cuda/12.0/include -DCUBLAS
