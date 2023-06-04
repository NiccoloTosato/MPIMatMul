#!/bin/bash

#SBATCH --partition=m100_usr_prod
#SBATCH --job-name=MM1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=200gb
#SBATCH --time=18:00:00
#SBATCH --account=tra23_units
#SBATCH --exclusive

echo "1 nodi"
module load autoload openblas/0.3.9--gnu--8.4.0
module load autoload spectrum_mpi/10.4.0--binary
echo  "my"
srun -n1 -N1 make clean
srun -n1 -N1 mpicc main.c  -Wall -O3 -mcpu=native -Wextra -I/cineca/prod/opt/libraries/openblas/0.3.9/gnu--8.4.0/include -L/cineca/prod/opt/libraries/openblas/0.3.9/gnu--8.4.0/lib -o matrix.x  -lopenblas
mpirun --bind-to core  ./matrix.x

echo "blas"
srun -n1 -N1 make clean 
srun -n1 -N1 mpicc main.c  -Wall -O3 -mcpu=native -Wextra -I/cineca/prod/opt/libraries/openblas/0.3.9/gnu--8.4.0/include -L/cineca/prod/opt/libraries/openblas/0.3.9/gnu--8.4.0/lib -o matrix.x  -lopenblas -DDGEMM
mpirun --bind-to core -x OMP_NUM_THREADS=1 ./matrix.x

