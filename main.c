#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#define N 15

void print_matrix(double* A,int n_col) {
  //debug purpose: print a matrix
  for(int i=0;i<n_col;++i){
    for(int j=0;j<N;++j)
      printf("%3.3g ",A[j+i*N]);
    printf("\n");
  }
}

void print_matrix_transpose(double* A,int n_col) {
  for(int i=0;i<N;++i){
    for(int j=0;j<n_col;++j)
      printf("%3.3g ",A[j+i*n_col]);
    printf("\n");
  }
}

void print_matrix_square(double* A,int n_col) {
  for(int i=0;i<n_col;++i){
    for(int j=0;j<n_col;++j)
      printf("%3.3g ",A[j+i*n_col]);
    printf("\n");
  }
}

void mat_mul(double* A, double* B, double* C, int n_col) {

  for(int i=0; i<n_col;++i) {
    for(int j=0;j<n_col;++j){
      for(int k=0;k<N;++k) {
        //printf("C %d A %d B B %d \n", i*N+j,i*N+k,k*n_col+j);
        C[i*N+j]+=A[i*N+k]*B[k*n_col+j];
      }
    }
  }
}

void init_mat(double* A, int n_elem,int offset){
  for(int i=0;i<n_elem;++i)
    A[i]=i+offset;
}
void rank_mat(double* A, int n_elem,int rank){
  for(int i=0;i<n_elem;++i)
    A[i]=rank;
}

int main(int argc,char* argv[]) {
  MPI_Init(&argc,&argv);
  int procs,rank;
  MPI_Comm_size(MPI_COMM_WORLD,&procs);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Barrier(MPI_COMM_WORLD);

  int n_col=N/procs;
  double* A=malloc(N*n_col*sizeof(double));
  memset(A, 0, N*n_col);

  double* B=malloc(N*n_col*sizeof(double));
  memset(B, 0, N*n_col);
  rank_mat(B,N*n_col,rank); //ogni matrice ha il rank del possesore

  double* C=malloc(N*n_col*sizeof(double));
  memset(C, 0, N*n_col);

#ifdef DEBUG
    if(rank==0) {
      rank_mat(A,n_col*N,1);
      printf("Matricia a\n");
      print_matrix(A ,n_col);
      printf("\n");
    }
#endif


  MPI_Barrier(MPI_COMM_WORLD);
  double* buffer=malloc(N*n_col*sizeof(double));
  MPI_Datatype blocco;
  MPI_Type_vector(n_col,n_col,n_col*(n_col-2),MPI_DOUBLE,&blocco);
  MPI_Type_commit(&blocco);

  MPI_Allgather(B, 1, blocco,
		buffer , n_col*n_col, MPI_DOUBLE,
		MPI_COMM_WORLD);

#ifdef DEBUG
  if(rank==0) {
    printf("Stampo la colonna intera\n");
    print_matrix_transpose(buffer,n_col);
  }
#endif


#ifdef DEBUG
  if(rank==0){
    printf("Stampo na roba square!\n");
    mat_mul(A, buffer, C, n_col);
    print_matrix(C, n_col);
  }
#endif


MPI_Finalize();

}
