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
    int row=i*N;
    for(int j=0;j<n_col;++j){
      int idx=row+j;
      for(int k=0;k<N;++k) {
        C[idx]+=A[row+k]*B[k*n_col+j];
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
  //memset(A, 0, N*n_col);
  rank_mat(A,n_col*N,rank);

  double* B=malloc(N*n_col*sizeof(double));
  //memset(B, 0, N*n_col);
  rank_mat(B,N*n_col,rank); //ogni matrice ha il rank del possesore

  double* C=malloc(N*n_col*sizeof(double));
  memset(C, 0, N*n_col);

  double* buffer=malloc(N*n_col*sizeof(double));


  

  MPI_Datatype blocco;
  MPI_Type_vector(n_col,n_col,n_col*(n_col-2),MPI_DOUBLE,&blocco);
  MPI_Type_commit(&blocco);


  for(int p=0;p<procs;++p){
      MPI_Allgather(B+n_col*p, 1, blocco,
                  buffer , n_col*n_col, MPI_DOUBLE,
                  MPI_COMM_WORLD);

#ifdef DEBUG
    if(rank==2) {
      printf("Stampo la colonna buffer\n");
      print_matrix_transpose(buffer,n_col);
    }
#endif


#ifdef DEBUG
    if(rank==2){
      printf("Stampo la matrice C!\n");
      print_matrix(C, n_col);
    }
#endif
    mat_mul(A, buffer, C+n_col*p, n_col);
  }

  double* C_final=malloc(N*N*sizeof(double));
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Gather(C,
             N*n_col,
             MPI_DOUBLE,
             C_final,
             N*n_col,
             MPI_DOUBLE,
             0,
             MPI_COMM_WORLD);

  if(rank==0){
    printf("FINALEEEE \n");
    print_matrix(C_final,N);
  }
  MPI_Finalize();

}
