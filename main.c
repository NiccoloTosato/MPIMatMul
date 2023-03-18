#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <cblas.h>

#ifndef N
   #define N 18
#endif


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

void mat_mul(double* restrict A, double* restrict B, double* restrict C, int n_col) {
  for(int i=0; i<n_col;++i) {
    int register row=i*N;
    for(int j=0;j<n_col;++j){
      int register idx=row+j;
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

int calculate_col(int tot_col,int procs,int rank) {
  return (rank < tot_col % procs) ? tot_col/procs +1 : tot_col/procs;
}

void set_recvcout(int* recvcount, int iter, int procs){
  int current=calculate_col(N,procs,iter);
  for(int p=0;p<procs;++p){
    recvcount[p]=calculate_col(N,procs,p)*current;
  }
}

void set_displacement(int* displacement,const int* recvcount,int procs) {
  displacement[0]=0;
  for(int p=1;p<procs;++p)
    displacement[p]=displacement[p-1]+recvcount[p];
}

int main(int argc,char* argv[]) {
  MPI_Init(&argc,&argv);
  int procs,rank;
  MPI_Comm_size(MPI_COMM_WORLD,&procs);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Barrier(MPI_COMM_WORLD);

  //dimensione locale per il blocco relativo al rank, non cambiera mai, se sono uno dei primi mi becco il resto
  int n_fix=(rank < N % procs) ? N/procs +1 : N/procs;
  //dimensione del buffer, deve essere in grado di contenere il blocco piu grande che c'e' in giro
  int n_buffer= ( (N%procs) >= 1) ? (N/procs)+1 : N/procs ;
  //ad ogni iterazione verra' rivisto il numero di colonne
  int n_col;

#ifdef DEBUG3
  printf("rank %d n_fix %d, n_buffer %d\n",rank,n_fix,n_buffer);
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);
#endif


  double* A=malloc(N*n_fix*sizeof(double));
  //memset(A, 0, N*n_col);
  rank_mat(A,n_fix*N,rank);

  double* B=malloc(N*n_fix*sizeof(double));
  //memset(B, 0, N*n_col);
  rank_mat(B,N*n_fix,rank); //ogni matrice ha il rank del possesore

  double* C=malloc(N*n_fix*sizeof(double));
  memset(C, 0, N*n_fix);

  double* buffer=malloc(N*n_buffer*sizeof(double));

  MPI_Datatype blocco;

  int* displacement = malloc(procs*sizeof(int));
  int* recvcount = malloc(procs*sizeof(int));

  #ifdef SUPER
  if(rank==0){
    for(int pp=0;pp<procs;++pp){
      int current=calculate_col(N,procs,pp);
      printf("IT %d , n_col %d : ",pp,current);
      set_recvcout(recvcount,pp,procs);
      set_displacement(displacement,recvcount,procs);
      for(int p=0;p<procs;++p){
        printf(" %d|%d ",recvcount[p],displacement[p]);
      }
      printf("\n");
    }
  }
  return 0;
  #endif
    for(int p=0;p<procs;++p){
    //numero di colonne all'iterazione corrente

      n_col=calculate_col(N,procs,p);

    MPI_Type_vector(n_fix,n_col,N,MPI_DOUBLE,&blocco);
    MPI_Type_commit(&blocco);
    
    MPI_Allgather(B+n_col*p, 1, blocco,
                  buffer , n_col*n_col, MPI_DOUBLE,
                  MPI_COMM_WORLD);
#ifdef DEBUG2
    if(rank==3) {
      printf("Stampo la colonna buffer\n");
      print_matrix(buffer,n_col);
    }
#endif


    mat_mul(A, buffer, C+n_col*p, n_col);
    //cblas_dgemm ( CblasRowMajor, CblasNoTrans, CblasNoTrans , n_col , n_col , N , 1.0 , A , N , buffer , n_col , 0.0 ,  C+n_col*p, N );

#ifdef DEBUG2
    if(rank==3){
      printf("Stampo la matrice C!\n");
      print_matrix(C, n_col);
    }
#endif


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
#if ( defined DEBUG2 || defined DEBUG)
  if(rank==0){
    printf("FINALEEEE \n");
    print_matrix(C_final,N);
  }
  #endif
  MPI_Finalize();

}
