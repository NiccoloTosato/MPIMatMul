#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <cblas.h>

#ifndef N
   #define N 20
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
  //debug purpose: print a matrix
  for(int i=0;i<N;++i){
    for(int j=0;j<n_col;++j)
      printf("%3.3g ",A[j+i*n_col]);
    printf("\n");
  }
}

void print_matrix_square(double* A,int n_col) {
  //debug purpose: print a matrix
  for(int i=0;i<n_col;++i){
    for(int j=0;j<n_col;++j)
      printf("%3.3g ",A[j+i*n_col]);
    printf("\n");
  }
}

void mat_mul(double* restrict A, double* restrict B, double* restrict C, int n_col,int n_fix) {
  for(int i=0; i<n_fix;++i) {
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
  //init the matrix sequentially
  for(int i=0;i<n_elem;++i)
    A[i]=i+offset;
}
void rank_mat(double* A, int n_elem,int rank){
  //init the matrix with the value of the rank
  for(int i=0;i<n_elem;++i)
    A[i]=rank;
}

int calculate_col(int tot_col,int procs,int rank) {
  //calculate how many row belong to the current rank
  return (rank < tot_col % procs) ? tot_col/procs +1 : tot_col/procs;
}

void set_recvcout(int* recvcount, int iter, int procs){
  //set the recv_count array 
  int current=calculate_col(N,procs,iter);
  for(int p=0;p<procs;++p){
    recvcount[p]=calculate_col(N,procs,p)*current;
  }

}

void set_displacement(int* displacement,const int* recvcount,int procs) {
  //calculate the displacement array using the recv_count array
  displacement[0]=0;
  for(int p=1;p<procs;++p)
    displacement[p]=displacement[p-1]+recvcount[p-1];
}

int calculate_offset(int procs,int tot_col,int iter) {
  //calculate the offset for each block
  int n_resto = (tot_col / procs) + 1;
  int n_normale = (tot_col / procs);
  int diff = iter - (tot_col % procs);
    return (iter < tot_col % procs) ? n_resto*iter  : n_resto * (tot_col % procs) + n_normale * diff ;
}


void extract(double* destination ,double*source,int n_fix,int n_col) {
  //linearize the block
  for(int line=0;line<n_fix;++line) {
    memcpy( destination+n_col*line,source+N*line, n_col*sizeof(double));
  }
}

int main(int argc,char* argv[]) {
  MPI_Init(&argc,&argv);
  double comm_time=0;
  double tot_time=0;
  double accumulator=0;
  int procs,rank;
  MPI_Comm_size(MPI_COMM_WORLD,&procs);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  //dimensione locale per il blocco relativo al rank, non cambiera mai, se sono uno dei primi mi becco il resto
  int n_fix=(rank < N % procs) ? N/procs +1 : N/procs;
  //dimensione del buffer, deve essere in grado di contenere il blocco piu grande che c'e' in giro
  int n_buffer= ( (N%procs) >= 1) ? (N/procs)+1 : N/procs ;
  //ad ogni iterazione verra' rivisto il numero di colonne
  int n_col;

  double* A=malloc(N*n_fix*sizeof(double));
  //memset(A, 0, N*n_col);
  rank_mat(A,n_fix*N,rank);

  double* B=malloc(N*n_fix*sizeof(double));
  //memset(B, 0, N*n_col);
  rank_mat(B,N*n_fix,rank); //ogni matrice ha il rank del possesore

  double* C=malloc(N*n_fix*sizeof(double));
  memset(C, 0, N*n_fix);

  //allocate the buffer, it use the larger n_col possible
  double* buffer=malloc(N*n_buffer*sizeof(double));
  //allocate the buffer to linearize the block
  double* square=malloc(n_buffer*n_buffer*sizeof(double));

  //allocate displacement ad recvout array
  int* displacement = malloc(procs*sizeof(int));
  int* recvcount = malloc(procs*sizeof(int));


  //MPI_Datatype blocco;
  tot_time+=MPI_Wtime();
  for(int p=0;p<procs;++p){
    
    //numero di colonne all'iterazione corrente
    n_col=calculate_col(N,procs,p);
    set_recvcout(recvcount,p,procs);
    set_displacement(displacement,recvcount,procs);

    //MPI_Type_vector(n_buffer, n_buffer, N, MPI_DOUBLE, &blocco);
    //MPI_Type_commit(&blocco);

    int offset=calculate_offset(procs,N,p);

    
    //MPI_Type_size(blocco, &size);
    
    
    extract(square,B+offset,n_fix,n_col);

    comm_time=MPI_Wtime();

    MPI_Allgatherv( square , n_col*n_fix, MPI_DOUBLE,
                    buffer ,recvcount,displacement,MPI_DOUBLE,MPI_COMM_WORLD);
    comm_time=MPI_Wtime() - comm_time;
    accumulator+=comm_time;
    //MPI_Type_free(&blocco);

#ifdef DGEMM
    cblas_dgemm ( CblasRowMajor, CblasNoTrans, CblasNoTrans , n_fix , n_col , N , 1.0 , A , N , buffer , n_col , 0.0 ,  C+offset, N );

#else
    mat_mul(A, buffer, C+offset, n_col, n_fix);

#endif
#ifdef SUPER
    if(p>=(procs-3))
      printf("IT %d| ncol %d nfix %d offset %d rank %d\n",p,n_col,n_fix,offset,rank);
#endif

  }
  tot_time=MPI_Wtime()-tot_time;
  if(rank==0)
    printf("%d %f %f\n",procs,tot_time,accumulator);
  

#ifdef DEBUG

  double* C_final=malloc(N*N*sizeof(double));
  for(int p=0;p<procs;++p){
    recvcount[p]=N * calculate_col(N,procs,p);
  }

  set_displacement(displacement,recvcount,procs);

  double* B_final=malloc(N*N*sizeof(double));
  double* A_final=malloc(N*N*sizeof(double));
  double* C_final2=malloc(N*N*sizeof(double));
  MPI_Gatherv(A,
              N*n_fix,
              MPI_DOUBLE,
              A_final,
              recvcount,displacement,
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);
  MPI_Gatherv(B,
              N*n_fix,
              MPI_DOUBLE,
              B_final,
              recvcount,displacement,
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);
  MPI_Gatherv(C,
              N*n_fix,
              MPI_DOUBLE,
              C_final,
              recvcount,displacement,
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);
  if (rank==0) {
  cblas_dgemm ( CblasRowMajor, CblasNoTrans, CblasNoTrans , N , N , N , 1.0 , A_final , N , B_final , N , 0.0 ,  C_final2, N );
  for(int i=0;i<N*N;++i) {
    if (C_final[i] != C_final2[i])
      print("Errore");
  }
  }
#endif

#if ( defined DEBUG2 || defined DEBUG)
  if(rank==0){
    printf("FINALEEEE \n");
    print_matrix(C_final2,N);
  }
  #endif
  MPI_Finalize();

}
