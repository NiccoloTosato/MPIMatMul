#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <cblas.h>
#include "auxiliary.h"
#include <cublas.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

int main(int argc,char* argv[]) {
  MPI_Init(&argc,&argv);
  double comm_time=0;
  double tot_time=0;
  double accumulator=0;
  int procs,rank;
  MPI_Comm_size(MPI_COMM_WORLD,&procs);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  int N=20;
  if (argc > 1)
    N=atoi(argv[1]);
  //dimensione locale per il blocco relativo al rank, non cambiera mai, se sono uno dei primi mi becco il resto
  int n_fix=(rank < N % procs) ? N/procs +1 : N/procs;
  //dimensione del buffer, deve essere in grado di contenere il blocco piu grande che c'e' in giro
  int n_buffer= ( (N%procs) >= 1) ? (N/procs)+1 : N/procs ;
  //ad ogni iterazione verra' rivisto il numero di colonne
  int n_col;

  double* A= (double*) malloc(N*n_fix*sizeof(double));
  //memset(A, 0, N*n_col);
  rank_mat(A,n_fix*N,rank); //ogni matrice ha il rank del possesore

  double* B= (double*) malloc(N*n_fix*sizeof(double));
  //memset(B, 0, N*n_col);
  rank_mat(B,N*n_fix,rank); //ogni matrice ha il rank del possesore

  double* C= (double*) malloc(N*n_fix*sizeof(double));
  memset(C, 0, N*n_fix*sizeof(double)); //inizializzo C a zero

  //allocate the buffer, it use the larger n_col possible
  double* buffer= (double*)  malloc(N*n_buffer*sizeof(double)); 
  //allocate the buffer to linearize the block
  double* square= (double*)  malloc(n_buffer*n_buffer*sizeof(double));

  //allocate displacement ad recvout array
  int* displacement = (int*)  malloc(procs*sizeof(int));
  int* recvcount = (int*)  malloc(procs*sizeof(int));

#ifdef CUBLAS
  double* A_device;
  double* buffer_device;
  double* C_device;
  cudaMalloc((void**)&A_device, N * n_fix * sizeof(double));
  cudaMalloc((void**)&buffer_device, N * n_fix * sizeof(double));
  cudaMalloc((void**)&C_device, n_fix * n_fix * sizeof(double));

  cudaMemcpy(A_device, A, N * n_fix * sizeof(double),cudaMemcpyHostToDevice);
  cublasHandle_t handle;
  cublasCreate(&handle);
#endif
  
  //MPI_Datatype blocco;

  MPI_Barrier(MPI_COMM_WORLD);
  tot_time+=MPI_Wtime();
  for(int p=0;p<procs;++p){

    //numero di colonne all'iterazione corrente
    n_col=calculate_col(N,procs,p);
    set_recvcout(recvcount,p,procs,N);
    set_displacement(displacement,recvcount,procs);

    //MPI_Type_vector(n_buffer, n_buffer, N, MPI_DOUBLE, &blocco);
    //MPI_Type_commit(&blocco);

    int offset=calculate_offset(procs,N,p);
    
    //MPI_Type_size(blocco, &size);

    comm_time=MPI_Wtime();    
    extract(square,B+offset,n_fix,n_col,N);

    MPI_Allgatherv( square , n_col*n_fix, MPI_DOUBLE,
                    buffer ,recvcount,displacement,MPI_DOUBLE,MPI_COMM_WORLD);
#ifdef CUBLAS
    cudaMemcpy(buffer_device, buffer, N * n_col * sizeof(double),cudaMemcpyHostToDevice);
#endif

    comm_time=MPI_Wtime() - comm_time;
    accumulator+=comm_time;
    //MPI_Type_free(&blocco);

#ifdef DGEMM
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans , n_fix , n_col , N , 1.0 , A , N , buffer , n_col , 0.0 ,  C+offset, N );
#elif CUBLAS
    /*
    cublasStatus_t cublasDgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc) */
    double alpha = 1.0;
        double beta = 0.0;	
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_fix, n_col, N, &alpha, buffer_device, n_fix, A_device, N, &beta, C_device, n_fix);
    for(int i=0;i<n_fix;++i)
      cudaMemcpy(C+offset+N*i, C_device+n_fix*i, n_col *  sizeof(double),cudaMemcpyDeviceToHost);
#else
    mat_mul(A, buffer, C+offset, n_col, n_fix,N);
#endif

#ifdef SUPER
    if(p>=(procs-3))
      printf("IT %d| ncol %d nfix %d offset %d rank %d\n",p,n_col,n_fix,offset,rank);
#endif
  }
  MPI_Barrier(MPI_COMM_WORLD);
  tot_time=MPI_Wtime()-tot_time;
  if(rank==0)
    printf("%d %f %f\n",procs,tot_time,accumulator);


#if (defined TEST || defined DEBUG)
  double* C_final=NULL;
  double* B_final=NULL;
  double* A_final=NULL;
  double* C_final_dgemm=NULL;
    
  if(rank==0) {
    //il rank 0 raccoglie tutto quanto, quindi inizializza 3 matrici NxN
    C_final=malloc(N*N*sizeof(double));
    B_final=malloc(N*N*sizeof(double));
    A_final=malloc(N*N*sizeof(double));

    //alloco una matrice NxN da calcolare cona la dgemm
    C_final_dgemm=malloc(N*N*sizeof(double));

    for(int p=0;p<procs;++p){
      recvcount[p]=N * calculate_col(N,procs,p);
    }
    set_displacement(displacement,recvcount,procs);
  }
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
    cblas_dgemm ( CblasRowMajor, CblasNoTrans, CblasNoTrans , N , N , N , 1.0 , A_final , N , B_final , N , 0.0 ,  C_final_dgemm, N );

    int flag=0;
    for(int i=0;i<N*N;++i) {
      if (abs(C_final[i] - C_final_dgemm[i]) > 1E-6) {
        printf("Errore %f %f\n",C_final[i],C_final_dgemm[i]);
        flag++;
      }
    }
    printf("Errori rilevati: %d\n",flag);
  }

#ifdef DEBUG
  if(rank==0){
    printf("Matrice risultante:\n");
    print_matrix(C_final,N,N);
  }
#endif


#endif


  MPI_Finalize();
}
