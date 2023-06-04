#ifdef _OPENACC
#include <accel.h>              // OpenACC
#endif
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

int main(int argc, char* argv[]){
  MPI_Init(&argc,&argv);
  int commsize,rank;
  MPI_Comm_size(MPI_COMM_WORLD,&commsize);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  acc_init(acc_device_nvidia);                                 // OpenACC call
  const int num_dev = acc_get_num_devices(acc_device_nvidia);  // #GPUs
  const int dev_id = rank % num_dev;         
  acc_set_device_num(dev_id,acc_device_nvidia); // assign GPU to one MPI process

  int send_to,recv_from;
  if(rank==0) {
    send_to=1;
    recv_from=1;
  } else {
    recv_from=0;
    send_to=0;
  }
  
  int size=100000;
  double* matrix = ( double* )malloc( sizeof(double)*size );
  double* matrix_new = ( double* )malloc( sizeof(double)*size );
#pragma acc enter data copyin(matrix[:size]) create(matrix_new[:size]) 
#pragma acc host_data use_device(matrix,matrix_new)
  { 

    MPI_Sendrecv(matrix, size, MPI_DOUBLE,
                 send_to, 0,
                 matrix_new, size, MPI_DOUBLE,
                 recv_from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

 }

#pragma acc kernels present(matrix[:size],matrix_new[:size]) 
  {
  for(int i = 1 ; i < size; ++i )
    matrix_new[i]=2*matrix[i];
  }

#pragma acc exit data copyout(matrix[:size],matrix_new[:size]) 

  return 0;
}
