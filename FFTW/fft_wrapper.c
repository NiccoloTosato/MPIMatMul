/* Assignement:
 * Here you have to modify the includes, the array sizes and the fftw calls, to use the fftw-mpi
 *
 * Regarding the fftw calls. here is the substitution 
 * fftw_plan_dft_3d -> fftw_mpi_plan_dft_3d
 * ftw_execute_dft  > fftw_mpi_execute_dft 
 * use fftw_mpi_local_size_3d for local size of the arrays
 * 
 * Created by G.P. Brandino, I. Girotto, R. Gebauer
 * Last revision: March 2016
 *
 */ 

#include <stdbool.h>
#include <string.h>
#include "utilities.h"
#include <stdlib.h>

double seconds(){
/* Return the second elapsed since Epoch (00:00:00 UTC, January 1, 1970) */
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;
}

/* 
 *  Index linearization is computed following row-major order.
 *  For more informtion see FFTW documentation:
 *  http://www.fftw.org/doc/Row_002dmajor-Format.html#Row_002dmajor-Format
 *
 */
int index_f ( int i1, int i2, int i3, int n1, int n2, int n3){

  return n3*n2*i1 + n3*i2 + i3; 
}

void init_fftw(fftw_mpi_handler *fft, int n1, int n2, int n3, MPI_Comm comm){
  

  /*
   *  Allocate a distributed grid for complex FFT using aligned memory allocation
   *  See details here:
   *  http://www.fftw.org/fftw3_doc/Allocating-aligned-memory-in-Fortran.html#Allocating-aligned-memory-in-Fortran
   *  HINT: the allocation size is given by fftw_mpi_local_size_3d (see also http://www.fftw.org/doc/MPI-Plan-Creation.html)
   *
   */
  
  fft->global_size_grid = n1 * n2 * n3;
  fft->mpi_comm = comm;
  
  #ifdef __HOMEMADE
  /*
   * Call to fftw_mpi_init is needed to initialize a parallel enviroment for the fftw_mpi
   */

  fftw_mpi_init();
  fft->local_size_grid = fftw_mpi_local_size_3d( n1, n2, n3, fft->mpi_comm, &fft->local_n1, &fft->local_n1_offset);
  fft->fftw_data = ( fftw_complex* ) fftw_malloc( fft->local_size_grid * sizeof( fftw_complex ) );
  /*
   * Create an FFTW plan for a distributed FFT grid
   * Use fftw_mpi_plan_dft_3d: http://www.fftw.org/doc/MPI-Plan-Creation.html#MPI-Plan-Creation
   * easy part
   */
  fft->fw_plan = fftw_mpi_plan_dft_3d( n1, n2, n3, fft->fftw_data, fft->fftw_data, fft->mpi_comm, FFTW_FORWARD, FFTW_ESTIMATE );
  fft->bw_plan = fftw_mpi_plan_dft_3d( n1, n2, n3, fft->fftw_data, fft->fftw_data, fft->mpi_comm, FFTW_BACKWARD, FFTW_ESTIMATE );
  
  #else

  int size;
  MPI_Comm_size(comm, &size);
  int rank;
  MPI_Comm_rank(comm, &rank);
  //slab data points
  fft->local_size_grid = n1*n2*n3/size;
  //slab n1 size
  fft->local_n1=n1/size;
  fft->local_n1_offset=(n1/size)*rank;
  
  fft->fftw_data = ( fftw_complex* ) fftw_malloc( fft->local_size_grid * sizeof( fftw_complex ) );
  fft->fftw_tmp = ( fftw_complex* ) fftw_malloc( fft->local_size_grid * sizeof( fftw_complex ) );

/*
fftw_plan fftw_plan_many_dft(int rank, const int *n, int howmany,
                             fftw_complex *in, const int *inembed,
                             int istride, int idist,
                             fftw_complex *out, const int *onembed,
                             int ostride, int odist,
                             int sign, unsigned flags);
*/

int n23[]={n2, n3};
int n11[]={n1};

fft->fw_2d_plan =fftw_plan_many_dft(2, n23, n1/size, 
                            fft->fftw_data, n23,  
                            1, n2*n3, 
                            fft->fftw_data,   n23, 
                             1, n2*n3, FFTW_FORWARD, FFTW_ESTIMATE);

fft->fw_1d_plan =fftw_plan_many_dft(1, n11, n2 * n3 /size, 
                            fft->fftw_tmp, n11,  
                             n2 * n3 /size, 1, 
                            fft->fftw_tmp,  n11, 
                            n2 * n3 /size,1, FFTW_FORWARD, FFTW_ESTIMATE);

fft->bw_2d_plan =fftw_plan_many_dft(2, n23, n1/size, 
                            fft->fftw_data, n23,  
                            1, n2*n3, 
                            fft->fftw_data,   n23, 
                            1, n2*n3, FFTW_BACKWARD, FFTW_ESTIMATE);

fft->bw_1d_plan =fftw_plan_many_dft(1, n11, n2 * n3 /size, 
                            fft->fftw_tmp, n11,  
                             n2 * n3 /size, 1, 
                            fft->fftw_tmp,  n11, 
                            n2 * n3  / size,1, FFTW_BACKWARD, FFTW_ESTIMATE);

/*
  fft->bw_plan_1d =
      fftw_plan_many_dft(1, n_1d, nz * ny / size, fft->buffer, n_1d, nz * ny / size, 1, fft->buffer,
                         n_1d, nz * ny / size, 1, FFTW_BACKWARD, FFTW_ESTIMATE);
*/

/*
 int MPI_Type_vector(int count,
                    int blocklength, int stride, MPI_Datatype oldtype, MPI_Datatype * newtype)
MPI_C_DOUBLE_COMPLEX, NOT USE MPI_DOUBLE_COMPLEX
*/

//slab size N1_loc X N2 X N3
// send datatype
MPI_Datatype sendvector;
MPI_Type_vector(fft->local_n1,
                    n2 * n3 / size, n2 * n3, MPI_C_DOUBLE_COMPLEX, &sendvector);

MPI_Type_commit(&sendvector);
fft->sendvector=sendvector;

//stick size N1_loc X N2_loc X N3
// recv datatype
MPI_Datatype recvector;
MPI_Type_vector(1,fft->local_size_grid / size, 0, MPI_C_DOUBLE_COMPLEX, &recvector);
MPI_Type_commit(&recvector);

fft->recvector=recvector;

/*
int MPI_Alltoallv(const void *sendbuf, const int *sendcounts,
                  const int *sdispls, MPI_Datatype sendtype, void *recvbuf,
                  const int *recvcounts, const int *rdispls, MPI_Datatype recvtype,
                  MPI_Comm comm)
*/

int* sendcounts=malloc(size*sizeof(int));
int* recvcounts=malloc(size*sizeof(int));
int* rdispls=malloc(size*sizeof(int));
int* sdispls=malloc(size*sizeof(int));

MPI_Datatype* rdatatype=malloc(size*sizeof(MPI_Datatype));
MPI_Datatype* sdatatype=malloc(size*sizeof(MPI_Datatype));

for(int i=0;i<size;++i) {
  sendcounts[i]=1;
  recvcounts[i]=1;
  rdatatype[i]=recvector;
  sdatatype[i]=sendvector;
  sdispls[i]=i*n2*n3/size * sizeof(fftw_complex);
  rdispls[i]=i*fft->local_size_grid / size * sizeof(fftw_complex);
}

sdispls[0]=0;
rdispls[0]=0;
/*
for(int i=1;i>size;++i) {
  sdispls[i]=sdispls[i-1]+ n2*n3/size * sizeof(fftw_complex);
  //salto la dimensione del cubetto
  rdispls[i]=rdispls[i-1]+ fft->local_size_grid / size * sizeof(fftw_complex);
}
*/

fft->rdatatype=rdatatype;
fft->sdatatype=sdatatype;

fft->sendcounts=sendcounts;
fft->recvcounts=recvcounts;

fft->sdispls=sdispls;
fft->rdispls=rdispls;

#endif

}

void close_fftw( fftw_mpi_handler *fft ){
    //fftw_destroy_plan( fft->bw_plan );
    //fftw_destroy_plan( fft->fw_plan );
    //fftw_free( fft->fftw_data );
}

/* This subroutine uses fftw to calculate 3-dimensional discrete FFTs.
 * The data in direct space is assumed to be real-valued
 * The data in reciprocal space is complex. 
 * direct_to_reciprocal indicates in which direction the FFT is to be calculated
 * 
 * Note that for real data in direct space (like here), we have
 * F(N-j) = conj(F(j)) where F is the array in reciprocal space.
 * Here, we do not make use of this property.
 * Also, we do not use the special (time-saving) routines of FFTW which
 * allow one to save time and memory for such real-to-complex transforms.
 *
 * f: array in direct space
 * F: array in reciprocal space
 * 
 * F(k) = \sum_{l=0}^{N-1} exp(- 2 \pi I k*l/N) f(l)
 * f(l) = 1/N \sum_{k=0}^{N-1} exp(+ 2 \pi I k*l/N) F(k)
 * 
 */

void fft_3d( fftw_mpi_handler* fft, double *data_direct, fftw_complex* data_rec, bool direct_to_reciprocal ){

    double fac;
    int i;
    // Now distinguish in which direction the FFT is performed
    if( direct_to_reciprocal ){
      for(i = 0; i < fft->local_size_grid; i++){
      	  fft->fftw_data[i] =  data_direct[i] + 0.0 * I;
      	}   
  #ifdef __HOMEMADE
    fftw_mpi_execute_dft( fft->fw_plan, fft->fftw_data, fft->fftw_data );
  #else
    fftw_execute(fft->fw_2d_plan); // 2D transform along planes
    //fftw_print_plan(fft->fw_2d_plan);
    MPI_Alltoallw(fft->fftw_data, fft->sendcounts, fft->sdispls, fft->sdatatype, fft->fftw_tmp, fft->recvcounts, fft->rdispls, fft->rdatatype,
                  fft->mpi_comm);
    fftw_execute(fft->fw_1d_plan);
    MPI_Alltoallw(fft->fftw_tmp, fft->recvcounts, fft->rdispls, fft->rdatatype, fft->fftw_data, fft->sendcounts, fft->sdispls, fft->sdatatype,
                  fft->mpi_comm);   
  #endif
   memcpy( data_rec, fft->fftw_data, fft->local_size_grid * sizeof(fftw_complex) );
  } else {   
  memcpy(fft->fftw_data, data_rec, fft->local_size_grid * sizeof(fftw_complex) );

  #ifdef __HOMEMADE
  fftw_mpi_execute_dft(fft->bw_plan, fft->fftw_data, fft->fftw_data);
  #else

  fftw_execute(fft->bw_2d_plan); // 2D transform along planes
    MPI_Alltoallw(fft->fftw_data, fft->sendcounts, fft->sdispls, fft->sdatatype, fft->fftw_tmp, fft->recvcounts, fft->rdispls, fft->rdatatype,
                  fft->mpi_comm);
    fftw_execute(fft->fw_1d_plan);
    MPI_Alltoallw(fft->fftw_tmp, fft->recvcounts, fft->rdispls, fft->rdatatype, fft->fftw_data, fft->sendcounts, fft->sdispls, fft->sdatatype,
                  fft->mpi_comm);  
 #endif
      fac = 1.0 / fft->global_size_grid;     
      for( i = 0; i < fft->local_size_grid; ++i ){
      	data_direct[i] = creal( fft->fftw_data[i] ) * fac;
      }
    }
}

