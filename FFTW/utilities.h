/* 
 * Created by G.P. Brandino, I. Girotto, R. Gebauer
 * Last revision: March 2016
 */

#ifndef _FFTW_UTLITIES_
#define _FFTW_UTLITIES_
#include <complex.h>
#include <fftw3-mpi.h>
#include <sys/time.h>
#include <stdbool.h>
#define pi 3.14159265358979323846

#include <mpi.h>

typedef struct {

#ifdef __HOMEMADE
  fftw_plan fw_plan; 
  fftw_plan bw_plan;
#else
  fftw_plan fw_1d_plan;
  fftw_plan bw_1d_plan;

  fftw_plan fw_2d_plan;
  fftw_plan bw_2d_plan;

  MPI_Datatype sendvector;
  MPI_Datatype recvector;
  MPI_Datatype* rdatatype;
  MPI_Datatype* sdatatype;
  int* sendcounts;
  int* recvcounts;
  int* rdispls;
  int* sdispls;
  fftw_complex *fftw_tmp;

#endif

  fftw_complex *fftw_data;
  ptrdiff_t global_size_grid;
  ptrdiff_t local_size_grid;
  ptrdiff_t local_n1;
  ptrdiff_t local_n1_offset;
  MPI_Comm mpi_comm;  
  
} fftw_mpi_handler;



double seconds();
inline int index_f ( int i1, int i2, int i3, int n1, int n2, int n3 );


void plot_data_1d( char* name, int n1, int n2, int n3, int n1_local, int  n1_local_offset, int dir, double* data );
void plot_data_2d( char* name, int n1, int n2, int n3, int n1_local, int  n1_local_offset, int dir, double* data );
void init_fftw( fftw_mpi_handler* fft, int n1, int n2, int n3, MPI_Comm comm );
void close_fftw( fftw_mpi_handler* fft );

void derivative( fftw_mpi_handler* fft,int n1, int n2, int n3, double L1, double L2, double L3, int ipol, double* data, double* deriv );
void fft_3d( fftw_mpi_handler* fft, double *data_direct, fftw_complex* data_rec, bool direct_to_reciprocal );

#endif