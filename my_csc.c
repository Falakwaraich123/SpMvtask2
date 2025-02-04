#ifdef _GSL_
#include "spmv.h"
#include <gsl/gsl_spmatrix.h>
#endif
#ifdef _MKL_
#include <mkl.h>
#include "spmv_mkl.h"
#endif

#ifdef _GSL_
int my_csc(const unsigned int n, gsl_spmatrix *m, double vec[], double result[])
{
  int i, j=0, k=0;
  for (i=0; i < n; i++)
  	result[i]=0;

  for(i=0; i < n; i++){
  	if ((j = m->p[i+1]) != m->p[i]){
		for(; k < j; k++){
	
			result[m->i[k]] += m->data[k] * vec[i];
		}
	}
  }
  return 0;
}
#endif
#ifdef _MKL_
int my_csc(const unsigned int n, MKL_INT *cols_start, MKL_INT *rows_indx, const double *values, double vec[], double result[])
{
  int i, j=0, k=0;
  for (i=0; i < n; i++)
  	result[i]=0;
  for(i=0; i < n; i++){
  	if ((j = cols_start[i+1]) != cols_start[i]){
		for(; k < j; k++){
			result[rows_indx[k]] += values[k] * vec[i];
		}
	}
  }
  return 0;
}
#endif
