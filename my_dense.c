#ifdef _GSL_
  #include "spmv.h"
#endif
#ifdef _MKL_
  #include "spmv_mkl.h"
#endif
#include <math.h>
#include <stdio.h>

int my_dense(const unsigned int n, const double mat[], double vec[], double result[])
{
  int row_offset=n, j=0, z=0, size=n*n;
  double tmp = 0.0;
  while(row_offset <= size){
  	for (; j < row_offset ; j++)
		tmp += mat[j] * vec[j%n];
	result[z] = tmp;
	tmp=0;
	++z;
	row_offset += n;
  }
  return 0;
}
