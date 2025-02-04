#include <stddef.h>
#include <mkl.h>

int my_dense(const unsigned int n, const double mat[], double vec[], double result[]);
int my_csr(const unsigned int n, sparse_matrix_t *m, double vec[], double result[]);
int my_csc(const unsigned int n, MKL_INT *cols_start, MKL_INT *rows_indx, const double *values, double vec[], double result[]);
int my_coo(const unsigned int n, const unsigned int nnz, MKL_INT *rows_indx, MKL_INT *cols_indx, const double *values, double vec[], double result[]);
