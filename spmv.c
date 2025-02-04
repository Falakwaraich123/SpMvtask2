#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef _GSL_
  #include <gsl/gsl_cblas.h>      
  #include <gsl/gsl_spmatrix.h>
  #include <gsl/gsl_vector.h>
  #include "spmv.h"
#endif
#ifdef _MKL_
  #include <mkl.h>
  #include "spmv_mkl.h"
#endif

#include "timer.h"
//#include "csr.h"

#define DEFAULT_SIZE 1024
#define DEFAULT_DENSITY 0.25

unsigned int populate_sparse_matrix(double mat[], unsigned int n, double density, unsigned int seed)
{
  unsigned int nnz = 0;

  srand(seed);

  for (unsigned int i = 0; i < n * n; i++) {
    if ((rand() % 100) / 100.0 < density) {
      mat[i] = ((double)(rand() % 10) + (double)rand() / RAND_MAX) * (rand() % 2 == 0 ? 1 : -1);
      nnz++;
    } else {
      mat[i] = 0;
    }
  }

  return nnz;
}

unsigned int populate_vector(double vec[], unsigned int size, unsigned int seed)
{
  srand(seed);
  for (unsigned int i = 0; i < size; i++) {
    vec[i] = ((double)(rand() % 10) + (double)rand() / RAND_MAX) * (rand() % 2 == 0 ? 1 : -1);
  }

  return size;
}

int is_nearly_equal(double x, double y)
{
  const double epsilon = 1e-5 
  return fabs(x - y) <= epsilon * fabs(x);
}

unsigned int check_result(double ref[], double result[], unsigned int size)
{
  for(unsigned int i = 0; i < size; i++) {
    if (!is_nearly_equal(ref[i], result[i]))
      return 0;
  }

  return 1;
}

int main(int argc, char *argv[])
{
  int size;       
  double density; 
  double value;
  int i, j;

  if (argc < 2) {
    size = DEFAULT_SIZE;
    density = DEFAULT_DENSITY;
  } else if (argc < 3) {
    size = atoi(argv[1]);
    density = DEFAULT_DENSITY;
  } else {
    size = atoi(argv[1]);
    density = (double) atoi(argv[2]) / 100.0;
  }
  
  double *mat, *vec, *refsol, *mysol;

  mat = (double *) malloc(size * size * sizeof(double));
  vec = (double *) malloc(size * sizeof(double));
  refsol = (double *) malloc(size * sizeof(double));
  mysol = (double *) malloc(size * sizeof(double));

  unsigned int nnz = populate_sparse_matrix(mat, size, density, 1);
  populate_vector(vec, size, 2);

  printf("Matriz size: %d x %d (%d elements)\n", size, size, size*size);
  printf("%d non-zero elements (%.2lf%%)\n\n", nnz, (double) nnz / (size*size) * 100.0);

 
  printf("Dense computation\n----------------\n");

  timeinfo start, now;
  timestamp(&start);

  cblas_dgemv(CblasRowMajor, CblasNoTrans, size, size, 1.0, mat, size, vec, 1, 0.0, refsol, 1);

  timestamp(&now);

  #ifdef _GSL_
  printf("Time taken by CBLAS (GSL) dense computation: %ld ms\n", diff_milli(&start, &now));
  #endif
  #ifdef _MKL_
  printf("Time taken by CBLAS (MKL) dense computation: %ld ms\n", diff_milli(&start, &now));
  #endif

  timestamp(&start);

  my_dense(size, mat, vec, mysol);

  timestamp(&now);
  #ifdef _GSL_
  printf("Time taken by my dense matrix-vector product (GSL): %ld ms\n", diff_milli(&start, &now));
  #endif

  #ifdef _MKL_
  printf("Time taken by my dense matrix-vector product (MKL): %ld ms\n", diff_milli(&start, &now));
  #endif

  if (check_result(refsol, mysol, size) == 1)
    printf("Result is ok!\n");
  else
    printf("Result is wrong!\n");

  #ifdef _GSL_
  gsl_spmatrix *m = gsl_spmatrix_compress(gsl_spmatrix_alloc(size, size), GSL_SPMATRIX_CSR); //gsl_spmatrix in CSR format
  gsl_spmatrix *src = gsl_spmatrix_alloc(size, size);	   // gsl_spmatrix 
  for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            value = mat[i * size + j];
            if (value != 0.0) {
                gsl_spmatrix_set(src, i, j, value);
            }
        }
    }
  gsl_spmatrix_csr(m, src);
 
  
  gsl_vector *x = gsl_vector_alloc(size);

  for (i = 0; i < size; i++) {
  	gsl_vector_set(x, i, vec[i]);
  }

  gsl_vector *y = gsl_vector_alloc(size);
  gsl_vector_set_zero(y);
  #endif
  
 
 
  #ifdef _MKL_
  MKL_INT row_ind[nnz]; 
  MKL_INT col_ind[nnz];
  int u=0;
  double values[nnz];
  for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            value = mat[i * size + j];
            if (value != 0.0) {
	      values[u]=value;
	      row_ind[u]=i;
	      col_ind[u]=j;
	      u++;
            }
        }
    }
  sparse_matrix_t src;
  sparse_matrix_t m;
  sparse_status_t status;

   status = mkl_sparse_d_create_coo(&src, SPARSE_INDEX_BASE_ZERO, size, size, nnz, row_ind, col_ind, values);
  if (status != SPARSE_STATUS_SUCCESS) {
        printf("Error creating the matrix\n");
        return 1;
  }
  status = mkl_sparse_convert_csr(src, SPARSE_OPERATION_NON_TRANSPOSE, &m);
  if (status != SPARSE_STATUS_SUCCESS) {
        printf("Error creating the matrix\n");
        return 1;
  }
  sparse_index_base_t indexing;
  double *csr_values;
  MKL_INT nrows, ncols; 
  MKL_INT *rows_start, *rows_end, *cols_indx;

  mkl_sparse_d_export_csr(m, &indexing, &nrows, &ncols, &rows_start, &rows_end, &cols_indx, &csr_values);

  char transa = 'N';
  double alpha = 1.0, beta= 0.0;
   struct matrix_descr descrA;
  descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
  descrA.diag = SPARSE_DIAG_NON_UNIT;
  #endif

  timestamp(&start);

#ifdef _GSL_
  gsl_spblas_dgemv(CblasNoTrans, 1.0, m, x, 0.0, y);
  #endif
  #ifdef _MKL_
 
  status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, m, descrA, vec, beta, mysol);

  if (status != SPARSE_STATUS_SUCCESS) {
        printf(" Error in CSR mkl_sparse_d_mv: %d \n", status);
	return 1;
  }
  #endif

  timestamp(&now);

  #ifdef _GSL_
  printf("Time taken by GSL (CSR) sparse matrix - vector product: %ld ms\n", diff_milli(&start, &now));
  for(i=0; i < size; i++){
	mysol[i] = gsl_vector_get(y, i);
  }
  #endif

  #ifdef _MKL_
  printf("Time taken by MKL (CSR) sparse matrix - vector product: %ld ms\n", diff_milli(&start, &now));
  #endif
  
  
  if (check_result(refsol, mysol, size) == 1)
    printf("Result is ok!\n");
  else
    printf("Result is wrong!\n");

 timestamp(&start);

  #ifdef _GSL_
  my_csr(size, m, vec, mysol);
  #endif
  #ifdef _MKL_
  my_csr(size, &m, vec, mysol);
  #endif

  timestamp(&now);
  #ifdef _GSL_
  printf("Time taken by my csr matrix (GSL) - vector product: %ld ms\n", diff_milli(&start, &now));
  #endif
  #ifdef _MKL_
  printf("Time taken by my csr matrix (MKL) - vector product: %ld ms\n", diff_milli(&start, &now));
  #endif

  if (check_result(refsol, mysol, size) == 1)
    printf("Result is ok!\n");
  else
    printf("Result is wrong!\n");

 #ifdef _GSL_
  gsl_spmatrix_free(m);
  m = gsl_spmatrix_compress(gsl_spmatrix_alloc(size, size), GSL_SPMATRIX_CSC); //gsl_spmatrix in CSC format
  gsl_spmatrix_csc(m, src);
  #endif
  #ifdef _MKL_

  MKL_INT job[6] = { 0, 0, 0, 0, nnz, 1 };
  double csc_values[nnz];
  MKL_INT csc_row_indices[nnz];
  MKL_INT csc_col_ptr[size + 1];
  MKL_INT info;
  sparse_matrix_t cscA;

 mkl_dcsrcsc(job, &size , csr_values, cols_indx, rows_start, csc_values, csc_row_indices, csc_col_ptr, &info);
  status = mkl_sparse_d_create_csc( &cscA,
                                      SPARSE_INDEX_BASE_ZERO,
                                      size,  
                                      size,  
                                      csc_col_ptr,
                                      csc_col_ptr+1,
                                      csc_row_indices,
                                      csc_values );
  #endif

  timestamp(&start);

  #ifdef _GSL_
  gsl_spblas_dgemv(CblasNoTrans, 1.0, m, x, 0.0, y);
  #endif

  #ifdef _MKL_
  status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, cscA, descrA, vec, beta, mysol);

  if (status != SPARSE_STATUS_SUCCESS) {
        printf(" Error in CSC mkl_sparse_d_mv: %d \n", status);
	return 1;
  }
  #endif

  timestamp(&now);

  #ifdef _GSL_
  printf("Time taken by GSL (CSC) sparse matrix - vector product: %ld ms\n", diff_milli(&start, &now));

  for(i=0; i < size; i++){
	mysol[i] = gsl_vector_get(y, i);
  }
  #endif

  #ifdef _MKL_
  printf("Time taken by MKL (CSC) sparse matrix - vector product: %ld ms\n", diff_milli(&start, &now));
  #endif

  if (check_result(refsol, mysol, size) == 1)
    printf("Result is ok!\n");
  else
    printf("Result is wrong!\n");

  timestamp(&start);

  #ifdef _GSL_
  my_csc(size, m, vec, mysol);
  #endif
  #ifdef _MKL_
  my_csc(size, csc_col_ptr, csc_row_indices, csc_values, vec, mysol);
  #endif

  timestamp(&now);
  #ifdef _GSL_
  printf("Time taken by my csc matrix (GSL) - vector product: %ld ms\n", diff_milli(&start, &now));
  #endif
  #ifdef _MKL_
  printf("Time taken by my csc matrix (MKL) - vector product: %ld ms\n", diff_milli(&start, &now));
  #endif

  if (check_result(refsol, mysol, size) == 1)
    printf("Result is ok!\n");
  else
    printf("Result is wrong!\n");
    timestamp(&start);

  #ifdef _GSL_
  gsl_spblas_dgemv(CblasNoTrans, 1.0, src, x, 0.0, y);
  #endif

  #ifdef _MKL_
  status = mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, alpha, src, descrA, vec, beta, mysol);
  if (status != SPARSE_STATUS_SUCCESS) {
        printf(" Error in COO mkl_sparse_d_mv: %d \n", status);
	return 1;
  }

  #endif
  timestamp(&now);

  #ifdef _GSL_
  printf("Time taken by GSL (COO) sparse matrix - vector product: %ld ms\n", diff_milli(&start, &now));

  for(i=0; i < size; i++){
	mysol[i] = gsl_vector_get(y, i);
  }
  #endif
  #ifdef _MKL_
  printf("Time taken by MKL (COO) sparse matrix - vector product: %ld ms\n", diff_milli(&start, &now));
  #endif

  if (check_result(refsol, mysol, size) == 1)
    printf("Result is ok!\n");
  else
    printf("Result is wrong!\n");

  timestamp(&start);

  #ifdef _GSL_
  my_coo(size, src, vec, mysol);
  #endif
  #ifdef _MKL_
  my_coo(size, nnz, row_ind, col_ind, values, vec, mysol);
  #endif

  timestamp(&now);
  #ifdef _GSL_
  printf("Time taken by my coo matrix (GSL) - vector product: %ld ms\n", diff_milli(&start, &now));
  #endif
  #ifdef _MKL_
  printf("Time taken by my coo matrix (MKL) - vector product: %ld ms\n", diff_milli(&start, &now));
  #endif

  if (check_result(refsol, mysol, size) == 1)
    printf("Result is ok!\n");
  else
    printf("Result is wrong!\n");


 free(mat);
  free(vec);
  free(refsol);
  free(mysol);
  #ifdef _GSL_
  gsl_spmatrix_free(m);
  gsl_spmatrix_free(src);
  gsl_vector_free(x);
  gsl_vector_free(y);
  #endif
  #ifdef _MKL_
  mkl_sparse_destroy(src);
  mkl_sparse_destroy(m);
  mkl_sparse_destroy(cscA);
  
  #endif

  return 0;
}
