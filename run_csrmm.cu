#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <random>
#include <vector>
#include <string>
#include <iostream>
#include "cusparse.h"
#include "load_data.h"
#include "utility.h"

#define CLEANUP(s)                          \
  do {                                      \
    printf("%s\n", s);                      \
    if (hostCsrRowPtr) free(hostCsrRowPtr); \
    if (hostCsrColInd) free(hostCsrColInd); \
    if (hostCsrVal) free(hostCsrVal);       \
    if (yHostPtr) free(yHostPtr);           \
    if (zHostPtr) free(zHostPtr);           \
    if (csrRowPtr) cudaFree(csrRowPtr);     \
    if (csrColInd) cudaFree(csrColInd);     \
    if (csrVal) cudaFree(csrVal);           \
    if (y) cudaFree(y);                     \
    if (z) cudaFree(z);                     \
    cudaDeviceReset();                      \
    fflush(stdout);                         \
  } while (0)

#define HANDLE_ERROR(err)                     \
  if (!checkError(err, __FILE__, __LINE__)) { \
    CLEANUP("CUDA ERROR");                    \
    exit(-1);                                 \
  }

#define HANDLE_CUSPARSE_ERROR(err)                    \
  if (!checkCusparseError(err, __FILE__, __LINE__)) { \
    CLEANUP("CUSPARSE ERROR");                        \
    exit(-1);                                         \
  }

int main(int argc, char* argv[]) {
  std::string prefix = "tmp/" + std::string(argv[1]);
  std::cout << prefix << std::endl;
  int dim = std::stoi(argv[2]);

  cusparseHandle_t handle = 0;
  cusparseMatDescr_t descr = 0;

  int* hostCsrRowPtr = 0;
  int* hostCsrColInd = 0;
  float* hostCsrVal = 0;

  printf("load CSR matrix...\n");
  std::pair<int, int> pair = loadCSRFromFile(prefix, &hostCsrRowPtr, &hostCsrColInd);
  int n = pair.first;
  int nnz = pair.second;
  std::cout << "n=" << n << " nnz=" << nnz << std::endl;
  hostCsrVal = vec2ptr(std::vector<float>(nnz, 1.0));
  
  float alpha = 1.0;
  float beta = 1.0;

  int* csrRowPtr = 0;
  int* csrColInd = 0;
  float* csrVal = 0;

  float* yHostPtr = 0;
  float* y = 0;
  float* zHostPtr = 0;
  float* z = 0;

  printf("gpu memory malloc and memcpy...\n");

  HANDLE_ERROR( cudaMalloc((void**)&csrRowPtr, (n + 1) * sizeof(int)) );
  HANDLE_ERROR( cudaMalloc((void**)&csrColInd, nnz * sizeof(int)) );
  HANDLE_ERROR( cudaMalloc((void**)&csrVal, nnz * sizeof(float)) );

  HANDLE_ERROR( cudaMemcpy(csrRowPtr, hostCsrRowPtr, (size_t)((n + 1) * sizeof(int)), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(csrColInd, hostCsrColInd, (size_t)(nnz * sizeof(int)), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(csrVal, hostCsrVal, (size_t)(nnz * sizeof(float)), cudaMemcpyHostToDevice) );

  HANDLE_CUSPARSE_ERROR( cusparseCreate(&handle) );

  HANDLE_CUSPARSE_ERROR(cusparseCreateMatDescr(&descr) );
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  yHostPtr = randomDenseMatrix(n, dim);
  zHostPtr = (float*)malloc(n * dim * sizeof(float));

  HANDLE_ERROR( cudaMalloc((void**)&y, n * dim * sizeof(float)) );
  HANDLE_ERROR( cudaMalloc((void**)&z, n * dim * sizeof(float)) );

  HANDLE_ERROR( cudaMemcpy(y, yHostPtr, (size_t)(n * dim * sizeof(float)), cudaMemcpyHostToDevice) );

  HANDLE_ERROR( cudaMemset((void*)z, 0, n * dim * sizeof(float)) );

  printf("cusparseScsrmm...\n");

  float time;
  cudaEvent_t start, stop;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  HANDLE_CUSPARSE_ERROR( cusparseScsrmm2(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE, n, dim, n, nnz, &alpha,
      descr, csrVal, csrRowPtr, csrColInd, y, dim, &beta, z, n) );

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));

  printf("csrmm cost time:  %3.10f ms \n", time);

  HANDLE_ERROR( cudaMemcpy(zHostPtr, z, (size_t)(n * dim * sizeof(float)),
                           cudaMemcpyDeviceToHost) );

  HANDLE_CUSPARSE_ERROR(cusparseDestroyMatDescr(descr));
  descr = 0;
  HANDLE_CUSPARSE_ERROR(cusparseDestroy(handle));
  handle = 0;

  CLEANUP("end");

  return 0;
}