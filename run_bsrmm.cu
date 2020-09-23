#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <random>
#include <vector>
#include <iostream>
#include "rocsparse_bsrmm.h"
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
    if (bsrRowPtr) cudaFree(bsrRowPtr);     \
    if (bsrColInd) cudaFree(bsrColInd);     \
    if (bsrVal) cudaFree(bsrVal);           \
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
  int blockDim = std::stoi(argv[2]);
  int dim = std::stoi(argv[3]);
  std::string bsrmmImpl(argv[4]);

  cusparseHandle_t handle = 0;
  cusparseMatDescr_t csrDescr = 0, bsrDescr = 0;

  int* hostCsrRowPtr = 0;
  int* hostCsrColInd = 0;
  float* hostCsrVal = 0;

  printf("load CSR matrix...\n");
  std::pair<int, int> pair = loadCSRFromFile(prefix, &hostCsrRowPtr, &hostCsrColInd);
  int n = pair.first;
  int nnz = pair.second;
  std::cout << "n=" << n << " nnz=" << nnz << std::endl;
  hostCsrVal = vec2ptr(std::vector<float>(nnz, 1.0));

  int nb = (n + blockDim - 1) / blockDim;
  int nnzb = 0;
  float alpha = 1.0;
  float beta = 1.0;

  int* csrRowPtr = 0;
  int* csrColInd = 0;
  float* csrVal = 0;

  int* bsrRowPtr = 0;
  int* bsrColInd = 0;
  float* bsrVal = 0;

  float* yHostPtr = 0;
  float* y = 0;
  float* zHostPtr = 0;
  float* z = 0;

  yHostPtr = randomDenseMatrix(nb * blockDim, dim);
  zHostPtr = (float*)malloc(nb * blockDim * dim * sizeof(float));

  HANDLE_ERROR( cudaMalloc((void**)&y, nb * blockDim * dim * sizeof(float)) );
  HANDLE_ERROR( cudaMalloc((void**)&z, nb * blockDim * dim * sizeof(float)) );

  HANDLE_ERROR( cudaMemcpy(y, yHostPtr, (size_t)(nb * blockDim * dim * sizeof(float)), cudaMemcpyHostToDevice) );

  HANDLE_ERROR( cudaMemset((void*)z, 0, nb * blockDim * dim * sizeof(float)) );

  printf("gpu memory malloc and memcpy...\n");

  HANDLE_ERROR( cudaMalloc((void**)&csrRowPtr, (n + 1) * sizeof(int)) );
  HANDLE_ERROR( cudaMalloc((void**)&csrColInd, nnz * sizeof(int)) );
  HANDLE_ERROR( cudaMalloc((void**)&csrVal, nnz * sizeof(float)) );

  HANDLE_ERROR( cudaMemcpy(csrRowPtr, hostCsrRowPtr, (size_t)((n + 1) * sizeof(int)), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(csrColInd, hostCsrColInd, (size_t)(nnz * sizeof(int)), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(csrVal, hostCsrVal, (size_t)(nnz * sizeof(float)), cudaMemcpyHostToDevice) );

  HANDLE_CUSPARSE_ERROR(cusparseCreate(&handle));

  HANDLE_CUSPARSE_ERROR(cusparseCreateMatDescr(&csrDescr));
  cusparseSetMatType(csrDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(csrDescr, CUSPARSE_INDEX_BASE_ZERO);

  HANDLE_CUSPARSE_ERROR(cusparseCreateMatDescr(&bsrDescr));
  cusparseSetMatType(bsrDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(bsrDescr, CUSPARSE_INDEX_BASE_ZERO);

  HANDLE_ERROR( cudaMalloc((void**)&bsrRowPtr, (nb + 1) * sizeof(int)) );

  int base;
  int* nnzTotalDevHostPtr = &nnzb;

  HANDLE_CUSPARSE_ERROR( cusparseXcsr2bsrNnz(
      handle, CUSPARSE_DIRECTION_ROW, n, n, csrDescr,
      csrRowPtr, csrColInd, blockDim, bsrDescr,
      bsrRowPtr, nnzTotalDevHostPtr) );
  if (NULL != nnzTotalDevHostPtr) {
    nnzb = *nnzTotalDevHostPtr;
  } else {
    HANDLE_ERROR( cudaMemcpy(&nnzb, bsrRowPtr + nb, sizeof(int), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(&base, bsrRowPtr, sizeof(int), cudaMemcpyDeviceToHost) );
    nnzb -= base;
  }

  long long numVal =
      (long long)nnzb * (long long)(blockDim * blockDim) * sizeof(float);
  printf("numVal = %lld\n", numVal);
  HANDLE_ERROR( cudaMalloc((void**)&bsrColInd, nnzb * sizeof(int)) );
  HANDLE_ERROR( cudaMalloc((void**)&bsrVal, numVal) );

  HANDLE_CUSPARSE_ERROR( cusparseScsr2bsr(
      handle, CUSPARSE_DIRECTION_ROW, n, n, csrDescr,
      csrVal, csrRowPtr, csrColInd, blockDim, bsrDescr,
      bsrVal, bsrRowPtr, bsrColInd) );

  printf("density:  %3.10f \n", (1.0 * nnzb) / ((nb * 1.0) * (nb * 1.0)));

  printf("cusparseSbsrmm...\n");

  float time;
  cudaEvent_t start, stop;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  if (bsrmmImpl == "rocsparse") {
    HANDLE_CUSPARSE_ERROR( rocsparse_bsrmm_template<float>(
        handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, nb, dim, nb, nnzb, alpha, bsrDescr, bsrVal,
        bsrRowPtr, bsrColInd, blockDim, y, nb * blockDim, beta, z, nb * blockDim) );
  } else if (bsrmmImpl == "cusparse") {
    HANDLE_CUSPARSE_ERROR( cusparseSbsrmm(
        handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, nb, dim, nb, nnzb, &alpha, bsrDescr, bsrVal,
        bsrRowPtr, bsrColInd, blockDim, y, nb * blockDim, &beta, z, nb * blockDim) );
  }

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));

  printf("bsrmm cost time:  %3.10f ms \n", time);

  HANDLE_ERROR( cudaMemcpy(zHostPtr, z, (size_t)(nb * blockDim * dim * sizeof(float)), cudaMemcpyDeviceToHost) );

  HANDLE_CUSPARSE_ERROR(cusparseDestroyMatDescr(csrDescr));
  csrDescr = 0;

  HANDLE_CUSPARSE_ERROR(cusparseDestroyMatDescr(bsrDescr));
  bsrDescr = 0;

  HANDLE_CUSPARSE_ERROR(cusparseDestroy(handle));
  handle = 0;

  CLEANUP("end");

  return 0;
}