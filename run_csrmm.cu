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
#include "cuda_profiler_api.h"
#include "cusparse.h"
#include "load_data.h"
#include "gespmm_csrmm.h"
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
  int dim = std::stoi(argv[2]);
  std::string csrmmImpl(argv[3]);
  int transposeB = std::stoi(argv[4]);
  printf("graph = %s dim = %d csrmmImpl = %s transposeB = %d\n", 
         argv[1], dim, csrmmImpl.c_str(), transposeB);

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
  float beta = 0.0;

  int* csrRowPtr = 0;
  int* csrColInd = 0;
  float* csrVal = 0;

  float* yHostPtr = 0;
  float* y = 0;
  float* zHostPtr = 0;
  float* z = 0;

  cusparseOperation_t transB;
  int ldb;
  if (transposeB == 0) {
    transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    ldb = n;
  } else if (transposeB == 1) {
    transB = CUSPARSE_OPERATION_TRANSPOSE;
    ldb = dim;
  } else {
    assert(false);
  }

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

  int epoch = 10;
  float totalTime = 0;

  for (int i = 0; i < epoch; ++i) {
    float time;
    cudaEvent_t start, stop;

    // cudaProfilerStart();

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    if (csrmmImpl == "cusparseScsrmm") {
      assert(transB == CUSPARSE_OPERATION_NON_TRANSPOSE);
      HANDLE_CUSPARSE_ERROR( cusparseScsrmm(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, dim, n, nnz, &alpha,
        descr, csrVal, csrRowPtr, csrColInd, y, ldb, &beta, z, n) );
    } else if (csrmmImpl == "cusparseScsrmm2") {
      HANDLE_CUSPARSE_ERROR( cusparseScsrmm2(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        transB, n, dim, n, nnz, &alpha,
        descr, csrVal, csrRowPtr, csrColInd, y, ldb, &beta, z, n) );    
    } else if (csrmmImpl == "gespmm") {
      gespmm_csrmm<float>(n, dim, csrRowPtr, csrColInd, csrVal, y, z);
    } else {
      assert(false);
    }

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));

    // cudaProfilerStop();

    printf("csrmm cost time:  %3.10f ms \n", time);
    totalTime += time;
  }

  printf("average csrmm cost time: %3.10f ms\n", totalTime / epoch);

  HANDLE_ERROR( cudaMemcpy(zHostPtr, z, (size_t)(n * dim * sizeof(float)),
                           cudaMemcpyDeviceToHost) );

  HANDLE_CUSPARSE_ERROR(cusparseDestroyMatDescr(descr));
  descr = 0;
  HANDLE_CUSPARSE_ERROR(cusparseDestroy(handle));
  handle = 0;

  CLEANUP("end");

  return 0;
}