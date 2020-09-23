#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <random>
#include <vector>
#include "cuda_profiler_api.h"
#include "cusparse.h"
#include "gespmm_csrmm.h"
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
  float p = std::stof(argv[1]);
  int dim = std::stoi(argv[2]);
  std::string csrmmImpl(argv[3]);
  printf("p = %f dim = %d csrmmImpl = %s\n", p, dim, csrmmImpl.c_str());

  cusparseHandle_t handle = 0;
  cusparseMatDescr_t descr = 0;

  int m = 2 << 16;
  int n = m;
  int nnz = 0;
  float fzero = 0.0;
  float fone = 1.0;

  int* hostCsrRowPtr = 0;
  int* hostCsrColInd = 0;
  float* hostCsrVal = 0;
  int* csrRowPtr = 0;
  int* csrColInd = 0;
  float* csrVal = 0;

  float* yHostPtr = 0;
  float* y = 0;
  float* zHostPtr = 0;
  float* z = 0;

  printf("generate random CSR matrix\n");

  // nnz = randomCSRMatrix(m, n, p, &hostCsrRowPtr, &hostCsrColInd, &hostCsrVal,
  // -1, 1, true);
  nnz = readAndFillCSRMatrix(m, n, p, &hostCsrRowPtr, &hostCsrColInd,
                             &hostCsrVal);

  printf("density of CSR matrix is %f\n", ((nnz * 1.0) / m) / n);

  printf("gpu memory malloc and memcpy...\n");

  HANDLE_ERROR(cudaMalloc((void**)&csrRowPtr, (m + 1) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&csrColInd, nnz * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&csrVal, nnz * sizeof(float)));

  HANDLE_ERROR(cudaMemcpy(csrRowPtr, hostCsrRowPtr,
                          (size_t)((m + 1) * sizeof(int)),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(csrColInd, hostCsrColInd, (size_t)(nnz * sizeof(int)),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(csrVal, hostCsrVal, (size_t)(nnz * sizeof(float)),
                          cudaMemcpyHostToDevice));

  HANDLE_CUSPARSE_ERROR(cusparseCreate(&handle));

  HANDLE_CUSPARSE_ERROR(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  printf("prepare y and z...\n");

  yHostPtr = randomDenseMatrix(n, dim);
  zHostPtr = (float*)malloc(m * dim * sizeof(float));

  HANDLE_ERROR(cudaMalloc((void**)&y, n * dim * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&z, m * dim * sizeof(float)));

  HANDLE_ERROR(cudaMemcpy(y, yHostPtr, (size_t)(n * dim * sizeof(float)),
                          cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaMemset((void*)z, 0, m * dim * sizeof(float)));

  printf("cusparseScsrmm...\n");

  float time;
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  cudaProfilerStart();

  if (csrmmImpl == "gespmm") {
    gespmm_csrmm<float>(m, dim, csrRowPtr, csrColInd, csrVal, y, z);
  } else if (csrmmImpl == "cusparse") {
    HANDLE_CUSPARSE_ERROR(cusparseScsrmm2(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        m, dim, n, nnz, &fone, descr, csrVal, csrRowPtr, csrColInd, y, dim,
        &fzero, z, m));
  } else {
    assert(false);
  }

  cudaProfilerStop();

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));

  float gflops = (nnz / 1.0e6) * dim / time;
  printf("csrmm cost time: %6.10f ms\nGFLOPs: %6.10f\n", time, gflops);

  HANDLE_ERROR(cudaMemcpy(zHostPtr, z, (size_t)(m * dim * sizeof(float)),
                          cudaMemcpyDeviceToHost));

  HANDLE_CUSPARSE_ERROR(cusparseDestroyMatDescr(descr));
  descr = 0;
  HANDLE_CUSPARSE_ERROR(cusparseDestroy(handle));
  handle = 0;

  CLEANUP("end");

  return 0;
}