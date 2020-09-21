#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <vector>
#include "cusparse.h"

static void handle_error(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(-1);
  }
}

#define HANDLE_ERROR(err) (handle_error(err, __FILE__, __LINE__))

std::mt19937_64 gen(1234);

template <typename T>
T* vec2ptr(std::vector<T> v) {
  T* ptr = (T*)malloc(v.size() * sizeof(T));
  for (size_t i = 0; i < v.size(); ++i) {
    ptr[i] = v[i];
  }
  return ptr;
}

int randomCSRMatrix(int m, int n, float p, int** hostCsrRowPtr,
                    int** hostCsrColInd, float** hostCsrVal, float minVal = -10,
                    float maxVal = 10) {
  std::uniform_real_distribution<float> flip(0, 1), dist(minVal, maxVal);
  *hostCsrRowPtr = (int*)malloc((m + 1) * sizeof(int));
  int cnt = 0;
  (*hostCsrRowPtr)[0] = cnt;
  std::vector<int> indices;
  std::vector<float> vals;
  for (int i = 1; i <= m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (flip(gen) < p) {
        indices.push_back(j);
        vals.push_back(dist(gen));
        ++cnt;
      }
    }
    (*hostCsrRowPtr)[i] = cnt;
  }
  *hostCsrColInd = vec2ptr(std::move(indices));
  *hostCsrVal = vec2ptr(std::move(vals));

  return cnt;
}

float* randomDenseMatrix(int n, int dim, float minVal = -10,
                         float maxVal = 10) {
  std::uniform_real_distribution<float> dist(minVal, maxVal);
  int sz = n * dim;
  float* ptr = (float*)malloc(sz * sizeof(float));
  for (int i = 0; i < sz; ++i) {
    ptr[i] = dist(gen);
  }
  return ptr;
}

#define CLEANUP(s)                          \
  do {                                      \
    printf("%s\n", s);                      \
    if (hostCsrRowPtr) free(hostCsrRowPtr); \
    if (hostCsrColInd) free(hostCsrColInd); \
    if (hostCsrVal) free(hostCsrVal);       \
    if (yHostPtr) free(yHostPtr);           \
    if (z1HostPtr) free(z1HostPtr);         \
    if (z2HostPtr) free(z2HostPtr);         \
    if (csrRowPtr) cudaFree(csrRowPtr);     \
    if (csrColInd) cudaFree(csrColInd);     \
    if (csrVal) cudaFree(csrVal);           \
    if (bsrRowPtr) cudaFree(bsrRowPtr);     \
    if (bsrColInd) cudaFree(bsrColInd);     \
    if (bsrVal) cudaFree(bsrVal);           \
    if (y) cudaFree(y);                     \
    if (z1) cudaFree(z1);                   \
    if (z2) cudaFree(z2);                   \
    cudaDeviceReset();                      \
    fflush(stdout);                         \
  } while (0)

int main(int argc, char* argv[]) {
  float p = std::stof(argv[1]);
  printf("%f\n", p);

  cudaError_t cudaStat1, cudaStat2, cudaStat3;
  cusparseStatus_t status;
  cusparseHandle_t handle = 0;
  cusparseMatDescr_t csrDescr = 0, bsrDescr = 0;

  int m = 1000;
  int n = 1200;
  int nnz = 0;
  int blockDim = 2;
  int mb = (m + blockDim - 1) / blockDim;
  int nb = (n + blockDim - 1) / blockDim;
  int nnzb = 0;
  int dim = 100;
  float fzero = 0.0;
  float fone = 1.0;

  int* hostCsrRowPtr = 0;
  int* hostCsrColInd = 0;
  float* hostCsrVal = 0;
  int* csrRowPtr = 0;
  int* csrColInd = 0;
  float* csrVal = 0;

  int* bsrRowPtr = 0;
  int* bsrColInd = 0;
  float* bsrVal = 0;

  float* yHostPtr = 0;
  float* y = 0;
  float* z1HostPtr = 0;
  float* z1 = 0;
  float* z2HostPtr = 0;
  float* z2 = 0;

  nnz = randomCSRMatrix(m, n, p, &hostCsrRowPtr, &hostCsrColInd, &hostCsrVal);

  cudaStat1 = cudaMalloc((void**)&csrRowPtr, (m + 1) * sizeof(int));
  cudaStat2 = cudaMalloc((void**)&csrColInd, nnz * sizeof(int));
  cudaStat3 = cudaMalloc((void**)&csrVal, nnz * sizeof(float));
  if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess ||
      cudaStat3 != cudaSuccess) {
    CLEANUP("Device malloc failed (CSR matrix)");
    return 1;
  }

  cudaStat1 =
      cudaMemcpy(csrRowPtr, hostCsrRowPtr, (size_t)((m + 1) * sizeof(int)),
                 cudaMemcpyHostToDevice);
  cudaStat2 = cudaMemcpy(csrColInd, hostCsrColInd, (size_t)(nnz * sizeof(int)),
                         cudaMemcpyHostToDevice);
  cudaStat3 = cudaMemcpy(csrVal, hostCsrVal, (size_t)(nnz * sizeof(float)),
                         cudaMemcpyHostToDevice);
  if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess ||
      cudaStat3 != cudaSuccess) {
    CLEANUP("Memcpy from Host to Device failed (CSR matrix)");
    return 1;
  }

  status = cusparseCreate(&handle);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    CLEANUP("CUSPARSE Library initialization failed");
    return 1;
  }

  status = cusparseCreateMatDescr(&csrDescr);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    CLEANUP("CSR Matrix descriptor initialization failed");
    return 1;
  }
  cusparseSetMatType(csrDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(csrDescr, CUSPARSE_INDEX_BASE_ZERO);

  status = cusparseCreateMatDescr(&bsrDescr);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    CLEANUP("BSR Matrix descriptor initialization failed");
    return 1;
  }
  cusparseSetMatType(bsrDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(bsrDescr, CUSPARSE_INDEX_BASE_ZERO);

  cudaStat1 = cudaMalloc((void**)&bsrRowPtr, (mb + 1) * sizeof(int));
  if (cudaStat1 != cudaSuccess) {
    CLEANUP("Device malloc failed (BSR matrix)");
    return 1;
  }
  status = cusparseXcsr2bsrNnz(handle, CUSPARSE_DIRECTION_ROW, m, n, csrDescr,
                               csrRowPtr, csrColInd, blockDim, bsrDescr,
                               bsrRowPtr, &nnzb);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    CLEANUP("cusparseXcsr2bsrNnz failed");
    return 1;
  }
  cudaStat1 = cudaMalloc((void**)&bsrColInd, nnzb * sizeof(int));
  cudaStat2 =
      cudaMalloc((void**)&bsrVal, nnzb * blockDim * blockDim * sizeof(float));
  if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess) {
    CLEANUP("Device malloc failed (BSR matrix)");
    return 1;
  }
  status = cusparseScsr2bsr(handle, CUSPARSE_DIRECTION_ROW, m, n, csrDescr,
                            csrVal, csrRowPtr, csrColInd, blockDim, bsrDescr,
                            bsrVal, bsrRowPtr, bsrColInd);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    CLEANUP("cusparseScsr2bsr failed");
    return 1;
  }

  yHostPtr = randomDenseMatrix(n, dim);
  z1HostPtr = (float*)malloc(m * dim * sizeof(float));
  z2HostPtr = (float*)malloc(m * dim * sizeof(float));

  cudaStat1 = cudaMalloc((void**)&y, n * dim * sizeof(float));
  cudaStat2 = cudaMalloc((void**)&z1, m * dim * sizeof(float));
  cudaStat3 = cudaMalloc((void**)&z2, m * dim * sizeof(float));
  if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess ||
      cudaStat3 != cudaSuccess) {
    CLEANUP("Device malloc failed (dense matrix)");
    return 1;
  }

  cudaStat1 = cudaMemcpy(y, yHostPtr, (size_t)(n * dim * sizeof(float)),
                         cudaMemcpyHostToDevice);
  if (cudaStat1 != cudaSuccess) {
    CLEANUP("Memcpy from Host to Device failed (y)");
    return 1;
  }

  cudaStat1 = cudaMemset((void*)z1, 0, m * dim * sizeof(float));
  cudaStat2 = cudaMemset((void*)z2, 0, m * dim * sizeof(float));
  if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess) {
    CLEANUP("Memset on Device failed (z)");
    return 1;
  }

  float time1, time2;
  cudaEvent_t start1, stop1, start2, stop2;

  HANDLE_ERROR(cudaEventCreate(&start1));
  HANDLE_ERROR(cudaEventCreate(&stop1));
  HANDLE_ERROR(cudaEventRecord(start1, 0));

  status = cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, dim, n,
                          nnz, &fone, csrDescr, csrVal, csrRowPtr, csrColInd, y,
                          n, &fzero, z1, m);

  HANDLE_ERROR(cudaEventRecord(stop1, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop1));
  HANDLE_ERROR(cudaEventElapsedTime(&time1, start1, stop1));

  printf("csrmm cost time:  %3.10f ms \n", time1);

  if (status != CUSPARSE_STATUS_SUCCESS) {
    CLEANUP("csrmm failed");
    return 1;
  }

  HANDLE_ERROR(cudaEventCreate(&start2));
  HANDLE_ERROR(cudaEventCreate(&stop2));
  HANDLE_ERROR(cudaEventRecord(start2, 0));

  status = cusparseSbsrmm(
      handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, mb, dim, nb, nnzb, &fone, bsrDescr,
      bsrVal, bsrRowPtr, bsrColInd, blockDim, y, n, &fzero, z2, m);

  HANDLE_ERROR(cudaEventRecord(stop2, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop2));
  HANDLE_ERROR(cudaEventElapsedTime(&time2, start2, stop2));

  printf("bsrmm cost time:  %3.10f ms \n", time2);

  if (status != CUSPARSE_STATUS_SUCCESS) {
    CLEANUP("bsrmm failed");
    return 1;
  }

  cudaStat1 = cudaMemcpy(z1HostPtr, z1, (size_t)(m * dim * sizeof(float)),
                         cudaMemcpyDeviceToHost);
  cudaStat2 = cudaMemcpy(z2HostPtr, z2, (size_t)(m * dim * sizeof(float)),
                         cudaMemcpyDeviceToHost);
  if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess) {
    CLEANUP("Memcpy from Device to Host failed (z)");
    return 1;
  }

  status = cusparseDestroyMatDescr(csrDescr);
  csrDescr = 0;
  if (status != CUSPARSE_STATUS_SUCCESS) {
    CLEANUP("CSR matrix descriptor destruction failed");
    return 1;
  }

  status = cusparseDestroyMatDescr(bsrDescr);
  bsrDescr = 0;
  if (status != CUSPARSE_STATUS_SUCCESS) {
    CLEANUP("BSR matrix descriptor destruction failed");
    return 1;
  }

  status = cusparseDestroy(handle);
  handle = 0;
  if (status != CUSPARSE_STATUS_SUCCESS) {
    CLEANUP("CUSPARSE Library release of resources failed");
    return 1;
  }

  bool flag = true;
  for (int i = 0; i < m * dim; ++i) {
    float error = fabs(z1HostPtr[i] - z2HostPtr[i]);
    if (error > 0.01) {
      printf("inconsistent result: %d %f", i, error);
      flag = false;
      break;
    }
  }

  if (flag) {
    printf("\nsame result\n");
  } else {
    printf("\ninconsistent result\n");
  }
  CLEANUP("end");
}