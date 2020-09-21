#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "cublas_v2.h"

template <typename T>
T* vec2ptr(std::vector<T> v) {
  T* ptr = (T*)malloc(v.size() * sizeof(T));
  for (int i = 0; i < v.size(); ++i) {
    ptr[i] = v[i];
  }
  return ptr;
}

#define CLEANUP(s)                          \
  do {                                      \
    printf("%s\n", s);                      \
    if (hostBsrRowPtr) free(hostBsrRowPtr); \
    if (hostBsrColInd) free(hostBsrColInd); \
    if (hostBsrVal) free(hostBsrVal);       \
    if (yHostPtr) free(yHostPtr);           \
    if (zHostPtr) free(zHostPtr);           \
    if (bsrRowPtr) cudaFree(bsrRowPtr);     \
    if (bsrColInd) cudaFree(bsrColInd);     \
    if (bsrVal) cudaFree(bsrVal);           \
    if (y) cudaFree(y);                     \
    if (z) cudaFree(z);                     \
    cudaDeviceReset();                      \
    fflush(stdout);                         \
  } while (0)

#define HANDLE_ERROR(err)                                            \
  if (err != cudaSuccess) {                                          \
    printf("error occurred in %s at line %d\n", __FILE__, __LINE__); \
    CLEANUP("");                                                     \
    exit(-1);                                                        \
  }

#define HANDLE_CUBLAS_ERROR(err)                                     \
  if (err != CUBLAS_STATUS_SUCCESS) {                                \
    printf("error occurred in %s at line %d\n", __FILE__, __LINE__); \
    CLEANUP("");                                                     \
    exit(-1);                                                        \
  }

int main() {
  int m = 4;
  int n = 4;
  int dim = 3;
  int bsize = 2;
  int mb = (m + bsize - 1) / bsize;
  int nb = (n + bsize - 1) / bsize;
  assert(mb * bsize == m && nb * bsize == n);
  int nnzb = 3;
  float alpha = 1.0;
  float beta = 1.0;
  // float* hostAlpha = 0;
  // float* hostBeta = 0;
  // float* alpha = 0;
  // float* beta = 0;

  int* hostBsrRowPtr = 0;
  int* hostBsrColInd = 0;
  float* hostBsrVal = 0;
  int* bsrRowPtr = 0;
  int* bsrColInd = 0;
  float* bsrVal = 0;

  float* yHostPtr = 0;
  float* y = 0;
  float* zHostPtr = 0;
  float* z = 0;

  // hostAlpha = (float*) malloc(sizeof(float));
  // hostBeta = (float*) malloc(sizeof(float));
  // *hostAlpha = 1.0;
  // *hostBeta = 1.0;
  // HANDLE_ERROR( cudaMalloc((void**)&alpha, sizeof(float)) );
  // HANDLE_ERROR( cudaMalloc((void**)&beta, sizeof(float)) );
  // HANDLE_ERROR( cudaMemcpy(alpha, hostAlpha, (size_t) sizeof(float),
  // cudaMemcpyHostToDevice) );
  // HANDLE_ERROR( cudaMemcpy(beta, hostBeta, (size_t) sizeof(float),
  // cudaMemcpyHostToDevice) );

  hostBsrRowPtr = vec2ptr<int>({0, 2, 3});
  hostBsrColInd = vec2ptr<int>({0, 1, 1});
  hostBsrVal = vec2ptr<float>({0, 3, 1, 2, 4, 2, 0, 0, 0, 5, 0, 8});
  yHostPtr = vec2ptr<float>({6, 0, 0, 0, 7, 5, 4, 3, 0, 0, 0, 7});
  zHostPtr = (float*)malloc(m * dim * sizeof(float));

  HANDLE_ERROR(cudaMalloc((void**)&bsrRowPtr, (mb + 1) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&bsrColInd, nnzb * sizeof(int)));
  HANDLE_ERROR(
      cudaMalloc((void**)&bsrVal, nnzb * bsize * bsize * sizeof(float)));

  HANDLE_ERROR(cudaMemcpy(bsrRowPtr, hostBsrRowPtr,
                          (size_t)((mb + 1) * sizeof(int)),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(bsrColInd, hostBsrColInd,
                          (size_t)(nnzb * sizeof(int)),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(bsrVal, hostBsrVal,
                          (size_t)(nnzb * bsize * bsize * sizeof(float)),
                          cudaMemcpyHostToDevice));

  HANDLE_ERROR(cudaMalloc((void**)&y, n * dim * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&z, m * dim * sizeof(float)));

  HANDLE_ERROR(cudaMemcpy(y, yHostPtr, (size_t)(n * dim * sizeof(float)),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemset((void*)z, 0, m * dim * sizeof(float)));

  cublasHandle_t handle;
  HANDLE_CUBLAS_ERROR(cublasCreate(&handle));

  cudaStream_t* streams = (cudaStream_t*)malloc(mb * sizeof(cudaStream_t));
  for (int i = 0; i < mb; ++i) {
    HANDLE_ERROR(cudaStreamCreate(&streams[i]));
  }

  for (int i = 0; i < mb; ++i) {
    HANDLE_CUBLAS_ERROR(cublasSetStream(handle, streams[i]));

    int start = hostBsrRowPtr[i], end = hostBsrRowPtr[i + 1];
    for (int j = start; j < end; ++j) {
      int idx = hostBsrColInd[j];
      float* A = bsrVal + j * bsize * bsize;
      float* B = y + idx * bsize * dim;
      float* C = z + i * bsize;
      HANDLE_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, bsize,
                                      dim, bsize, &alpha, A, bsize, B, dim,
                                      &beta, C, m));
    }
  }

  HANDLE_ERROR(cudaMemcpy(zHostPtr, z, (size_t)(m * dim * sizeof(float)),
                          cudaMemcpyDeviceToHost));

  for (int i = 0; i < m * dim; ++i) {
    printf("%f ", zHostPtr[i]);
  }
  printf("\n");

  CLEANUP("end");

  return 0;
}
