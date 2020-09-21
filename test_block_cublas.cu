#include <assert.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include "cublas_v2.h"

std::mt19937_64 gen(1234);

template <typename T>
T* vec2ptr(std::vector<T> v) {
  T* ptr = (T*)malloc(v.size() * sizeof(T));
  for (int i = 0; i < v.size(); ++i) {
    ptr[i] = v[i];
  }
  return ptr;
}

int randomBSRMatrix(int mb, int nb, int bsize, float p, int** hostBsrRowPtr,
                    int** hostBsrColInd, float** hostBsrVal, float minVal = -1,
                    float maxVal = 1) {
  std::uniform_real_distribution<float> flip(0, 1), dist(minVal, maxVal);
  int blockNum = bsize * bsize;
  *hostBsrRowPtr = (int*)malloc((mb + 1) * sizeof(int));
  int cnt = 0;
  (*hostBsrRowPtr)[0] = cnt;
  std::vector<int> indices;
  std::vector<float> vals;
  for (int i = 1; i <= mb; ++i) {
    for (int j = 0; j < nb; ++j) {
      if (flip(gen) < p) {
        indices.push_back(j);
        for (int k = 0; k < blockNum; ++k) {
          vals.push_back(dist(gen));
        }
        ++cnt;
      }
    }
    (*hostBsrRowPtr)[i] = cnt;
    // if (i % 1000 == 0) {
    //     printf("i = %d\n", i);
    // }
  }
  *hostBsrColInd = vec2ptr(std::move(indices));
  *hostBsrVal = vec2ptr(std::move(vals));

  // Generating random BSR matrix may be time-consuming, so we record it for
  // next time use.
  // std::string bd = std::to_string(bsize);
  // std::fstream s1("bsr_" + bd + "_indptr.txt", std::ios::out |
  // std::ios::trunc);
  // std::fstream s2("bsr_" + bd + "_indices.txt", std::ios::out |
  // std::ios::trunc);

  // s1 << mb + 1 << std::endl;
  // for (int i = 0; i <= mb; ++i) {
  //     s1 << (*hostBsrRowPtr)[i] << " ";
  // }
  // s1 << std::endl;

  // s2 << cnt << std::endl;
  // for (int i = 0; i < cnt; ++i) {
  //     s2 << (*hostBsrColInd)[i] << " ";
  // }
  // s2 << std::endl;

  return cnt;
}

int readAndFillBSRMatrix(int mb, int nb, int bsize, int** hostBsrRowPtr,
                         int** hostBsrColInd, float** hostBsrVal,
                         float minVal = -1, float maxVal = 1) {
  std::string bd = std::to_string(bsize);
  std::fstream s1("bsr_" + bd + "_indptr.txt", std::ios::in);
  std::fstream s2("bsr_" + bd + "_indices.txt", std::ios::in);

  int xx;
  s1 >> xx;
  assert(mb + 1 == xx);
  *hostBsrRowPtr = (int*)malloc((mb + 1) * sizeof(int));
  for (int i = 0; i <= mb; ++i) {
    s1 >> (*hostBsrRowPtr)[i];
  }

  int nnzb;
  s2 >> nnzb;
  *hostBsrColInd = (int*)malloc(nnzb * sizeof(int));
  for (int i = 0; i < nnzb; ++i) {
    s2 >> (*hostBsrColInd)[i];
  }

  int num = nnzb * bsize * bsize;
  *hostBsrVal = (float*)malloc(num * sizeof(float));
  std::uniform_real_distribution<float> dist(minVal, maxVal);
  for (int i = 0; i < num; ++i) {
    (*hostBsrVal)[i] = dist(gen);
  }

  return nnzb;
}

float* randomDenseMatrix(int n, int dim, float minVal = -1, float maxVal = 1) {
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

int main(int argc, char* argv[]) {
  float p = std::stof(argv[1]);
  int bsize = std::stoi(argv[2]);
  int dim = std::stoi(argv[3]);
  printf("p = %f bsize = %d dim = %d\n", p, bsize, dim);

  int m = 2 << 16;
  int n = m;
  int mb = (m + bsize - 1) / bsize;
  int nb = (n + bsize - 1) / bsize;
  assert(mb * bsize == m && nb * bsize == n);
  int nnzb = 0;
  // float alpha = 1.0;
  // float beta = 1.0;

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

  float* alpha = (float*)malloc(sizeof(float));
  *alpha = 1.0;
  float* beta = (float*)malloc(sizeof(float));
  *beta = 1.0;

  float* deviceAlpha = 0;
  float* deviceBeta = 0;
  HANDLE_ERROR(cudaMalloc((void**)&deviceAlpha, sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&deviceBeta, sizeof(float)));
  HANDLE_ERROR(cudaMemcpy((void*)deviceAlpha, (void*)alpha, sizeof(float),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy((void*)deviceBeta, (void*)beta, sizeof(float),
                          cudaMemcpyHostToDevice));

  printf("generate random BSR matrix\n");

  nnzb = randomBSRMatrix(mb, nb, bsize, p, &hostBsrRowPtr, &hostBsrColInd,
                         &hostBsrVal);
  // nnzb = readAndFillBSRMatrix(mb, nb, bsize, &hostBsrRowPtr, &hostBsrColInd,
  // &hostBsrVal);

  printf("nnzb = %d, density of BSR matrix is %f\n", nnzb,
         (nnzb * 1.0) / mb / nb);

  printf("gpu memory malloc and memcpy...\n");

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

  printf("prepare y and z...\n");

  yHostPtr = randomDenseMatrix(n, dim);
  zHostPtr = (float*)malloc(m * dim * sizeof(float));

  HANDLE_ERROR(cudaMalloc((void**)&y, n * dim * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&z, m * dim * sizeof(float)));

  HANDLE_ERROR(cudaMemcpy(y, yHostPtr, (size_t)(n * dim * sizeof(float)),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemset((void*)z, 0, m * dim * sizeof(float)));

  printf("block cublas...\n");

  int stream_num = 32;
  cudaStream_t* streams =
      (cudaStream_t*)malloc(stream_num * sizeof(cudaStream_t));
  cublasHandle_t* handles =
      (cublasHandle_t*)malloc(stream_num * sizeof(cublasHandle_t));
  for (int i = 0; i < stream_num; ++i) {
    HANDLE_CUBLAS_ERROR(cublasCreate(&handles[i]));
    HANDLE_ERROR(cudaStreamCreate(&streams[i]));
    HANDLE_CUBLAS_ERROR(cublasSetStream(handles[i], streams[i]));
  }

  float time;
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  // auto streamJob = [=](int i) {
  //     HANDLE_CUBLAS_ERROR( cublasSetStream(handle, streams[i]) );

  //     int start = hostBsrRowPtr[i], end = hostBsrRowPtr[i + 1];
  //     for (int j = start; j < end; ++j) {
  //         int idx = hostBsrColInd[j];
  //         float* A = bsrVal + j * bsize * bsize;
  //         float* B = y + idx * bsize * dim;
  //         float* C = z + i * bsize;
  //         HANDLE_CUBLAS_ERROR( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
  //                                          bsize, dim, bsize, &alpha,
  //                                          A, bsize, B, dim, &beta,
  //                                          C, m) );
  //     }
  // };

  // std::vector<std::thread> threads;
  // for (int i = 0; i < mb; ++i) {
  //     threads.emplace_back(streamJob, i);
  // }

  // for (auto&& t : threads) {
  //     t.join();
  // }

  for (int i = 0; i < mb; ++i) {
    int start = hostBsrRowPtr[i], end = hostBsrRowPtr[i + 1];
    for (int j = start; j < end; ++j) {
      int idx = hostBsrColInd[j];
      float* A = bsrVal + j * bsize * bsize;
      float* B = y + idx * bsize * dim;
      float* C = z + i * bsize;
      HANDLE_CUBLAS_ERROR(cublasSgemm(
          handles[i % stream_num], CUBLAS_OP_N, CUBLAS_OP_T, bsize, dim, bsize,
          deviceAlpha, A, bsize, B, dim, deviceBeta, C, m));
    }
  }

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
  printf("bsrmm cost time:  %3.10f ms \n", time);

  HANDLE_ERROR(cudaMemcpy(zHostPtr, z, (size_t)(m * dim * sizeof(float)),
                          cudaMemcpyDeviceToHost));

  for (int i = 0; i < stream_num; ++i) {
    HANDLE_ERROR(cudaStreamDestroy(streams[i]));
    HANDLE_CUBLAS_ERROR(cublasDestroy(handles[i]));
  }
  free(streams);
  free(handles);

  CLEANUP("end");

  return 0;
}