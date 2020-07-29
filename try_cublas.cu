#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cuda_runtime.h>
#include "cublas_v2.h"

template<typename T>
T* vec2ptr(std::vector<T> v) {
    T* ptr = (T*) malloc(v.size() * sizeof(T));
    for (int i = 0; i < v.size(); ++i) {
        ptr[i] = v[i];
    }
    return ptr;
}

#define CLEANUP(s) \
do { \
    printf("%s\n", s); \
    if (hostA) free(hostA); \
    if (hostB) free(hostB); \
    if (hostC) free(hostC); \
    if (A) cudaFree(A); \
    if (B) cudaFree(B); \
    if (C) cudaFree(C); \
    cudaDeviceReset(); \
    fflush(stdout); \
} while (0)

#define HANDLE_ERROR( err ) \
if (err != cudaSuccess) { \
    printf("%s in %s at linedd %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    CLEANUP("cuda error occurred"); \
    exit(-1); \
}

#define HANDLE_CUBLAS_ERROR( err, s ) \
if (err != CUBLAS_STATUS_SUCCESS) { \
    CLEANUP(s); \
    exit(-1); \
}

int main() {
    int m = 2, k = 3, n = 4;
    float* hostA = 0;
    float* hostB = 0;
    float* hostC = 0;
    float* A = 0;
    float* B = 0;
    float* C = 0;
    float alpha = 1.0;
    float beta = 0.0;

    hostA = vec2ptr<float>({1, 4, 2, 5, 3, 6});
    hostB = vec2ptr<float>({1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0});
    hostC = vec2ptr<float>({0, 0, 0, 0, 0, 0, 0, 0});

    HANDLE_ERROR( cudaMalloc((void**)&A, m * k * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void**)&B, k * n * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void**)&C, m * n * sizeof(float)) );

    cublasHandle_t handle;
    HANDLE_CUBLAS_ERROR( cublasCreate(&handle), "cublasCreate failed" );

    // HANDLE_CUBLAS_ERROR( cublasSetMatrix(m, k, sizeof(float), hostA, m, A, m), "cublasSetMatrix1 failed" );    
    // HANDLE_CUBLAS_ERROR( cublasSetMatrix(k, n, sizeof(float), hostB, k, B, k), "cublasSetMatrix2 failed" );
    // HANDLE_CUBLAS_ERROR( cublasSetMatrix(m, n, sizeof(float), hostC, m, C, m), "cublasSetMatrix3 failed" );

    HANDLE_ERROR( cudaMemcpy(A, hostA, (size_t) (m * k * sizeof(float)), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(B, hostB, (size_t) (k * n * sizeof(float)), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemset((void*)C, 0, m * n * sizeof(float)) );

    HANDLE_CUBLAS_ERROR( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m), "cublasSgemm failed" );

    // cublasGetMatrix(m, n, sizeof(float), C, m, hostC, m);
    HANDLE_ERROR( cudaMemcpy(hostC, C, (size_t) (m * n * sizeof(float)), cudaMemcpyDeviceToHost) );

    for (int i = 0; i < m * n; ++i) {
        printf("%f ", hostC[i]);
    }
    printf("\n");

    cublasDestroy(handle);

    CLEANUP("end");

    return 0;
}