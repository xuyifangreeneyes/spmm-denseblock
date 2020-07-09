#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cuda_runtime.h>
#include "cusparse.h"

#define CLEANUP(s) \
do { \
    printf("%s\n", s); \
    if (yHostPtr) free(yHostPtr); \
    if (zHostPtr) free(zHostPtr); \
    if (hostBsrRowPtr) free(hostBsrRowPtr); \
    if (hostBsrColInd) free(hostBsrColInd); \
    if (hostBsrVal) free(hostBsrVal); \
    if (y) cudaFree(y); \
    if (z) cudaFree(z); \
    if (bsrRowPtr) cudaFree(bsrRowPtr); \
    if (bsrColInd) cudaFree(bsrColInd); \
    if (bsrVal) cudaFree(bsrVal); \
    if (descr) cusparseDestroyMatDescr(descr); \
    if (handle) cusparseDestroy(handle); \
    cudaDeviceReset(); \
    fflush(stdout); \
} while (0) \

template<typename T>
T* vec2ptr(std::vector<T> v) {
    T* ptr = (T*) malloc(v.size() * sizeof(T));
    for (size_t i = 0; i < v.size(); ++i) {
        ptr[i] = v[i];
    }
    return ptr;
}

int main() {
    cudaError_t cudaStat1, cudaStat2, cudaStat3, cudaStat4;
    cusparseStatus_t status;
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;
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
    int blockDim = 2;
    int mb = 2, kb = 3;
    int m = mb * blockDim, k = kb * blockDim;
    int n = 2;
    int nnzb = 4; 
    float fzero = 0.0;
    float fone = 1.0;

    hostBsrRowPtr = vec2ptr<int>({0, 2, 4});
    hostBsrColInd = vec2ptr<int>({0, 2, 1, 2});
    hostBsrVal = vec2ptr<float>({0, 4, 2, 7, 1, 8, 2, 0, 9, 0, 0, 2, 0, 6, 7, 0});
    yHostPtr = vec2ptr<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    zHostPtr = (float*) malloc(m * n * sizeof(float));

    if ((!hostBsrRowPtr) || (!hostBsrColInd) || (!hostBsrVal) || (!yHostPtr) || (!zHostPtr)) {
        CLEANUP("Host malloc failed");
        return 1;
    }

    cudaStat1 = cudaMalloc((void**)&bsrRowPtr, (mb + 1) * sizeof(int));
    cudaStat2 = cudaMalloc((void**)&bsrColInd, nnzb * sizeof(int));
    cudaStat3 = cudaMalloc((void**)&bsrVal, nnzb * blockDim * blockDim * sizeof(float));
    cudaStat4 = cudaMalloc((void**)&y, k * n * sizeof(float));

    if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess || cudaStat4 != cudaSuccess) {
        CLEANUP("Device malloc failed");
        return 1;
    }

    cudaStat1 = cudaMemcpy(bsrRowPtr, hostBsrRowPtr, (size_t)((mb + 1) * sizeof(int)), cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(bsrColInd, hostBsrColInd, (size_t)(nnzb * sizeof(int)), cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(bsrVal, hostBsrVal, (size_t)(nnzb * blockDim * blockDim * sizeof(float)), cudaMemcpyHostToDevice);
    cudaStat4 = cudaMemcpy(y, yHostPtr, (size_t)(k * n * sizeof(float)), cudaMemcpyHostToDevice);

    if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess || cudaStat4 != cudaSuccess) {
        CLEANUP("Memcpy from Host to Device failed");
        return 1;
    }

    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("CUSPARSE Library initialization failed");
        return 1;
    }

    status = cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix descriptor initialization failed");
        return 1;
    }

    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    int devId;
    cudaDeviceProp prop;
    cudaError_t cudaStat;
    cudaStat = cudaGetDevice(&devId);
    if (cudaStat != cudaSuccess) {
        CLEANUP("cudaGetDevice failed");
        printf("Error: cudaStat %d, %s\n", cudaStat, cudaGetErrorString(cudaStat));
        return 1;
    }
    cudaStat = cudaGetDeviceProperties(&prop, devId);
    if (cudaStat != cudaSuccess) {
        CLEANUP("cudaGetDeviceProperties failed");
        printf("Error: cudaStat %d, %s\n", cudaStat, cudaGetErrorString(cudaStat));
        return 1;
    }

    cudaStat1 = cudaMalloc((void**)&z, m * n * sizeof(float));
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Device malloc failed (z)");
        return 1;
    }
    cudaStat1 = cudaMemset((void*)z, 0, m * n * sizeof(float));
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Memset on Device failed");
        return 1;
    }

    status = cusparseSbsrmm(handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE, mb, n, kb, nnzb, &fone, descr, bsrVal,
                            bsrRowPtr, bsrColInd, blockDim, y, k, &fzero, z, m);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("csrmm failed");
        return 1;
    }

    cudaStat1 = cudaMemcpy(zHostPtr, z, (size_t)(m * n * sizeof(float)), cudaMemcpyDeviceToHost);
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Memcpy from Device to Host failed");
        return 1;
    }

    status = cusparseDestroyMatDescr(descr);
    descr = 0;
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Matrix descriptor destruction failed");
        return 1;
    }

    status = cusparseDestroy(handle);
    handle = 0;
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("CUSPARSE Library release of resources failed");
        return 1;
    }

    for (int i = 0; i < m * n; ++i) {
        printf("%f ", zHostPtr[i]);
    }

    printf("\n");
    CLEANUP("end");
}