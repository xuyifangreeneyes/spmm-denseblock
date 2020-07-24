#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <random>
#include <fstream>
#include <cuda_runtime.h>
#include <assert.h>
#include "cusparse.h"

std::mt19937_64 gen(1234);

template<typename T>
T* vec2ptr(std::vector<T> v) {
    T* ptr = (T*) malloc(v.size() * sizeof(T));
    for (size_t i = 0; i < v.size(); ++i) {
        ptr[i] = v[i];
    }
    return ptr;
}

int randomCSRMatrix(int m, int n, float p, int** hostCsrRowPtr, int** hostCsrColInd, float** hostCsrVal, float minVal=-1, float maxVal=1) {
    std::uniform_real_distribution<float> flip(0, 1), dist(minVal, maxVal);
    *hostCsrRowPtr = (int*) malloc((m + 1) * sizeof(int));
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
        // if (i % 20 == 0) {
        //     printf("i = %d\n", i);
        // }
    }
    *hostCsrColInd = vec2ptr(std::move(indices));
    *hostCsrVal = vec2ptr(std::move(vals));

    return cnt;
}

float* constDenseMatrix(int n, int dim) {
    int sz = n * dim;
    float* ptr = (float*) malloc(sz * sizeof(float));
    for (int i = 0; i < sz; ++i) {
        if (i % 2 == 0) {
            ptr[i] = 0.5;
        } else {
            ptr[i] = -0.5;
        }
    }
    return ptr;
}

#define CLEANUP(s) \
do { \
    printf("%s\n", s); \
    if (hostCsrRowPtr) free(hostCsrRowPtr); \
    if (hostCsrColInd) free(hostCsrColInd); \
    if (hostCsrVal) free(hostCsrVal); \
    if (yHostPtr) free(yHostPtr); \
    if (z1HostPtr) free(z1HostPtr); \
    if (z2HostPtr) free(z2HostPtr); \
    if (csrRowPtr) cudaFree(csrRowPtr); \
    if (csrColInd) cudaFree(csrColInd); \
    if (csrVal) cudaFree(csrVal); \
    if (bsrRowPtr) cudaFree(bsrRowPtr); \
    if (bsrColInd) cudaFree(bsrColInd); \
    if (bsrVal) cudaFree(bsrVal); \
    if (y) cudaFree(y); \
    if (z1) cudaFree(z1); \
    if (z2) cudaFree(z2); \
    cudaDeviceReset(); \
    fflush(stdout); \
} while (0) 

#define HANDLE_ERROR( err ) \
if (err != cudaSuccess) { \
    printf("%s in %s at linedd %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    CLEANUP("cuda error occurred"); \
    exit(-1); \
}

#define HANDLE_CUSPARSE_ERROR( err, s ) \
if (err != CUSPARSE_STATUS_SUCCESS) { \
    CLEANUP(s); \
    exit(-1); \
}

int main() {
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t csrDescr = 0, bsrDescr = 0;

    float p = 0.01;

    int m = 2 << 14;
    int n = m;
    int nnz = 0;
    int blockDim = 4;
    int mb = (m + blockDim - 1) / blockDim;
    int nb = (n + blockDim - 1) / blockDim;
    assert(mb * blockDim == m && nb * blockDim == n);
    int nnzb = 0;
    int dim = 64;
    float fzero = 0.0;
    float fone = 1.0;
    float eps = 1e-4;

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

    printf("generate random CSR matrix...\n");

    nnz = randomCSRMatrix(m, n, p, &hostCsrRowPtr, &hostCsrColInd, &hostCsrVal);

    printf("move CSR matrix from CPU to GPU...\n");

    HANDLE_ERROR( cudaMalloc((void**)&csrRowPtr, (m + 1) * sizeof(int)) );
    HANDLE_ERROR( cudaMalloc((void**)&csrColInd, nnz * sizeof(int)) );
    HANDLE_ERROR( cudaMalloc((void**)&csrVal, nnz * sizeof(float)) );

    HANDLE_ERROR( cudaMemcpy(csrRowPtr, hostCsrRowPtr, (size_t)((m + 1) * sizeof(int)), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(csrColInd, hostCsrColInd, (size_t)(nnz * sizeof(int)), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(csrVal, hostCsrVal, (size_t)(nnz * sizeof(float)), cudaMemcpyHostToDevice) );

    HANDLE_CUSPARSE_ERROR( cusparseCreate(&handle), "CUSPARSE Library initialization failed" );
    
    HANDLE_CUSPARSE_ERROR( cusparseCreateMatDescr(&csrDescr), "CSR Matrix descriptor initialization failed" );
    cusparseSetMatType(csrDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(csrDescr, CUSPARSE_INDEX_BASE_ZERO);
    
    HANDLE_CUSPARSE_ERROR( cusparseCreateMatDescr(&bsrDescr), "BSR Matrix descriptor initialization failed" );
    cusparseSetMatType(bsrDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(bsrDescr, CUSPARSE_INDEX_BASE_ZERO);

    printf("convert CSR matrix to BSR matrix...\n");

    HANDLE_ERROR( cudaMalloc((void**)&bsrRowPtr, (mb + 1) * sizeof(int)) );
    int base;
    int *nnzTotalDevHostPtr = &nnzb;
    HANDLE_CUSPARSE_ERROR( cusparseXcsr2bsrNnz(handle, CUSPARSE_DIRECTION_ROW, m, n, csrDescr, csrRowPtr, 
                                               csrColInd, blockDim, bsrDescr, bsrRowPtr, nnzTotalDevHostPtr),
                           "cusparseXcsr2bsrNnz failed" ); 
    if (NULL != nnzTotalDevHostPtr) {
        nnzb = *nnzTotalDevHostPtr;
    } else {
        HANDLE_ERROR( cudaMemcpy(&nnzb, bsrRowPtr + mb, sizeof(int), cudaMemcpyDeviceToHost) );
        HANDLE_ERROR( cudaMemcpy(&base, bsrRowPtr, sizeof(int), cudaMemcpyDeviceToHost) );
        nnzb -= base;
    }
    HANDLE_ERROR( cudaMalloc((void**)&bsrColInd, nnzb * sizeof(int)) );
    HANDLE_ERROR( cudaMalloc((void**)&bsrVal, nnzb * blockDim * blockDim * sizeof(float)) );
    HANDLE_CUSPARSE_ERROR( cusparseScsr2bsr(handle, CUSPARSE_DIRECTION_ROW, m, n, csrDescr, csrVal, csrRowPtr, csrColInd,
                                            blockDim, bsrDescr, bsrVal, bsrRowPtr, bsrColInd),
                           "cusparseScsr2bsr failed");
    
    printf("prepare y and z...\n");
    yHostPtr = constDenseMatrix(n, dim);
    z1HostPtr = (float*) malloc(m * dim * sizeof(float));
    z2HostPtr = (float*) malloc(m * dim * sizeof(float));

    HANDLE_ERROR( cudaMalloc((void**)&y, n * dim * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void**)&z1, m * dim * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void**)&z2, m * dim * sizeof(float)) );

    HANDLE_ERROR( cudaMemcpy(y, yHostPtr, (size_t)(n * dim * sizeof(float)), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemset((void*)z1, 0, m * dim * sizeof(float)) );
    HANDLE_ERROR( cudaMemset((void*)z2, 0, m * dim * sizeof(float)) );

    printf("cusparseScsrmm...\n");

    HANDLE_CUSPARSE_ERROR( cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                           m, dim, n, nnz, &fone, csrDescr, csrVal, csrRowPtr, csrColInd, y, dim, &fzero, z1, m),
                           "cusparseScsrmm2 failed" );
    
    HANDLE_ERROR( cudaMemcpy(z1HostPtr, z1, (size_t)(m * dim * sizeof(float)), cudaMemcpyDeviceToHost) );

    printf("cusparseSbsrmm...\n");

    HANDLE_CUSPARSE_ERROR( cusparseSbsrmm(handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          CUSPARSE_OPERATION_TRANSPOSE, mb, dim, nb, nnzb, &fone, bsrDescr, bsrVal,
                                          bsrRowPtr, bsrColInd, blockDim, y, dim, &fzero, z2, m),
                           "cusparseSbsrmm failed" );
    
    HANDLE_ERROR( cudaMemcpy(z2HostPtr, z2, (size_t)(m * dim * sizeof(float)), cudaMemcpyDeviceToHost) );

    int sz = m * dim;
    for (int i = 0; i < sz; ++i) {
        float err = fabs(z1HostPtr[i] - z2HostPtr[i]);
        if (err >= eps) {
            printf("error of z[%d] = %f\n", i, err);
            CLEANUP("reuslt does not match");
            return 1;
        }
        // if (i % 1000 == 0) {
        //     printf("z1[%d] = %f, z2[%d] = %f\n", i, z1HostPtr[i], i, z2HostPtr[i]);
        // }
    }
    printf("z1 and z2 are all the same\n");

    HANDLE_CUSPARSE_ERROR( cusparseDestroyMatDescr(csrDescr), "CSR matrix descriptor destruction failed");
    csrDescr = 0;
    HANDLE_CUSPARSE_ERROR( cusparseDestroyMatDescr(bsrDescr), "BSR matrix descriptor destruction failed");
    bsrDescr = 0;
    HANDLE_CUSPARSE_ERROR( cusparseDestroy(handle), "CUSPARSE Library release of resources failed" );
    handle = 0;

    CLEANUP("end");

    return 0;
}