#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <random>
#include <fstream>
#include <assert.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include "load_matrix.h"

#define CLEANUP(s) \
do { \
    printf("%s\n", s); \
    if (hostCsrRowPtr) free(hostCsrRowPtr); \
    if (hostCsrColInd) free(hostCsrColInd); \
    if (hostCsrVal) free(hostCsrVal); \
    if (yHostPtr) free(yHostPtr); \
    if (zHostPtr) free(zHostPtr); \
    if (csrRowPtr) cudaFree(csrRowPtr); \
    if (csrColInd) cudaFree(csrColInd); \
    if (csrVal) cudaFree(csrVal); \
    if (y) cudaFree(y); \
    if (y0) cudaFree(y0); \
    if (z) cudaFree(z); \
    if (z0) cudaFree(z0); \
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

int main(int argc, char* argv[]) {
    float p = std::stof(argv[1]);
    int dim = std::stoi(argv[2]);
    printf("p = %f dim = %d\n", p, dim);

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
    float* y0 = 0;
    float* zHostPtr = 0;
    float* z = 0;
    float* z0 = 0;

    printf("generate random CSR matrix\n");

    // nnz = randomCSRMatrix(m, n, p, &hostCsrRowPtr, &hostCsrColInd, &hostCsrVal);
    nnz = readAndFillCSRMatrix(m, n, &hostCsrRowPtr, &hostCsrColInd, &hostCsrVal);

    printf("density of BSR matrix is %f\n", (nnz * 1.0) / (m * n));

    printf("gpu memory malloc and memcpy...\n");

    HANDLE_ERROR( cudaMalloc((void**)&csrRowPtr, (m + 1) * sizeof(int)) );
    HANDLE_ERROR( cudaMalloc((void**)&csrColInd, nnz * sizeof(int)) );
    HANDLE_ERROR( cudaMalloc((void**)&csrVal, nnz * sizeof(float)) );

    HANDLE_ERROR( cudaMemcpy(csrRowPtr, hostCsrRowPtr, (size_t)((m + 1) * sizeof(int)), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(csrColInd, hostCsrColInd, (size_t)(nnz * sizeof(int)), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(csrVal, hostCsrVal, (size_t)(nnz * sizeof(float)), cudaMemcpyHostToDevice) );

    HANDLE_CUSPARSE_ERROR( cusparseCreate(&handle), "CUSPARSE Library initialization failed" );

    HANDLE_CUSPARSE_ERROR( cusparseCreateMatDescr(&descr), "CSR Matrix descriptor initialization failed" );
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    printf("prepare y and z...\n");

    yHostPtr = randomDenseMatrix(n, dim);
    zHostPtr = (float*) malloc(m * dim * sizeof(float));

    HANDLE_ERROR( cudaMalloc((void**)&y, n * dim * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void**)&z, m * dim * sizeof(float)) );

    HANDLE_ERROR( cudaMemcpy(y, yHostPtr, (size_t)(n * dim * sizeof(float)), cudaMemcpyHostToDevice) );

    printf("warm up...\n");
    HANDLE_ERROR( cudaMalloc((void**)&y0, n * dim * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void**)&z0, m * dim * sizeof(float)) );
    HANDLE_ERROR( cudaMemset((void*)y0, 0, n * dim * sizeof(float)) );
    HANDLE_ERROR( cudaMemset((void*)z0, 0, m * dim * sizeof(float)) );
    int warnupRounds = 3;
    for (int i = 0; i < warnupRounds; ++i) {
        HANDLE_CUSPARSE_ERROR( cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                               m, dim, n, nnz, &fone, descr, csrVal, csrRowPtr, csrColInd, y0, dim, &fzero, z0, m),
                                               "warmup cusparseScsrmm2 failed" );
    }

    printf("cusparseScsrmm...\n");
    float totalTime = 0;
    int rounds = 10;
    for (int i = 0; i < rounds; ++i) {
        HANDLE_ERROR( cudaMemset((void*)z, 0, m * dim * sizeof(float)) );
        
        float time;
        cudaEvent_t start, stop;
        HANDLE_ERROR( cudaEventCreate(&start) );
        HANDLE_ERROR( cudaEventCreate(&stop) );
        HANDLE_ERROR( cudaEventRecord(start, 0) );

        HANDLE_CUSPARSE_ERROR( cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                               m, dim, n, nnz, &fone, descr, csrVal, csrRowPtr, csrColInd, y, dim, &fzero, z, m),
                                               "cusparseScsrmm2 failed" );

        HANDLE_ERROR( cudaEventRecord(stop, 0) );
        HANDLE_ERROR( cudaEventSynchronize(stop) );
        HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
        printf("[%d] csrmm cost time:  %3.10f ms \n", i, time);
        totalTime += time;  
    }
    printf("average csrmm cost time: %3.10f ms \n", totalTime / rounds);

    HANDLE_ERROR( cudaMemcpy(zHostPtr, z, (size_t)(m * dim * sizeof(float)), cudaMemcpyDeviceToHost) );

    HANDLE_CUSPARSE_ERROR( cusparseDestroyMatDescr(descr), "Matrix descriptor destruction failed" );
    descr = 0;
    HANDLE_CUSPARSE_ERROR( cusparseDestroy(handle), "CUSPARSE Library release of resources failed" );
    handle = 0;

    CLEANUP("end");

    return 0;
}    