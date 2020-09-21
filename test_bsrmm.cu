#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <assert.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include "cuda_profiler_api.h"
#include "load_data.h"
#include "utility.h"
#include "rocsparse_bsrmm.h"


#define CLEANUP(s) \
do { \
    printf("%s\n", s); \
    if (hostBsrRowPtr) free(hostBsrRowPtr); \
    if (hostBsrColInd) free(hostBsrColInd); \
    if (hostBsrVal) free(hostBsrVal); \
    if (yHostPtr) free(yHostPtr); \
    if (zHostPtr) free(zHostPtr); \
    if (bsrRowPtr) cudaFree(bsrRowPtr); \
    if (bsrColInd) cudaFree(bsrColInd); \
    if (bsrVal) cudaFree(bsrVal); \
    if (y) cudaFree(y); \
    if (z) cudaFree(z); \
    cudaDeviceReset(); \
    fflush(stdout); \
} while (0)

#define HANDLE_ERROR( err ) \
if (!checkError(err, __FILE__, __LINE__)) { \
    CLEANUP("CUDA ERROR"); \
    exit(-1); \
}

#define HANDLE_CUSPARSE_ERROR( err ) \
if (!checkCusparseError(err, __FILE__, __LINE__)) { \
    CLEANUP("CUSPARSE ERROR"); \
    exit(-1); \
}

int main(int argc, char* argv[]) {
    float p = std::stof(argv[1]);
    int blockDim = std::stoi(argv[2]); 
    int dim = std::stoi(argv[3]);
    std::string bsrmmImpl(argv[4]);
    int transposeB = std::stoi(argv[5]);
    printf("p = %f blockDim = %d dim = %d bsrmmImpl = %s transposeB = %d\n", p, blockDim, dim, bsrmmImpl.c_str(), transposeB);

    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;

    int m = 2 << 16;
    int n = m;
    int mb = (m + blockDim - 1) / blockDim;
    int nb = (n + blockDim - 1) / blockDim;
    assert(mb * blockDim == m && nb * blockDim == n);
    int nnzb = 0;
    float fzero = 0.0;
    float fone = 1.0;

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

    printf("generate random BSR matrix\n");

    // nnzb = randomBSRMatrix(mb, nb, blockDim, p, &hostBsrRowPtr, &hostBsrColInd, &hostBsrVal, -1, 1, true);
    nnzb = readAndFillBSRMatrix(mb, nb, blockDim, p, &hostBsrRowPtr, &hostBsrColInd, &hostBsrVal);

    printf("nnzb = %d mb = %d nb = %d\n", nnzb, mb, nb);
    printf("density of BSR matrix is %f\n", (nnzb * 1.0) / ((mb * 1.0) * (nb * 1.0)));

    printf("gpu memory malloc and memcpy...\n");

    HANDLE_ERROR( cudaMalloc((void**)&bsrRowPtr, (mb + 1) * sizeof(int)) );
    HANDLE_ERROR( cudaMalloc((void**)&bsrColInd, nnzb * sizeof(int)) );
    HANDLE_ERROR( cudaMalloc((void**)&bsrVal, nnzb * blockDim * blockDim * sizeof(float)) );

    HANDLE_ERROR( cudaMemcpy(bsrRowPtr, hostBsrRowPtr, (size_t)((mb + 1) * sizeof(int)), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(bsrColInd, hostBsrColInd, (size_t)(nnzb * sizeof(int)), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(bsrVal, hostBsrVal, (size_t)(nnzb * blockDim * blockDim * sizeof(float)), cudaMemcpyHostToDevice) );
    
    HANDLE_CUSPARSE_ERROR( cusparseCreate(&handle) );
    
    HANDLE_CUSPARSE_ERROR( cusparseCreateMatDescr(&descr) );
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    printf("prepare y and z...\n");

    yHostPtr = randomDenseMatrix(n, dim);
    zHostPtr = (float*) malloc(m * dim * sizeof(float));

    HANDLE_ERROR( cudaMalloc((void**)&y, n * dim * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void**)&z, m * dim * sizeof(float)) );

    HANDLE_ERROR( cudaMemcpy(y, yHostPtr, (size_t)(n * dim * sizeof(float)), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemset((void*)z, 0, m * dim * sizeof(float)) );

    printf("cusparseSbsrmm...\n"); 

    float time;
    cudaEvent_t start, stop;
    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    HANDLE_ERROR( cudaEventRecord(start, 0) );

    cudaProfilerStart();

    if (bsrmmImpl == "rocsparse") {
        HANDLE_CUSPARSE_ERROR( rocsparse_bsrmm_template<float>(handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                               transB, mb, dim, nb, nnzb, fone, descr, bsrVal,
                                                               bsrRowPtr, bsrColInd, blockDim, y, ldb, fzero, z, m) ); 
    }  else if (bsrmmImpl == "cusparse"){
        HANDLE_CUSPARSE_ERROR( cusparseSbsrmm(handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              transB, mb, dim, nb, nnzb, &fone, descr, bsrVal,
                                              bsrRowPtr, bsrColInd, blockDim, y, ldb, &fzero, z, m) ); 
    } else {
        assert(false);
    }

    cudaProfilerStop();

    HANDLE_ERROR( cudaEventRecord(stop, 0) );
    HANDLE_ERROR( cudaEventSynchronize(stop) );
    HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
    
    float gflops = (nnzb / 1.0e6) * (blockDim * blockDim * dim) / time; 
    printf("bsrmm cost time: %6.10f ms\nGFLOPs: %6.10f\n", time, gflops);   

    HANDLE_ERROR( cudaMemcpy(zHostPtr, z, (size_t)(m * dim * sizeof(float)), cudaMemcpyDeviceToHost) );

    HANDLE_CUSPARSE_ERROR( cusparseDestroyMatDescr(descr) );
    descr = 0;
    HANDLE_CUSPARSE_ERROR( cusparseDestroy(handle) );
    handle = 0;

    CLEANUP("end");

    return 0;
}