#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <random>
#include <fstream>
#include <cuda_runtime.h>
#include <assert.h>
#include "cusparse.h"

static void handle_error(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at linedd %d\n", cudaGetErrorString(err), file, line);
        exit(-1);
    }
}

#define HANDLE_ERROR( err ) (handle_error( err, __FILE__, __LINE__ ))

std::mt19937_64 gen(1234);

void readCSRMatrix(int m, int n, int nnz, int** hostCsrRowPtr, int** hostCsrColInd, float** hostCsrVal) {
    *hostCsrRowPtr = (int*) malloc((m + 1) * sizeof(int));
    *hostCsrColInd = (int*) malloc(nnz * sizeof(int));
    *hostCsrVal = (float*) malloc(nnz * sizeof(float));
    
    std::fstream s1("collab_ndmetis_indptr.txt");
    std::fstream s2("collab_ndmetis_indices.txt");
    int _m_1;
    s1 >> _m_1;
    printf("m = %d _m_1 = %d\n", m, _m_1);
    assert(m + 1 == _m_1);
    for (int i = 0; i <= m; ++i) {
        int x;
        s1 >> x;
        (*hostCsrRowPtr)[i] = x;
    }

    printf("kkkkk2\n");

    int _nnz;
    s2 >> _nnz;
    assert(nnz == _nnz);
    for (int i = 0; i < nnz; ++i) {
        int x;
        s2 >> x;
        (*hostCsrColInd)[i] = x;
        (*hostCsrVal)[i] = 1.0;
    }
}

float* randomDenseMatrix(int n, int dim, float minVal=-10, float maxVal=10) {
    std::uniform_real_distribution<float> dist(minVal, maxVal);
    int sz = n * dim;
    float* ptr = (float*) malloc(sz * sizeof(float));
    for (int i = 0; i < sz; ++i) {
        ptr[i] = dist(gen);
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
    if (zHostPtr) free(zHostPtr); \
    if (csrRowPtr) cudaFree(csrRowPtr); \
    if (csrColInd) cudaFree(csrColInd); \
    if (csrVal) cudaFree(csrVal); \
    if (bsrRowPtr) cudaFree(bsrRowPtr); \
    if (bsrColInd) cudaFree(bsrColInd); \
    if (bsrVal) cudaFree(bsrVal); \
    if (y) cudaFree(y); \
    if (z) cudaFree(z); \
    cudaDeviceReset(); \
    fflush(stdout); \
} while (0) \

int main() {
    cudaError_t cudaStat1, cudaStat2, cudaStat3;
    cusparseStatus_t status;
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t csrDescr = 0, bsrDescr = 0;

    int m = 235868;
    int n = m;    
    int nnz = 2358104;
    int blockDim = 16;
    int mb = (m + blockDim - 1) / blockDim;
    int nb = (n + blockDim - 1) / blockDim;
    int nnzb = 0;
    int dim = 64;
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
    float* zHostPtr = 0;
    float* z = 0;

    yHostPtr = randomDenseMatrix(nb * blockDim, dim);
    zHostPtr = (float*) malloc(mb * blockDim * dim * sizeof(float));

    cudaStat1 = cudaMalloc((void**)&y, nb * blockDim * dim * sizeof(float));
    cudaStat2 = cudaMalloc((void**)&z, mb * blockDim * dim * sizeof(float));
    if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess) {
        CLEANUP("Device malloc failed (dense matrix)");
        return 1;
    }

    cudaStat1 = cudaMemcpy(y, yHostPtr, (size_t)(nb * blockDim * dim * sizeof(float)), cudaMemcpyHostToDevice);
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Memcpy from Host to Device failed (y)");
        return 1;
    }

    cudaStat1 = cudaMemset((void*)z, 0, mb * blockDim * dim * sizeof(float));
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Memset on Device failed (z)");
        return 1;
    }

    printf("read CSR matrix...\n");

    readCSRMatrix(m, n, nnz, &hostCsrRowPtr, &hostCsrColInd, &hostCsrVal);

    printf("gpu memory malloc and memcpy...\n");

    cudaStat1 = cudaMalloc((void**)&csrRowPtr, (m + 1) * sizeof(int));
    printf("kkkkkn1\n");
    
    cudaStat2 = cudaMalloc((void**)&csrColInd, nnz * sizeof(int));
    
    printf("kkkkkn2\n");

    cudaStat3 = cudaMalloc((void**)&csrVal, nnz * sizeof(float));
    if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess) {
        CLEANUP("Device malloc failed (CSR matrix)");
        return 1;
    }

    printf("kkkkk3\n");


    cudaStat1 = cudaMemcpy(csrRowPtr, hostCsrRowPtr, (size_t)((m + 1) * sizeof(int)), cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(csrColInd, hostCsrColInd, (size_t)(nnz * sizeof(int)), cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(csrVal, hostCsrVal, (size_t)(nnz * sizeof(float)), cudaMemcpyHostToDevice);
    if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess) {
        CLEANUP("Memcpy from Host to Device failed (CSR matrix)");
        return 1;
    }
    
    printf("kkkkk4\n");

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

    printf("kkkkk5\n");


    cudaStat1 = cudaMalloc((void**)&bsrRowPtr, (mb + 1) * sizeof(int));
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Device malloc failed1 (BSR matrix)");
        printf("%s\n", cudaGetErrorString(cudaStat1));
        return 1;
    }
    printf("kkkkk5.5\n");

    int base;
    int *nnzTotalDevHostPtr = &nnzb;

    status = cusparseXcsr2bsrNnz(handle, CUSPARSE_DIRECTION_ROW, m, n, csrDescr, csrRowPtr, 
                                 csrColInd, blockDim, bsrDescr, bsrRowPtr, nnzTotalDevHostPtr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("cusparseXcsr2bsrNnz failed");
        return 1;
    }
    printf("nnz = %d", nnz);
    if (NULL != nnzTotalDevHostPtr) {
        nnzb = *nnzTotalDevHostPtr;
    } else {
        cudaMemcpy(&nnzb, bsrRowPtr + mb, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&base, bsrRowPtr, sizeof(int), cudaMemcpyDeviceToHost);
        nnzb -= base;
    }

    printf("kkkkk6\n");

    long long a = (long long) nnzb * (long long)(blockDim * blockDim) * sizeof(float);
    printf("aaa === %lld", a);
    cudaStat1 = cudaMalloc((void**)&bsrColInd, nnzb * sizeof(int));
    cudaStat2 = cudaMalloc((void**)&bsrVal, a);
    if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess) {
        CLEANUP("Device malloc failed2 (BSR matrix)");
        printf("%s\n", cudaGetErrorString(cudaStat1));
        printf("%s\n", cudaGetErrorString(cudaStat2));
        return 1;
    }
    status = cusparseScsr2bsr(handle, CUSPARSE_DIRECTION_ROW, m, n, csrDescr, csrVal, csrRowPtr, csrColInd,
                              blockDim, bsrDescr, bsrVal, bsrRowPtr, bsrColInd);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("cusparseScsr2bsr failed");
        return 1;
    }
    
    // if (csrVal) {
    //     cudaFree(csrVal);
    //     csrVal = 0;
    // }
    // if (csrRowPtr) {
    //     cudaFree(csrRowPtr);
    //     csrRowPtr = 0;
    // }
    // if (csrColInd) {
    //     cudaFree(csrColInd);
    //     csrColInd = 0;
    // }

    printf("density:  %3.10f \n", (1.0 * nnzb) / ((mb * 1.0) * (nb * 1.0)));  

    printf("cusparseSbsrmm...\n");

    float time;
    cudaEvent_t start, stop;

    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    HANDLE_ERROR( cudaEventRecord(start, 0) );

    status = cusparseSbsrmm(handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE, mb, dim, nb, nnzb, &fone, bsrDescr, bsrVal,
                            bsrRowPtr, bsrColInd, blockDim, y, nb * blockDim, &fzero, z, mb * blockDim);

    HANDLE_ERROR( cudaEventRecord(stop, 0) );
    HANDLE_ERROR( cudaEventSynchronize(stop) );
    HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
                           
    printf("bsrmm cost time:  %3.10f ms \n", time);   

    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("bsrmm failed");
        return 1;
    }

    cudaStat1 = cudaMemcpy(zHostPtr, z, (size_t)(mb * blockDim * dim * sizeof(float)), cudaMemcpyDeviceToHost);
    if (cudaStat1 != cudaSuccess) {
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

    CLEANUP("end");

    return 0;
}