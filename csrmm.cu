#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cusparse.h"

#define CLEANUP(s) \
do { \
    printf("%s\n", s); \
    if (yHostPtr) free(yHostPtr); \
    if (zHostPtr) free(zHostPtr); \
    if (cooRowIndexHostPtr) free(cooRowIndexHostPtr); \
    if (cooColIndexHostPtr) free(cooColIndexHostPtr); \
    if (cooValHostPtr) free(cooValHostPtr); \
    if (y) cudaFree(y); \
    if (z) cudaFree(z); \
    if (csrRowPtr) cudaFree(csrRowPtr); \
    if (cooRowIndex) cudaFree(cooRowIndex); \
    if (cooColIndex) cudaFree(cooColIndex); \
    if (cooVal) cudaFree(cooVal); \
    if (descr) cusparseDestroyMatDescr(descr); \
    if (handle) cusparseDestroy(handle); \
    cudaDeviceReset(); \
    fflush(stdout); \
} while (0) \


int main() {
    cudaError_t cudaStat1, cudaStat2, cudaStat3, cudaStat4;
    cusparseStatus_t status;
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;
    int* cooRowIndexHostPtr = 0;
    int* cooColIndexHostPtr = 0;
    float* cooValHostPtr = 0;
    int* cooRowIndex = 0;
    int* cooColIndex = 0;
    float* cooVal = 0;
    int* csrRowPtr = 0;
    float* yHostPtr = 0;
    float* y = 0;
    float* zHostPtr = 0;
    float* z = 0;
    int n, nnz;
    float fzero = 0.0;
    float fone = 1.0;

    printf("testing examples\n");
    n = 4; nnz = 9;
    cooRowIndexHostPtr = (int*) malloc(nnz * sizeof(int));
    cooColIndexHostPtr = (int*) malloc(nnz * sizeof(int));
    cooValHostPtr = (float*) malloc(nnz * sizeof(float));
    if ((!cooRowIndexHostPtr) || (!cooColIndexHostPtr) || (!cooValHostPtr)) {
        CLEANUP("Host malloc failed (coo matrix)");
        return 1;
    }

    cooRowIndexHostPtr[0]=0; cooColIndexHostPtr[0]=0; cooValHostPtr[0]=1.0;
    cooRowIndexHostPtr[1]=0; cooColIndexHostPtr[1]=2; cooValHostPtr[1]=2.0;
    cooRowIndexHostPtr[2]=0; cooColIndexHostPtr[2]=3; cooValHostPtr[2]=3.0;
    cooRowIndexHostPtr[3]=1; cooColIndexHostPtr[3]=1; cooValHostPtr[3]=4.0;
    cooRowIndexHostPtr[4]=2; cooColIndexHostPtr[4]=0; cooValHostPtr[4]=5.0;
    cooRowIndexHostPtr[5]=2; cooColIndexHostPtr[5]=2; cooValHostPtr[5]=6.0;
    cooRowIndexHostPtr[6]=2; cooColIndexHostPtr[6]=3; cooValHostPtr[6]=7.0;
    cooRowIndexHostPtr[7]=3; cooColIndexHostPtr[7]=1; cooValHostPtr[7]=8.0;
    cooRowIndexHostPtr[8]=3; cooColIndexHostPtr[8]=3; cooValHostPtr[8]=9.0;

    yHostPtr = (float*) malloc(2 * n * sizeof(float));
    zHostPtr = (float*) malloc(2 * n * sizeof(float));
    if ((!yHostPtr) || (!zHostPtr)) {
        CLEANUP("Host malloc failed (dense matrix)");
        return 1;
    }

    yHostPtr[0] = 10.0;  
    yHostPtr[1] = 20.0;  
    yHostPtr[2] = 30.0;
    yHostPtr[3] = 40.0;
    yHostPtr[4] = 50.0;
    yHostPtr[5] = 60.0;
    yHostPtr[6] = 70.0;
    yHostPtr[7] = 80.0;

    cudaStat1 = cudaMalloc((void**)&cooRowIndex, nnz * sizeof(int));
    cudaStat2 = cudaMalloc((void**)&cooColIndex, nnz * sizeof(int));
    cudaStat3 = cudaMalloc((void**)&cooVal, nnz * sizeof(float));
    cudaStat4 = cudaMalloc((void**)&y, 2 * n * sizeof(float));

    if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess || cudaStat4 != cudaSuccess) {
        CLEANUP("Device malloc failed");
        return 1;
    }

    cudaStat1 = cudaMemcpy(cooRowIndex, cooRowIndexHostPtr, (size_t)(nnz * sizeof(int)), cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(cooColIndex, cooColIndexHostPtr, (size_t)(nnz * sizeof(int)), cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(cooVal, cooValHostPtr, (size_t)(nnz * sizeof(float)), cudaMemcpyHostToDevice);
    cudaStat4 = cudaMemcpy(y, yHostPtr, (size_t)(2 * n * sizeof(float)), cudaMemcpyHostToDevice);

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

    cudaStat1 = cudaMalloc((void**)&csrRowPtr, (n + 1) * sizeof(int));
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Device malloc failed (csrRowPtr)");
        return 1;
    }

    status = cusparseXcoo2csr(handle, cooRowIndex, nnz, n, csrRowPtr, CUSPARSE_INDEX_BASE_ZERO);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("Conversion from COO to CSR format failed");
        return 1;
    }

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

    cudaStat1 = cudaMalloc((void**)&z, 2 * n * sizeof(float));
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Device malloc failed (z)");
        return 1;
    }

    cudaStat1 = cudaMemset((void*)z, 0, 2 * n * sizeof(float));
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Memset on Device failed");
        return 1;
    }

    status = cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, 2, n, nnz, 
                            &fone, descr, cooVal, csrRowPtr, cooColIndex, y, n, &fzero, z, n);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("csrmm failed");
        return 1;
    }

    cudaStat1 = cudaMemcpy(zHostPtr, z, (size_t)(2 * n * sizeof(float)), cudaMemcpyDeviceToHost);
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

    for (int i = 0; i < 2 * n; ++i) {
        printf("%f ", zHostPtr[i]);
    }

    printf("\n");
    CLEANUP("end");
}

