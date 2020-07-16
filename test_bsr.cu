#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "cusparse.h"

static void handleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(-1);
    }
}

#define HANDLE_ERROR( err ) (handleError( err, __FILE__, __LINE__ ))

std::mt19937_64 gen(1234);

template<typename T>
T* vec2ptr(std::vector<T> v) {
    T* ptr = (T*) malloc(v.size() * sizeof(T));
    for (int i = 0; i < v.size(); ++i) {
        ptr[i] = v[i];
    }
    return ptr;
}

int randomBSRMatrix(int mb, int nb, int blockDim, float p, int** hostBsrRowPtr, int** hostBsrColInd, float** hostBsrVal, float minVal=-10, float maxVal=10) {
    std::uniform_real_distribution<float> flip(0, 1), dist(minVal, maxVal);
    int blockNum = blockDim * blockDim;
    *hostBsrRowPtr = (int*) malloc((mb + 1) * sizeof(int));
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
    }
    *hostBsrColInd = vec2ptr(std::move(indices));
    *hostBsrVal = vec2ptr(std::move(vals));

    return cnt;
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
} while (0) \

int main(int argc, char* argv[]) {
    float p = std::stof(argv[1]);
    int blockDim = std::stoi(argv[2]); 
    int dim = std::stoi(argv[3]);
    printf("p = %f blockDim = %d dim = %d\n", p, blockDim, dim);

    cudaError_t cudaStat1, cudaStat2, cudaStat3;
    cusparseStatus_t status;
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;

    int m = 131072;
    int n = m;
    int mb = m / blockDim;
    int nb = n / blockDim;
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

    printf("generate random BSR matrix\n");

    nnzb = randomBSRMatrix(mb, nb, blockDim, p, &hostBsrRowPtr, &hostBsrColInd, &hostBsrVal);

    printf("gpu memory malloc and memcpy...\n");

    cudaStat1 = cudaMalloc((void**)&bsrRowPtr, (mb + 1) * sizeof(int));
    cudaStat2 = cudaMalloc((void**)&bsrColInd, nnzb * sizeof(int));
    cudaStat3 = cudaMalloc((void**)&bsrVal, nnzb * blockDim * blockDim * sizeof(float));
    if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess) {
        CLEANUP("Device malloc failed (BSR matrix)");
        return 1;
    }

    cudaStat1 = cudaMemcpy(bsrRowPtr, hostBsrRowPtr, (size_t)((mb + 1) * sizeof(int)), cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(bsrColInd, hostBsrColInd, (size_t)(nnzb * sizeof(int)), cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(bsrVal, hostBsrVal, (size_t)(nnzb * blockDim * blockDim * sizeof(float)), cudaMemcpyHostToDevice);
    if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess) {
        CLEANUP("Memcpy from Host to Device failed (BSR matrix)");
        return 1;
    }
    
    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("CUSPARSE Library initialization failed");
        return 1;
    }
    
    status = cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("BSR Matrix descriptor initialization failed");
        return 1;
    }
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    yHostPtr = randomDenseMatrix(n, dim);
    zHostPtr = (float*) malloc(m * dim * sizeof(float));

    cudaStat1 = cudaMalloc((void**)&y, n * dim * sizeof(float));
    cudaStat2 = cudaMalloc((void**)&z, m * dim * sizeof(float));
    if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess) {
        CLEANUP("Device malloc failed (dense matrix)");
        return 1;
    }

    cudaStat1 = cudaMemcpy(y, yHostPtr, (size_t)(n * dim * sizeof(float)), cudaMemcpyHostToDevice);
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Memcpy from Host to Device failed (y)");
        return 1;
    }

    cudaStat1 = cudaMemset((void*)z, 0, m * dim * sizeof(float));
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Memset on Device failed (z)");
        return 1;
    }

    printf("cusparseSbsrmm...\n");

    float time;
    cudaEvent_t start, stop;

    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    HANDLE_ERROR( cudaEventRecord(start, 0) );

    status = cusparseSbsrmm(handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE, mb, dim, nb, nnzb, &fone, descr, bsrVal,
                            bsrRowPtr, bsrColInd, blockDim, y, n, &fzero, z, m);

    HANDLE_ERROR( cudaEventRecord(stop, 0) );
    HANDLE_ERROR( cudaEventSynchronize(stop) );
    HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
                           
    printf("bsrmm cost time:  %3.10f ms \n", time);   

    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("bsrmm failed");
        return 1;
    }

    cudaStat1 = cudaMemcpy(zHostPtr, z, (size_t)(m * dim * sizeof(float)), cudaMemcpyDeviceToHost);
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Memcpy from Device to Host failed (z)");
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

    CLEANUP("end");

    return 0;
}