#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <random>
#include <fstream>
#include <assert.h>
#include <cuda_runtime.h>
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
        // if (i % 1000 == 0) {
        //     printf("i = %d\n", i);
        // }
    }
    *hostCsrColInd = vec2ptr(std::move(indices));
    *hostCsrVal = vec2ptr(std::move(vals));

    // Generating random CSR matrix is time-consuming, so we record it for next time use.
    // std::fstream s1("csr_indptr.txt", std::ios::out | std::ios::trunc);
    // std::fstream s2("csr_indices.txt", std::ios::out | std::ios::trunc);

    // s1 << m + 1 << std::endl;
    // for (int i = 0; i <= m; ++i) {
    //     s1 << (*hostCsrRowPtr)[i] << " ";
    // }
    // s1 << std::endl;

    // s2 << cnt << std::endl;
    // for (int i = 0; i < cnt; ++i) {
    //     s2 << (*hostCsrColInd)[i] << " ";
    // }
    // s2 << std::endl;

    return cnt;
}

int readAndFillCSRMatrix(int m, int n, int** hostCsrRowPtr, int** hostCsrColInd, float** hostCsrVal, float minVal=-1, float maxVal=1) {
    std::fstream s1("csr_indptr.txt", std::ios::in);
    std::fstream s2("csr_indices.txt", std::ios::in);

    int xx;
    s1 >> xx;
    assert(m + 1 == xx);
    *hostCsrRowPtr = (int*) malloc((m + 1) * sizeof(int));
    for (int i = 0; i <= m; ++i) {
        s1 >> (*hostCsrRowPtr)[i];
    }

    int nnz;
    s2 >> nnz;
    *hostCsrColInd = (int*) malloc(nnz * sizeof(int));
    for (int i = 0; i < nnz; ++i) {
        s2 >> (*hostCsrColInd)[i];
    }

    *hostCsrVal = (float*) malloc(nnz * sizeof(float));
    std::uniform_real_distribution<float> dist(minVal, maxVal);
    for (int i = 0; i < nnz; ++i) {
        (*hostCsrVal)[i] = dist(gen);
    }

    return nnz;
}

float* randomDenseMatrix(int n, int dim, float minVal=-1, float maxVal=1) {
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

    nnz = randomCSRMatrix(m, n, p, &hostCsrRowPtr, &hostCsrColInd, &hostCsrVal);
    // nnz = readAndFillCSRMatrix(m, n, &hostCsrRowPtr, &hostCsrColInd, &hostCsrVal);

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