#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <random>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>
#include <assert.h>
#include <map>
#include <queue>
#include "cusparse.h"

static void handleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at linedd %d\n", cudaGetErrorString(err), file, line);
        exit(-1);
    }
}

#define HANDLE_ERROR( err ) (handleError( err, __FILE__, __LINE__ ))

std::mt19937_64 gen(1234);

template<typename T>
T* vec2ptr(std::vector<T> v) {
    T* ptr = (T*) malloc(v.size() * sizeof(T));
    for (size_t i = 0; i < v.size(); ++i) {
        ptr[i] = v[i];
    }
    return ptr;
}

int getBSR(int n, int nnz, int bsize, int** hostBsrRowPtr, int** hostBsrColInd, float** hostBsrVal) {
    std::vector<int> csrRowPtr, csrColInd;
    
    std::fstream s1("collab_naive_indptr.txt");
    std::fstream s2("collab_naive_indices.txt");
    
    int xx;
    s1 >> xx;
    assert(n + 1 == xx);
    for (int i = 0; i <= n; ++i) {
        int x;
        s1 >> x;
        csrRowPtr.push_back(x);
    }

    printf("h1\n");

    s2 >> xx;
    assert(nnz == xx);
    for (int i = 0; i < nnz; ++i) {
        int x;
        s2 >> x;
        csrColInd.push_back(x);
    }

    printf("h2\n");

    int nb = (n + bsize - 1) / bsize;
    std::vector<int> bsrRowPtr(nb + 1, 0), bsrColInd;
    std::vector<float> bsrVal;
    std::vector<int> ybs;
    ybs.reserve(1000);
    // std::vector<float> vals;
    for (int x1 = 0; x1 < nb; ++x1) {
        if (x1 % 100 == 0) {
            printf("x1 = %d\n", x1);
        }
        ybs.clear();
        for (int x2 = 0; x2 < bsize; ++x2) {
            int x = x1 * bsize + x2;
            if (x >= n) {
                break;
            }
            int sy = csrRowPtr[x], ey = csrRowPtr[x + 1];
            for (int i = sy; i < ey; ++i) {
                int y = csrColInd[i];
                ybs.push_back(y / bsize);
            }
        }
        std::sort(ybs.begin(), ybs.end());
        int num = 0, last = -1;
        std::map<int, int> ymp;
        for (int yb : ybs) {
            if (last != yb) {
                bsrColInd.push_back(yb);
                ymp[yb] = num;
                ++num;
                last = yb;
            }
        }
        bsrRowPtr[x1 + 1] = bsrRowPtr[x1] + num;
        int pos = bsrVal.size();
        int bbnum = num * bsize * bsize;
        bsrVal.reserve(pos + bbnum);
        for (int i = 0; i < bbnum; ++i) {
            bsrVal.push_back(0);
        }
        for (int x2 = 0; x2 < bsize; ++x2) {
            int x = x1 * bsize + x2;
            if (x >= n) {
                break;
            }
            int sy = csrRowPtr[x], ey = csrRowPtr[x + 1];
            for (int i = sy; i < ey; ++i) {
                int y = csrColInd[i];
                int y1 = y / bsize, y2 = y % bsize;
                int nth = ymp[y1];
                bsrVal[pos + nth * bsize * bsize + y2 * bsize + x2] = 1;
            }
        }
    }
    int nnzb = bsrVal.size();
    *hostBsrRowPtr = vec2ptr<int>(std::move(bsrRowPtr));
    *hostBsrColInd = vec2ptr<int>(std::move(bsrColInd));
    *hostBsrVal = vec2ptr<float>(std::move(bsrVal));
    return nnzb;
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

int main() {
    cudaError_t cudaStat1, cudaStat2, cudaStat3;
    cusparseStatus_t status;
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;

    int n = 235868;    
    int nnz = 2358104;
    int bsize = 16;
    int nb = (n + bsize - 1) / bsize;
    int nnzb = 0;
    int dim = 64;
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

    yHostPtr = randomDenseMatrix(nb * bsize, dim);
    zHostPtr = (float*) malloc(nb * bsize * dim * sizeof(float));

    cudaStat1 = cudaMalloc((void**)&y, nb * bsize * dim * sizeof(float));
    cudaStat2 = cudaMalloc((void**)&z, nb * bsize * dim * sizeof(float));
    if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess) {
        CLEANUP("Device malloc failed (dense matrix)");
        return 1;
    }

    cudaStat1 = cudaMemcpy(y, yHostPtr, (size_t)(nb * bsize * dim * sizeof(float)), cudaMemcpyHostToDevice);
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Memcpy from Host to Device failed (y)");
        return 1;
    }

    cudaStat1 = cudaMemset((void*)z, 0, nb * bsize * dim * sizeof(float));
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Memset on Device failed (z)");
        return 1;
    }

    printf("getBSR...\n");
    nnzb = getBSR(n, nnz, bsize, &hostBsrRowPtr, &hostBsrColInd, &hostBsrVal);
    printf("gotBSR\n");

    cudaStat1 = cudaMalloc((void**)&bsrRowPtr, (nb + 1) * sizeof(int));
    cudaStat2 = cudaMalloc((void**)&bsrColInd, nnzb * sizeof(int));
    cudaStat3 = cudaMalloc((void**)&bsrVal, nnzb * bsize * bsize * sizeof(float));
    if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess) {
        CLEANUP("Device malloc failed (CSR matrix)");
        return 1;
    }

    printf("kkkkk3\n");


    cudaStat1 = cudaMemcpy(bsrRowPtr, hostBsrRowPtr, (size_t)((nb + 1) * sizeof(int)), cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(bsrColInd, hostBsrColInd, (size_t)(nnzb * sizeof(int)), cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(bsrVal, hostBsrVal, (size_t)(nnzb * bsize * bsize * sizeof(float)), cudaMemcpyHostToDevice);
    if (cudaStat1 != cudaSuccess || cudaStat2 != cudaSuccess || cudaStat3 != cudaSuccess) {
        CLEANUP("Memcpy from Host to Device failed (CSR matrix)");
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


    cudaStat1 = cudaMalloc((void**)&bsrRowPtr, (nb + 1) * sizeof(int));
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Device malloc failed1 (BSR matrix)");
        printf("%s\n", cudaGetErrorString(cudaStat1));
        return 1;
    }

    // printf("density:  %3.10f \n", (1.0 * nnzb) / ((nb * 1.0) * (nb * 1.0)));  

    printf("cusparseSbsrmm...\n");

    float time;
    cudaEvent_t start, stop;

    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    HANDLE_ERROR( cudaEventRecord(start, 0) );

    status = cusparseSbsrmm(handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE, nb, dim, nb, nnzb, &fone, descr, bsrVal,
                            bsrRowPtr, bsrColInd, bsize, y, nb * bsize, &fzero, z, nb * bsize);

    HANDLE_ERROR( cudaEventRecord(stop, 0) );
    HANDLE_ERROR( cudaEventSynchronize(stop) );
    HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
                           
    printf("bsrmm cost time:  %3.10f ms \n", time);   

    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("bsrmm failed");
        return 1;
    }

    cudaStat1 = cudaMemcpy(zHostPtr, z, (size_t)(nb * bsize * dim * sizeof(float)), cudaMemcpyDeviceToHost);
    if (cudaStat1 != cudaSuccess) {
        CLEANUP("Memcpy from Device to Host failed (z)");
        return 1;
    }

    status = cusparseDestroyMatDescr(descr);
    descr = 0;
    if (status != CUSPARSE_STATUS_SUCCESS) {
        CLEANUP("CSR matrix descriptor destruction failed");
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