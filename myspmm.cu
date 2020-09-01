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

template <typename T, int BSR_BLOCK_DIM, int BLK_SIZE_Y>
__global__ void bsrmm_large_blockdim_device(cusparseHandle_t handle,
                                            cusparseDirection_t dir_A,
                                                   rocsparse_operation trans_B,
                                                   rocsparse_int       Mb,
                                                   rocsparse_int       N,
                                                   T                   alpha,
                                                   const rocsparse_int* __restrict__ bsr_row_ptr,
                                                   const rocsparse_int* __restrict__ bsr_col_ind,
                                                   const T* __restrict__ bsr_val,
                                                   rocsparse_int block_dim,
                                                   const T* __restrict__ B,
                                                   rocsparse_int ldb,
                                                   T             beta,
                                                   T* __restrict__ C,
                                                   rocsparse_int        ldc,
                                                   rocsparse_index_base idx_base)
{
    rocsparse_int tidx = hipThreadIdx_x;
    rocsparse_int tidy = hipThreadIdx_y;

    rocsparse_int global_row = tidx + hipBlockIdx_x * block_dim;
    rocsparse_int global_col = tidy + hipBlockIdx_y * BLK_SIZE_Y;

    rocsparse_int block_row = hipBlockIdx_x;

    rocsparse_int block_row_start = 0;
    rocsparse_int block_row_end   = 0;
    if(block_row < Mb)
    {
        block_row_start = bsr_row_ptr[block_row] - idx_base;
        block_row_end   = bsr_row_ptr[block_row + 1] - idx_base;
    }

    rocsparse_int colB = global_col * ldb;
    rocsparse_int colC = global_col * ldc;

    __shared__ T shared_B[BSR_BLOCK_DIM * BLK_SIZE_Y];
    __shared__ T shared_A[BSR_BLOCK_DIM * BSR_BLOCK_DIM];

    T sum = static_cast<T>(0);

    rocsparse_int index         = BSR_BLOCK_DIM * tidy + tidx;
    rocsparse_int block_dim_sqr = block_dim * block_dim;

    for(rocsparse_int k = block_row_start; k < block_row_end; k++)
    {
        rocsparse_int block_col = (bsr_col_ind[k] - idx_base);

        if(trans_B == rocsparse_operation_none)
        {
            shared_B[index] = (global_col < N && tidx < block_dim)
                                  ? B[block_dim * block_col + tidx + colB]
                                  : static_cast<T>(0);
        }
        else
        {
            shared_B[index] = (global_col < N && tidx < block_dim)
                                  ? B[global_col + ldb * (block_dim * block_col + tidx)]
                                  : static_cast<T>(0);
        }

        if(direction == rocsparse_direction_row)
        {
            if(tidx < block_dim && tidy < block_dim)
            {
                shared_A[index] = bsr_val[block_dim_sqr * k + block_dim * tidx + tidy];
            }
        }
        else
        {
            if(tidx < block_dim && tidy < block_dim)
            {
                shared_A[index] = bsr_val[block_dim_sqr * k + block_dim * tidy + tidx];
            }
        }

        __syncthreads();

        for(rocsparse_int j = 0; j < block_dim; j++)
        {
            sum = rocsparse_fma(
                shared_A[BSR_BLOCK_DIM * j + tidx], shared_B[BSR_BLOCK_DIM * tidy + j], sum);
        }

        __syncthreads();
    }

    if(block_row < Mb && global_col < N && tidx < block_dim)
    {
        if(beta == static_cast<T>(0))
        {
            C[global_row + colC] = alpha * sum;
        }
        else
        {
            C[global_row + colC] = rocsparse_fma(beta, C[global_row + colC], alpha * sum);
        }
    }
}

template <typename T, int BSR_BLOCK_DIM, int BLK_SIZE_Y>
__global__ void bsrmm_general_blockdim_device(cusparseHandle_t handle,
                                              cusparseDirection_t dir_A,
                                              cusparseOperation_t trans_B,
                                              int Mb,
                                              int N,
                                              T   alpha,
                                              const int* __restrict__ bsr_row_ptr,
                                              const int* __restrict__ bsr_col_ind,
                                              const T* __restrict__ bsr_val,
                                              int block_dim,
                                              const T* __restrict__ B,
                                              int ldb,
                                              T   beta,
                                              T* __restrict__ C,
                                              int ldc,
                                              int idx_base)
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    int block_row = blockIdx.x;

    int block_row_start = 0;
    int block_row_end   = 0;
    if(block_row < Mb)
    {
        block_row_start = bsr_row_ptr[block_row] - idx_base;
        block_row_end   = bsr_row_ptr[block_row + 1] - idx_base;
    }

    __shared__ T shared_B[BSR_BLOCK_DIM * BLK_SIZE_Y];
    __shared__ T shared_A[BSR_BLOCK_DIM * BSR_BLOCK_DIM];

    int global_col = tidy + blockIdx.y * BLK_SIZE_Y;

    int colB = global_col * ldb;
    int colC = global_col * ldc;

    for(int x = 0; x < block_dim; x += BSR_BLOCK_DIM)
    {
        int global_row = tidx + x + blockIdx.x * block_dim;

        T sum = static_cast<T>(0);

        for(int k = block_row_start; k < block_row_end; k++)
        {
            int block_col = (bsr_col_ind[k] - idx_base);

            for(int y = 0; y < block_dim; y += BLK_SIZE_Y)
            {
                if(trans_B == CUSPARSE_OPERATION_NON_TRANSPOSE)
                {
                    shared_B[BSR_BLOCK_DIM * tidy + tidx]
                        = (global_col < N && (tidx + y) < block_dim)
                              ? B[block_dim * block_col + (tidx + y) + colB]
                              : static_cast<T>(0);
                }
                else
                {
                    shared_B[BSR_BLOCK_DIM * tidy + tidx]
                        = (global_col < N && (tidx + y) < block_dim)
                              ? B[global_col + ldb * (block_dim * block_col + (tidx + y))]
                              : static_cast<T>(0);
                }

                if(dir_A == CUSPARSE_DIRECTION_ROW)
                {
                    shared_A[BSR_BLOCK_DIM * tidy + tidx]
                        = ((tidx + x) < block_dim && (tidy + y) < block_dim)
                              ? bsr_val[block_dim * block_dim * k + block_dim * (tidx + x)
                                        + (tidy + y)]
                              : static_cast<T>(0);
                }
                else
                {
                    shared_A[BSR_BLOCK_DIM * tidy + tidx]
                        = ((tidx + x) < block_dim && (tidy + y) < block_dim)
                              ? bsr_val[block_dim * block_dim * k + block_dim * (tidy + y)
                                        + (tidx + x)]
                              : static_cast<T>(0);
                }

                __syncthreads();

                for(int j = 0; j < BSR_BLOCK_DIM; j++)
                {
                    sum = shared_A[BSR_BLOCK_DIM * j + tidx] * shared_B[BSR_BLOCK_DIM * tidy + j] + sum;
                }

                __syncthreads();
            }
        }

        if(block_row < Mb && global_col < N && (tidx + x) < block_dim)
        {
            if(beta == static_cast<T>(0))
            {
                C[global_row + colC] = alpha * sum;
            }
            else
            {
                C[global_row + colC] = beta * C[global_row + colC] + alpha * sum;
            }
        }
    }
}

std::mt19937_64 gen(1234);

template<typename T>
T* vec2ptr(std::vector<T> v) {
    T* ptr = (T*) malloc(v.size() * sizeof(T));
    for (int i = 0; i < v.size(); ++i) {
        ptr[i] = v[i];
    }
    return ptr;
}

int randomBSRMatrix(int mb, int nb, int blockDim, float p, int** hostBsrRowPtr, int** hostBsrColInd, float** hostBsrVal, float minVal=-1, float maxVal=1) {
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
        // if (i % 1000 == 0) {
        //     printf("i = %d\n", i);
        // }
    }
    *hostBsrColInd = vec2ptr(std::move(indices));
    *hostBsrVal = vec2ptr(std::move(vals));

    // Generating random BSR matrix may be time-consuming, so we record it for next time use.
    // std::string bd = std::to_string(blockDim);
    // std::fstream s1("bsr_" + bd + "_indptr.txt", std::ios::out | std::ios::trunc);
    // std::fstream s2("bsr_" + bd + "_indices.txt", std::ios::out | std::ios::trunc);

    // s1 << mb + 1 << std::endl;
    // for (int i = 0; i <= mb; ++i) {
    //     s1 << (*hostBsrRowPtr)[i] << " ";
    // }
    // s1 << std::endl;

    // s2 << cnt << std::endl;
    // for (int i = 0; i < cnt; ++i) {
    //     s2 << (*hostBsrColInd)[i] << " ";
    // }
    // s2 << std::endl;

    return cnt;
}

int readAndFillBSRMatrix(int mb, int nb, int blockDim, int** hostBsrRowPtr, int** hostBsrColInd, float** hostBsrVal, float minVal=-1, float maxVal=1) {
    std::string bd = std::to_string(blockDim);
    std::fstream s1("bsr_" + bd + "_indptr.txt", std::ios::in);
    std::fstream s2("bsr_" + bd + "_indices.txt", std::ios::in);

    int xx;
    s1 >> xx;
    assert(mb + 1 == xx);
    *hostBsrRowPtr = (int*) malloc((mb + 1) * sizeof(int));
    for (int i = 0; i <= mb; ++i) {
        s1 >> (*hostBsrRowPtr)[i];
    }

    int nnzb;
    s2 >> nnzb;
    *hostBsrColInd = (int*) malloc(nnzb * sizeof(int));
    for (int i = 0; i < nnzb; ++i) {
        s2 >> (*hostBsrColInd)[i];
    }

    int num = nnzb * blockDim * blockDim;
    *hostBsrVal = (float*) malloc(num * sizeof(float));
    std::uniform_real_distribution<float> dist(minVal, maxVal);
    for (int i = 0; i < num; ++i) {
        (*hostBsrVal)[i] = dist(gen);
    }

    return nnzb;
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
if (err != cudaSuccess) { \
    printf("error occurred in %s at line %d\n", __FILE__, __LINE__); \
    CLEANUP(cudaGetErrorString(err)); \
    exit(-1); \
}

#define HANDLE_CUSPARSE_ERROR( err, s ) \
if (err != CUSPARSE_STATUS_SUCCESS) { \
    CLEANUP(s); \
    exit(-1); \
}

int main(int argc, char* argv[]) {
    std::string which_bsrmm(argv[1]);
    std::cout << "which bsrmm: " << which_bsrmm << std::endl;
    float p = 0.02;
    int blockDim = 64; 
    int dim = 64;
    printf("p = %f blockDim = %d dim = %d\n", p, blockDim, dim);

    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;

    int m = 131072;
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

    printf("generate random BSR matrix\n");

    nnzb = randomBSRMatrix(mb, nb, blockDim, p, &hostBsrRowPtr, &hostBsrColInd, &hostBsrVal);
    // nnzb = readAndFillBSRMatrix(mb, nb, blockDim, &hostBsrRowPtr, &hostBsrColInd, &hostBsrVal);

    printf("density of BSR matrix is %f\n", (nnzb * 1.0) / (mb * nb));

    printf("gpu memory malloc and memcpy...\n");

    HANDLE_ERROR( cudaMalloc((void**)&bsrRowPtr, (mb + 1) * sizeof(int)) );
    HANDLE_ERROR( cudaMalloc((void**)&bsrColInd, nnzb * sizeof(int)) );
    HANDLE_ERROR( cudaMalloc((void**)&bsrVal, nnzb * blockDim * blockDim * sizeof(float)) );

    HANDLE_ERROR( cudaMemcpy(bsrRowPtr, hostBsrRowPtr, (size_t)((mb + 1) * sizeof(int)), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(bsrColInd, hostBsrColInd, (size_t)(nnzb * sizeof(int)), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(bsrVal, hostBsrVal, (size_t)(nnzb * blockDim * blockDim * sizeof(float)), cudaMemcpyHostToDevice) );
    
    HANDLE_CUSPARSE_ERROR( cusparseCreate(&handle), "CUSPARSE Library initialization failed" );
    
    HANDLE_CUSPARSE_ERROR( cusparseCreateMatDescr(&descr), "BSR Matrix descriptor initialization failed" );
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    printf("prepare y and z...\n");

    yHostPtr = randomDenseMatrix(n, dim);
    zHostPtr = (float*) malloc(m * dim * sizeof(float));

    HANDLE_ERROR( cudaMalloc((void**)&y, n * dim * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void**)&z, m * dim * sizeof(float)) );

    HANDLE_ERROR( cudaMemcpy(y, yHostPtr, (size_t)(n * dim * sizeof(float)), cudaMemcpyHostToDevice) );

    printf("cusparseSbsrmm...\n"); 

    HANDLE_ERROR( cudaMemset((void*)z, 0, m * dim * sizeof(float)) );

    dim3 gridSize(mb, (dim - 1) / 32 + 1);
    dim3 blockSize(32, 32, 1);

    float time;
    cudaEvent_t start, stop;
    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    HANDLE_ERROR( cudaEventRecord(start, 0) );

    if (which_bsrmm == "rocm") {
        bsrmm_general_blockdim_device<float, 32, 32><<<gridSize, blockSize>>>(
            handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, dim, fone,
            bsrRowPtr, bsrColInd, bsrVal, blockDim, y, n, fzero, z, m, 0);
    }  else {
        HANDLE_CUSPARSE_ERROR( cusparseSbsrmm(handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE, mb, dim, nb, nnzb, &fone, descr, bsrVal,
                                              bsrRowPtr, bsrColInd, blockDim, y, n, &fzero, z, m),
                               "cusparseSbsrmm failed" ); 
    }

    HANDLE_ERROR( cudaEventRecord(stop, 0) );
    HANDLE_ERROR( cudaEventSynchronize(stop) );
    HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
    printf("bsrmm cost time:  %3.10f ms \n", time);   

    HANDLE_ERROR( cudaMemcpy(zHostPtr, z, (size_t)(m * dim * sizeof(float)), cudaMemcpyDeviceToHost) );

    HANDLE_CUSPARSE_ERROR( cusparseDestroyMatDescr(descr), "Matrix descriptor destruction failed" );
    descr = 0;
    HANDLE_CUSPARSE_ERROR( cusparseDestroy(handle), "CUSPARSE Library release of resources failed" );
    handle = 0;

    CLEANUP("end");

    return 0;
}