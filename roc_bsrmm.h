#pragma once
#ifndef BSRMM_DEVICE_H
#define BSRMM_DEVICE_H

#include <cuda_runtime.h>
#include "cusparse.h"

__device__ __forceinline__ float myfma(float x, float y, float z) { return fmaf(x, y, z); }
__device__ __forceinline__ double myfma(double x, double y, double z) { return fma(x, y, z); }

template <typename T, int BLOCKSIZE, int WF_SIZE, int BSR_BLOCK_DIM>
static __device__ void bsrmmnn_small_blockdim_device(cusparseDirection_t direction,
                                                     int Mb,
                                                     int N,
                                                     T   alpha,
                                                     const int* __restrict__ bsr_row_ptr,
                                                     const int* __restrict__ bsr_col_ind,
                                                     const T* __restrict__ bsr_val,
                                                     const T* __restrict__ B,
                                                     int ldb,
                                                     T   beta,
                                                     T* __restrict__ C,
                                                     int ldc,
                                                     int idx_base)
{
    constexpr int PADDED_BSR_BLOCK_DIM = (BSR_BLOCK_DIM + 1);

    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x + tid;
    int lid  = gid & (WF_SIZE - 1);
    int wid  = tid / WF_SIZE;
    int nwfb = gridDim.x * blockDim.x / (WF_SIZE * BSR_BLOCK_DIM);
    int col  = lid + blockDim.y * WF_SIZE;

    int colB = col * ldb;
    int colC = col * ldc;

    // global row
    int global_row = (gid / WF_SIZE);

    // local row within block row
    int local_row = (gid / WF_SIZE) % BSR_BLOCK_DIM;

    __shared__ int shared_col[BLOCKSIZE / WF_SIZE][WF_SIZE];
    __shared__ T   shared_val[BLOCKSIZE / WF_SIZE][WF_SIZE * PADDED_BSR_BLOCK_DIM];

    for(int block_row = gid / (WF_SIZE * BSR_BLOCK_DIM); block_row < Mb;
        block_row += nwfb)
    {
        int block_row_start = bsr_row_ptr[block_row] - idx_base;
        int block_row_end   = bsr_row_ptr[block_row + 1] - idx_base;

        T sum = static_cast<T>(0);

        for(int j = block_row_start; j < block_row_end; j += WF_SIZE)
        {
            int k = j + lid;

            shared_col[wid][lid]
                = (k < block_row_end) ? BSR_BLOCK_DIM * (bsr_col_ind[k] - idx_base) : 0;

            if(direction == CUSPARSE_DIRECTION_ROW)
            {
                // Perform:
                // for(int l = 0; l < BSR_BLOCK_DIM; l++)
                // {
                //     shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + l]
                //         = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                //                                               + BSR_BLOCK_DIM * local_row + l]
                //                               : static_cast<T>(0);
                // }
                // as unrolled loop.
                shared_val[wid][PADDED_BSR_BLOCK_DIM * lid]
                    = (k < block_row_end)
                          ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k + BSR_BLOCK_DIM * local_row]
                          : static_cast<T>(0);
                if(BSR_BLOCK_DIM >= 2)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 1]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * local_row + 1]
                                              : static_cast<T>(0);
                }
                if(BSR_BLOCK_DIM >= 3)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 2]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * local_row + 2]
                                              : static_cast<T>(0);
                }
                if(BSR_BLOCK_DIM >= 4)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 3]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * local_row + 3]
                                              : static_cast<T>(0);
                }
            }
            else
            {
                // Perform:
                // for(int l = 0; l < BSR_BLOCK_DIM; l++)
                // {
                //     shared_val[wid][BSR_BLOCK_DIM * lid + l]
                //         = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                //                                               + BSR_BLOCK_DIM * l + local_row]
                //                               : static_cast<T>(0);
                // }
                // as unrolled loop.
                shared_val[wid][PADDED_BSR_BLOCK_DIM * lid]
                    = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k + local_row]
                                          : static_cast<T>(0);
                if(BSR_BLOCK_DIM >= 2)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 1]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * 1 + local_row]
                                              : static_cast<T>(0);
                }
                if(BSR_BLOCK_DIM >= 3)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 2]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * 2 + local_row]
                                              : static_cast<T>(0);
                }
                if(BSR_BLOCK_DIM >= 4)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 3]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * 3 + local_row]
                                              : static_cast<T>(0);
                }
            }

            __syncthreads();

            if(col < N)
            {
                for(int i = 0; i < WF_SIZE; ++i)
                {
                    // Perform:
                    // for(int l = 0; l < BSR_BLOCK_DIM; l++)
                    // {
                    //     sum = myfma(shared_val[wid][PADDED_BSR_BLOCK_DIM * i + l],
                    //                         B[shared_col[wid][i] + l],
                    //                         sum);
                    // }
                    // as unrolled loop.
                    sum = myfma(shared_val[wid][PADDED_BSR_BLOCK_DIM * i],
                                        B[shared_col[wid][i] + colB],
                                        sum);
                    if(BSR_BLOCK_DIM >= 2)
                    {
                        sum = myfma(shared_val[wid][PADDED_BSR_BLOCK_DIM * i + 1],
                                            B[shared_col[wid][i] + 1 + colB],
                                            sum);
                    }
                    if(BSR_BLOCK_DIM >= 3)
                    {
                        sum = myfma(shared_val[wid][PADDED_BSR_BLOCK_DIM * i + 2],
                                            B[shared_col[wid][i] + 2 + colB],
                                            sum);
                    }
                    if(BSR_BLOCK_DIM >= 4)
                    {
                        sum = myfma(shared_val[wid][PADDED_BSR_BLOCK_DIM * i + 3],
                                            B[shared_col[wid][i] + 3 + colB],
                                            sum);
                    }
                }
            }
        }

        if(col < N)
        {
            if(beta == static_cast<T>(0))
            {
                C[global_row + colC] = alpha * sum;
            }
            else
            {
                C[global_row + colC] = myfma(beta, C[global_row + colC], alpha * sum);
            }
        }
    }
}

template <typename T, int BLOCKSIZE, int WF_SIZE, int BSR_BLOCK_DIM>
static __device__ void bsrmmnt_small_blockdim_device(cusparseDirection_t direction,
                                                     int Mb,
                                                     int N,
                                                     T   alpha,
                                                     const int* __restrict__ bsr_row_ptr,
                                                     const int* __restrict__ bsr_col_ind,
                                                     const T* __restrict__ bsr_val,
                                                     const T* __restrict__ B,
                                                     int ldb,
                                                     T   beta,
                                                     T* __restrict__ C,
                                                     int ldc,
                                                     int idx_base)
{
    constexpr int PADDED_BSR_BLOCK_DIM = (BSR_BLOCK_DIM + 1);

    int tid        = threadIdx.x;
    int gid        = blockIdx.x * blockDim.x + tid;
    int block_row  = gid / (WF_SIZE * BSR_BLOCK_DIM);
    int global_row = gid / WF_SIZE;
    int local_row  = (gid / WF_SIZE) % BSR_BLOCK_DIM;
    int lid        = tid & (WF_SIZE - 1);
    int wid        = tid / WF_SIZE;

    if(block_row >= Mb)
    {
        return;
    }

    __shared__ int shared_col[BLOCKSIZE / WF_SIZE][WF_SIZE];
    __shared__ T   shared_val[BLOCKSIZE / WF_SIZE][WF_SIZE * PADDED_BSR_BLOCK_DIM];

    int block_row_start = bsr_row_ptr[block_row] - idx_base;
    int block_row_end   = bsr_row_ptr[block_row + 1] - idx_base;

    for(int l = 0; l < N; l += WF_SIZE)
    {
        int col = l + lid;
        T   sum = static_cast<T>(0);

        for(int j = block_row_start; j < block_row_end; j += WF_SIZE)
        {
            int k = j + lid;

            shared_col[wid][lid]
                = (k < block_row_end) ? N * BSR_BLOCK_DIM * (bsr_col_ind[k] - idx_base) : 0;

            if(direction == CUSPARSE_DIRECTION_ROW)
            {
                // Perform:
                // for(int p = 0; p < BSR_BLOCK_DIM; p++)
                // {
                //     shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + p]
                //         = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                //                                               + BSR_BLOCK_DIM * local_row + p]
                //                               : static_cast<T>(0);
                // }
                // as unrolled loop.
                shared_val[wid][PADDED_BSR_BLOCK_DIM * lid]
                    = (k < block_row_end)
                          ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k + BSR_BLOCK_DIM * local_row]
                          : static_cast<T>(0);
                if(BSR_BLOCK_DIM >= 2)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 1]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * local_row + 1]
                                              : static_cast<T>(0);
                }
                if(BSR_BLOCK_DIM >= 3)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 2]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * local_row + 2]
                                              : static_cast<T>(0);
                }
                if(BSR_BLOCK_DIM >= 4)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 3]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * local_row + 3]
                                              : static_cast<T>(0);
                }
            }
            else
            {
                // Perform:
                // for(int p = 0; p < BSR_BLOCK_DIM; p++)
                // {
                //     shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + p]
                //         = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                //                                               + BSR_BLOCK_DIM * p + local_row]
                //                               : static_cast<T>(0);
                // }
                // as unrolled loop.
                shared_val[wid][PADDED_BSR_BLOCK_DIM * lid]
                    = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k + local_row]
                                          : static_cast<T>(0);
                if(BSR_BLOCK_DIM >= 2)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 1]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * 1 + local_row]
                                              : static_cast<T>(0);
                }
                if(BSR_BLOCK_DIM >= 3)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 2]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * 2 + local_row]
                                              : static_cast<T>(0);
                }
                if(BSR_BLOCK_DIM >= 4)
                {
                    shared_val[wid][PADDED_BSR_BLOCK_DIM * lid + 3]
                        = (k < block_row_end) ? bsr_val[BSR_BLOCK_DIM * BSR_BLOCK_DIM * k
                                                        + BSR_BLOCK_DIM * 3 + local_row]
                                              : static_cast<T>(0);
                }
            }

            __syncthreads();

            if(col < N)
            {
                for(int i = 0; i < WF_SIZE; ++i)
                {
                    // Perform:
                    // for(int p = 0; p < BSR_BLOCK_DIM; p++)
                    // {
                    //     T val_B = rocsparse_ldg(B + col + N * p + shared_col[wid][i]);
                    //     sum = myfma(shared_val[wid][PADDED_BSR_BLOCK_DIM * i + p], val_B, sum);
                    // }
                    // as unrolled loop.
                    T val_B = rocsparse_ldg(B + col + shared_col[wid][i]);
                    sum     = myfma(shared_val[wid][PADDED_BSR_BLOCK_DIM * i], val_B, sum);
                    if(BSR_BLOCK_DIM >= 2)
                    {
                        val_B = rocsparse_ldg(B + col + N * 1 + shared_col[wid][i]);
                        sum   = myfma(
                            shared_val[wid][PADDED_BSR_BLOCK_DIM * i + 1], val_B, sum);
                    }
                    if(BSR_BLOCK_DIM >= 3)
                    {
                        val_B = rocsparse_ldg(B + col + N * 2 + shared_col[wid][i]);
                        sum   = myfma(
                            shared_val[wid][PADDED_BSR_BLOCK_DIM * i + 2], val_B, sum);
                    }
                    if(BSR_BLOCK_DIM >= 4)
                    {
                        val_B = rocsparse_ldg(B + col + N * 3 + shared_col[wid][i]);
                        sum   = myfma(
                            shared_val[wid][PADDED_BSR_BLOCK_DIM * i + 3], val_B, sum);
                    }
                }
            }
        }

        if(col < N)
        {
            if(beta == static_cast<T>(0))
            {
                C[global_row + col * ldc] = alpha * sum;
            }
            else
            {
                C[global_row + col * ldc]
                    = myfma(beta, C[global_row + col * ldc], alpha * sum);
            }
        }
    }
}

template <typename T, int BSR_BLOCK_DIM, int BLK_SIZE_Y>
static __device__ void bsrmm_large_blockdim_device(cusparseDirection_t direction,
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
    int tidy = hipThreadIdx_y;

    int global_row = tidx + blockIdx.x * block_dim;
    int global_col = tidy + blockDim.y * BLK_SIZE_Y;

    int block_row = blockIdx.x;

    int block_row_start = 0;
    int block_row_end   = 0;
    if(block_row < Mb)
    {
        block_row_start = bsr_row_ptr[block_row] - idx_base;
        block_row_end   = bsr_row_ptr[block_row + 1] - idx_base;
    }

    int colB = global_col * ldb;
    int colC = global_col * ldc;

    __shared__ T shared_B[BSR_BLOCK_DIM * BLK_SIZE_Y];
    __shared__ T shared_A[BSR_BLOCK_DIM * BSR_BLOCK_DIM];

    T sum = static_cast<T>(0);

    int index         = BSR_BLOCK_DIM * tidy + tidx;
    int block_dim_sqr = block_dim * block_dim;

    for(int k = block_row_start; k < block_row_end; k++)
    {
        int block_col = (bsr_col_ind[k] - idx_base);

        if(trans_B == CUSPARSE_OPERATION_NON_TRANSPOSE)
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

        if(direction == CUSPARSE_DIRECTION_ROW)
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

        for(int j = 0; j < block_dim; j++)
        {
            sum = myfma(
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
            C[global_row + colC] = myfma(beta, C[global_row + colC], alpha * sum);
        }
    }
}

template <typename T, int BSR_BLOCK_DIM, int BLK_SIZE_Y>
static __device__ void bsrmm_general_blockdim_device(cusparseDirection_t direction,
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
    int tidy = hipThreadIdx_y;

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

    int global_col = tidy + blockDim.y * BLK_SIZE_Y;

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

                if(direction == CUSPARSE_DIRECTION_ROW)
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
                    sum = myfma(shared_A[BSR_BLOCK_DIM * j + tidx],
                                        shared_B[BSR_BLOCK_DIM * tidy + j],
                                        sum);
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
                C[global_row + colC] = myfma(beta, C[global_row + colC], alpha * sum);
            }
        }
    }
}

#endif // BSRMM_DEVICE_H