#ifndef ROCSPARSE_BSRMM_H
#define ROCSPARSE_BSRMM_H

#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include "rocsparse_bsrmm_impl.h"

#define launch_bsrmmnn_small_blockdim_kernel(T, block_size, wf_size,         \
                                             bsr_block_dim)                  \
  bsrmmnn_small_blockdim_kernel<                                             \
      T, block_size, wf_size,                                                \
      bsr_block_dim><<<bsrmmnn_blocks, bsrmmnn_threads>>>(                   \
      dir, mb, n, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, B, ldb, beta, C, \
      ldc, idx_base);

#define launch_bsrmmnt_small_blockdim_kernel(T, block_size, wf_size,         \
                                             bsr_block_dim)                  \
  bsrmmnt_small_blockdim_kernel<                                             \
      T, block_size, wf_size,                                                \
      bsr_block_dim><<<bsrmmnt_blocks, bsrmmnt_threads>>>(                   \
      dir, mb, n, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, B, ldb, beta, C, \
      ldc, idx_base);

#define launch_bsrmm_large_blockdim_kernel(T, bsr_block_dim, blk_size_y)    \
  bsrmm_large_blockdim_kernel<T, bsr_block_dim,                             \
                              blk_size_y><<<bsrmm_blocks, bsrmm_threads>>>( \
      dir, trans_B, mb, n, alpha, bsr_row_ptr, bsr_col_ind, bsr_val,        \
      block_dim, B, ldb, beta, C, ldc, idx_base);

#define launch_bsrmm_general_blockdim_kernel(T, bsr_block_dim, blk_size_y)    \
  bsrmm_general_blockdim_kernel<T, bsr_block_dim,                             \
                                blk_size_y><<<bsrmm_blocks, bsrmm_threads>>>( \
      dir, trans_B, mb, n, alpha, bsr_row_ptr, bsr_col_ind, bsr_val,          \
      block_dim, B, ldb, beta, C, ldc, idx_base);

template <typename T, int BLOCKSIZE, int WF_SIZE, int BSR_BLOCK_DIM>
__launch_bounds__(BLOCKSIZE) __global__ void bsrmmnn_small_blockdim_kernel(
    cusparseDirection_t direction, int mb, int n, T alpha,
    const int* __restrict__ bsr_row_ptr, const int* __restrict__ bsr_col_ind,
    const T* __restrict__ bsr_val, const T* __restrict__ B, int ldb, T beta,
    T* __restrict__ C, int ldc, int idx_base) {
  if (alpha == static_cast<T>(0) && beta == static_cast<T>(1)) {
    return;
  }

  bsrmmnn_small_blockdim_device<T, BLOCKSIZE, WF_SIZE, BSR_BLOCK_DIM>(
      direction, mb, n, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, B, ldb, beta,
      C, ldc, idx_base);
}

template <typename T, int BLOCKSIZE, int WF_SIZE, int BSR_BLOCK_DIM>
__launch_bounds__(BLOCKSIZE) __global__ void bsrmmnt_small_blockdim_kernel(
    cusparseDirection_t direction, int mb, int n, T alpha,
    const int* __restrict__ bsr_row_ptr, const int* __restrict__ bsr_col_ind,
    const T* __restrict__ bsr_val, const T* __restrict__ B, int ldb, T beta,
    T* __restrict__ C, int ldc, int idx_base) {
  if (alpha == static_cast<T>(0) && beta == static_cast<T>(1)) {
    return;
  }

  bsrmmnt_small_blockdim_device<T, BLOCKSIZE, WF_SIZE, BSR_BLOCK_DIM>(
      direction, mb, n, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, B, ldb, beta,
      C, ldc, idx_base);
}

template <typename T, int BSR_BLOCK_DIM, int BLK_SIZE_Y>
__launch_bounds__(BSR_BLOCK_DIM* BLK_SIZE_Y) __global__
    void bsrmm_large_blockdim_kernel(
        cusparseDirection_t direction, cusparseOperation_t trans_B, int mb,
        int n, T alpha, const int* __restrict__ bsr_row_ptr,
        const int* __restrict__ bsr_col_ind, const T* __restrict__ bsr_val,
        int block_dim, const T* __restrict__ B, int ldb, T beta,
        T* __restrict__ C, int ldc, int idx_base) {
  if (alpha == static_cast<T>(0) && beta == static_cast<T>(1)) {
    return;
  }

  bsrmm_large_blockdim_device<T, BSR_BLOCK_DIM, BLK_SIZE_Y>(
      direction, trans_B, mb, n, alpha, bsr_row_ptr, bsr_col_ind, bsr_val,
      block_dim, B, ldb, beta, C, ldc, idx_base);
}

template <typename T, int BSR_BLOCK_DIM, int BLK_SIZE_Y>
__launch_bounds__(BSR_BLOCK_DIM* BLK_SIZE_Y) __global__
    void bsrmm_general_blockdim_kernel(
        cusparseDirection_t direction, cusparseOperation_t trans_B, int mb,
        int n, T alpha, const int* __restrict__ bsr_row_ptr,
        const int* __restrict__ bsr_col_ind, const T* __restrict__ bsr_val,
        int block_dim, const T* __restrict__ B, int ldb, T beta,
        T* __restrict__ C, int ldc, int idx_base) {
  if (alpha == static_cast<T>(0) && beta == static_cast<T>(1)) {
    return;
  }

  bsrmm_general_blockdim_device<T, BSR_BLOCK_DIM, BLK_SIZE_Y>(
      direction, trans_B, mb, n, alpha, bsr_row_ptr, bsr_col_ind, bsr_val,
      block_dim, B, ldb, beta, C, ldc, idx_base);
}

template <typename T>
cusparseStatus_t rocsparse_bsrmm_template(
    cusparseHandle_t handle, cusparseDirection_t dir,
    cusparseOperation_t trans_A, cusparseOperation_t trans_B, int mb, int n,
    int kb, int nnzb, T alpha, const cusparseMatDescr_t descr, const T* bsr_val,
    const int* bsr_row_ptr, const int* bsr_col_ind, int block_dim, T* B,
    int ldb, T beta, T* C, int ldc) {
  // Check for valid handle and matrix descriptor
  if (handle == nullptr) {
    return CUSPARSE_STATUS_NOT_INITIALIZED;
  } else if (descr == nullptr) {
    return CUSPARSE_STATUS_INVALID_VALUE;  // TODO
  }

  // Check index base
  // if(descr.IndexBase != CUSPARSE_INDEX_BASE_ZERO && descr.IndexBase !=
  // CUSPARSE_INDEX_BASE_ONE)
  // {
  //     return CUSPARSE_STATUS_INVALID_VALUE;
  // }

  int idx_base = 0;
  // if(descr.IndexBase == CUSPARSE_INDEX_BASE_ONE)
  // {
  //     idx_base = 1;
  // }

  // Check matrix type
  // if(descr->MatrixType != CUSPARSE_MATRIX_TYPE_GENERAL)
  // {
  //     // TODO
  //     return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
  // }

  // Check operation
  if (trans_A != CUSPARSE_OPERATION_NON_TRANSPOSE) {
    return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
  } else if (trans_B != CUSPARSE_OPERATION_NON_TRANSPOSE &&
             trans_B != CUSPARSE_OPERATION_TRANSPOSE) {
    return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
  }

  // std::cout << "hello1" << std::endl;

  // Check sizes
  if (mb < 0 || n < 0 || kb < 0 || nnzb < 0 || block_dim <= 0) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }

  // Quick return if possible
  if (mb == 0 || n == 0 || kb == 0 || nnzb == 0) {
    return CUSPARSE_STATUS_SUCCESS;
  }

  // Check pointer arguments
  if (bsr_val == nullptr || bsr_row_ptr == nullptr || bsr_col_ind == nullptr ||
      B == nullptr || C == nullptr) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }

  // Check leading dimension of B
  if (trans_B == CUSPARSE_OPERATION_NON_TRANSPOSE) {
    if (ldb < kb) {
      return CUSPARSE_STATUS_INVALID_VALUE;
    }
  } else {
    if (ldb < n) {
      return CUSPARSE_STATUS_INVALID_VALUE;
    }
  }

  // Check leading dimension of C
  if (ldc < mb) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }

  int m = mb * block_dim;
  // int k   = kb * block_dim;
  // int nnz = nnzb * block_dim;

  if (n == 1) {
    assert(false);
  }

  if (block_dim == 1) {
    assert(false);
  }

  // std::cout << "hello" << std::endl;

  if (block_dim == 2) {
    if (trans_B == CUSPARSE_OPERATION_NON_TRANSPOSE) {
      constexpr int BSRMMNN_DIM = 64;
      constexpr int SUB_WF_SIZE = 8;

      dim3 bsrmmnn_blocks((SUB_WF_SIZE * m - 1) / BSRMMNN_DIM + 1,
                          (n - 1) / SUB_WF_SIZE + 1);
      dim3 bsrmmnn_threads(BSRMMNN_DIM);
      launch_bsrmmnn_small_blockdim_kernel(T, BSRMMNN_DIM, SUB_WF_SIZE, 2);
    } else {
      constexpr int BSRMMNT_DIM = 64;

      // Average nnzb per row of A
      int avg_row_nnzb = (nnzb - 1) / mb + 1;

      // Launch appropriate kernel depending on row nnz of A
      if (avg_row_nnzb < 16) {
        dim3 bsrmmnt_blocks((8 * m - 1) / BSRMMNT_DIM + 1);
        dim3 bsrmmnt_threads(BSRMMNT_DIM);
        launch_bsrmmnt_small_blockdim_kernel(T, BSRMMNT_DIM, 8, 2);
      } else if (avg_row_nnzb < 32) {
        dim3 bsrmmnt_blocks((16 * m - 1) / BSRMMNT_DIM + 1);
        dim3 bsrmmnt_threads(BSRMMNT_DIM);
        launch_bsrmmnt_small_blockdim_kernel(T, BSRMMNT_DIM, 16, 2);
      } else {
        dim3 bsrmmnt_blocks((32 * m - 1) / BSRMMNT_DIM + 1);
        dim3 bsrmmnt_threads(BSRMMNT_DIM);
        launch_bsrmmnt_small_blockdim_kernel(T, BSRMMNT_DIM, 32, 2);
      }
    }

    return CUSPARSE_STATUS_SUCCESS;
  }

  // Run different bsrmm kernels for block dim > 2
  if (n <= 16 && block_dim > 4 && block_dim <= 8) {
    dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / 16 + 1);
    dim3 bsrmm_threads(8, 16, 1);
    launch_bsrmm_large_blockdim_kernel(T, 8, 16);
  } else {
    if (block_dim <= 4) {
      dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / 16 + 1);
      dim3 bsrmm_threads(4, 16, 1);
      launch_bsrmm_large_blockdim_kernel(T, 4, 16);
    } else if (block_dim <= 8) {
      dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / 32 + 1);
      dim3 bsrmm_threads(8, 32, 1);
      launch_bsrmm_large_blockdim_kernel(T, 8, 32);
    } else if (block_dim <= 16) {
      dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / 16 + 1);
      dim3 bsrmm_threads(16, 16, 1);
      launch_bsrmm_large_blockdim_kernel(T, 16, 16);
    } else if (block_dim <= 32) {
      dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / 32 + 1);
      dim3 bsrmm_threads(32, 32, 1);
      launch_bsrmm_large_blockdim_kernel(T, 32, 32);
    } else {
      dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / 32 + 1);
      dim3 bsrmm_threads(32, 32, 1);
      launch_bsrmm_general_blockdim_kernel(T, 32, 32);
    }
  }

  return CUSPARSE_STATUS_SUCCESS;
}

#endif