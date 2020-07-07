#include <cuda_runtime_api.h>
#include <cuda.h>
#include <assert.h>
#include <chrono>
#include <iostream>
#include <stdio.h>

#include "matrix.h"

static void handle_error(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(-1);
    }
}

#define HANDLE_ERROR( err ) (handle_error( err, __FILE__, __LINE__ ))

__global__ void fill_kernel(double* ptr, int64_t length, double val) {
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_x = gridDim.x * blockDim.x;
    while (tx < length) {
        ptr[tx] = val;
        tx += stride_x;
    }
}

__global__ void coo_spmm_kernel(
    int64_t* rows, int64_t* cols, double* data, double* out,
    int64_t l, int64_t m, int64_t n, int64_t nnz) {
    int64_t ty = blockIdx.y * blockDim.y + threadIdx.y, 
            tx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_y = gridDim.y * blockDim.y,
            stride_x = gridDim.x * blockDim.x;
    while (ty < nnz) {
        int64_t rid = rows[ty], cid = cols[ty];
        double* out_off = out + rid * n; 
        while (tx < n) {
            atomicAdd(out_off + tx, data[cid * n + tx]);
            tx += stride_x;
        }
        ty += stride_y;
    }
}

__global__ void csr_spmm_kernel(
    int64_t* indptr, int64_t* indices, double* data, double* out,
    int64_t l, int64_t m, int64_t n) {
    int64_t ty = blockIdx.y * blockDim.y + threadIdx.y, 
            tx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_y = gridDim.y * blockDim.y,
            stride_x = gridDim.x * blockDim.x;
    while (ty < l) {
        int64_t row_start = indptr[ty], row_end = indptr[ty + 1];
        double* out_off = out + ty * n;
        while (tx < n) {
            double acc = 0;
            for (int64_t i = row_start; i < row_end; ++i) {
                int64_t cid = indices[i];
                acc += data[cid * n + tx];
            }
            out_off[tx] = acc;
            tx += stride_x;
        }
        ty += stride_y;
    }
}

void coo_spmm(const COOMatrix& coo, const DenseMatrix& dense, DenseMatrix& out) {
    assert(coo.num_cols == dense.num_rows);
    assert(coo.num_rows == out.num_rows);
    assert(dense.num_cols == out.num_cols);

    int64_t *coo_row, *coo_col;
    double *dense_data, *out_data;

    int64_t coo_bytes = coo.nnz * sizeof(int64_t),
            dense_bytes = dense.num_rows * dense.num_cols * sizeof(double),
            out_bytes = out.num_rows * out.num_cols * sizeof(double);
    
    cudaMalloc((void**)&coo_row, coo_bytes);
    cudaMalloc((void**)&coo_col, coo_bytes);
    cudaMalloc((void**)&dense_data, dense_bytes);
    cudaMalloc((void**)&out_data, out_bytes);

    cudaMemcpy((void*)coo_row, (void*)(coo.row), coo_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)coo_col, (void*)(coo.col), coo_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)dense_data, (void*)(dense.data), dense_bytes, cudaMemcpyHostToDevice);

    fill_kernel<<<(out.num_rows * out.num_cols + 511) / 512, 512>>>(out_data, out.num_rows * out.num_cols, 0);
    
    dim3 block_size(32, 32);
    dim3 grid_size((out.num_cols + block_size.x - 1) / block_size.x, (coo.nnz + block_size.y - 1) / block_size.y);

    float time;
    cudaEvent_t start, stop;
    
    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    HANDLE_ERROR( cudaEventRecord(start, 0) );

    coo_spmm_kernel<<<grid_size, block_size>>>(coo_row, coo_col, dense_data, out_data, 
                                               coo.num_rows, coo.num_cols, dense.num_cols, coo.nnz);

    HANDLE_ERROR( cudaEventRecord(stop, 0) );
    HANDLE_ERROR( cudaEventSynchronize(stop) );
    HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
                                               
    printf("coo cost time:  %3.10f ms \n", time);                                               
                                               

    cudaMemcpy((void*)out.data, (void*)out_data, out_bytes, cudaMemcpyDeviceToHost);    
    
    cudaFree(coo_row);
    cudaFree(coo_col);
    cudaFree(dense_data);
    cudaFree(out_data);
}

void csr_spmm(const CSRMatrix& csr, const DenseMatrix& dense, DenseMatrix& out) {
    assert(csr.num_cols == dense.num_rows);
    assert(csr.num_rows == out.num_rows);
    assert(dense.num_cols == out.num_cols);

    int64_t *csr_indptr, *csr_indices;
    double *dense_data, *out_data;

    int64_t indptr_bytes = (csr.num_rows + 1) * sizeof(int64_t),
            indices_bytes = csr.indptr[csr.num_rows] * sizeof(int64_t),
            dense_bytes = dense.num_rows * dense.num_cols * sizeof(double),
            out_bytes = out.num_rows * out.num_cols * sizeof(double);
    
    cudaMalloc((void**)&csr_indptr, indptr_bytes);
    cudaMalloc((void**)&csr_indices, indices_bytes);
    cudaMalloc((void**)&dense_data, dense_bytes);
    cudaMalloc((void**)&out_data, out_bytes);

    cudaMemcpy((void*)csr_indptr, (void*)(csr.indptr), indptr_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)csr_indices, (void*)(csr.indices), indices_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)dense_data, (void*)(dense.data), dense_bytes, cudaMemcpyHostToDevice);



    dim3 block_size(32, 32);
    dim3 grid_size((out.num_cols + block_size.x - 1) / block_size.x, (csr.num_rows + block_size.y - 1) / block_size.y);

    float time;
    cudaEvent_t start, stop;
    
    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    HANDLE_ERROR( cudaEventRecord(start, 0) );

    csr_spmm_kernel<<<grid_size, block_size>>>(csr_indptr, csr_indices, dense_data, out_data,
                                               csr.num_rows, csr.num_cols, dense.num_cols);
    
    HANDLE_ERROR( cudaEventRecord(stop, 0) );
    HANDLE_ERROR( cudaEventSynchronize(stop) );
    HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
                                               
    printf("csr cost time:  %3.10f ms \n", time);

    cudaMemcpy((void*)out.data, (void*)out_data, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(csr_indptr);
    cudaFree(csr_indices);
    cudaFree(dense_data);
    cudaFree(out_data);
}

void test_small_coo_spmm() {
    int64_t l = 2, m = 3, n = 3, nnz = 3;
    COOMatrix coo(l, m, nnz, vec2ptr({0, 1, 1}), vec2ptr({1, 0, 2}));
    DenseMatrix dense(m, n, vec2ptr({{3, 9, 2}, {4, 6, 7}, {5, 8 , 1}}));
    DenseMatrix out(l, n, new double[l * n]());
    coo_spmm(coo, dense, out);
    out.dump();
}

void test_small_csr_spmm() {
    int64_t l = 2, m = 3, n = 3;
    CSRMatrix csr(l, m, vec2ptr({0, 1, 3}), vec2ptr({1, 0, 2}));
    DenseMatrix dense(m, n, vec2ptr({{3, 9, 2}, {4, 6, 7}, {5, 8 , 1}}));
    DenseMatrix out(l, n, new double[l * n]());
    csr_spmm(csr, dense, out);
    out.dump();
}

void test_large_coo_spmm(int64_t l, int64_t m, int64_t n) {
    COOMatrix coo = random_coo(l, m);
    DenseMatrix dense = random_dense(m, n);
    DenseMatrix out(l, n, new double[l * n]());
    std::cout << "nnz" << coo.nnz << std::endl;
    // auto start_time = std::chrono::high_resolution_clock::now();
    coo_spmm(coo, dense, out);
    // auto end_time = std::chrono::high_resolution_clock::now();
    // double duration = std::chrono::duration<double>(end_time - start_time).count();
    // std::cout << "coo_spmm time cost: " << duration << "s" << std::endl;
}

void test_large_csr_spmm(int64_t l, int64_t m, int64_t n) {
    CSRMatrix csr = random_csr(l, m);
    DenseMatrix dense = random_dense(m, n);
    DenseMatrix out(l, n, new double[l * n]());
    // auto start_time = std::chrono::high_resolution_clock::now();
    csr_spmm(csr, dense, out);
    // auto end_time = std::chrono::high_resolution_clock::now();
    // double duration = std::chrono::duration<double>(end_time - start_time).count();
    // std::cout << "csr_spmm time cost: " << duration << "s" << std::endl;
}

int main() {
    test_small_coo_spmm();
    test_small_csr_spmm();

    test_large_coo_spmm(5000, 5500, 5200);
    test_large_csr_spmm(5000, 5500, 5200);

    return 0;
}
