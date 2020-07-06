
__global__ void fill_kernel(double* ptr, int64_t length, double val) {
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_x = gridDim.x * blockDim.x;
    while (tx < length) {
        ptr[tx] = val;
        tx += length;
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