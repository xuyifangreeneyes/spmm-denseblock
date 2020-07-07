#include <chrono>
#include <assert.h>
#include <cstring>

#include "matrix.h"

void csr_spmm(const CSRMatrix& csr, const DenseMatrix& dense, DenseMatrix& out) {
    assert(csr.num_cols == dense.num_rows);
    assert(csr.num_rows == out.num_rows);
    assert(dense.num_cols == out.num_cols);
#pragma omp parallel for
    for (int64_t rid = 0; rid < csr.num_rows; ++rid) {
        int64_t row_start = csr.indptr[rid], row_end = csr.indptr[rid + 1];
        double* out_off = out.data + rid * out.num_cols;
        for (int64_t k = 0; k < out.num_cols; ++k) {
            double acc = 0;
            for (int64_t j = row_start; j < row_end; ++j) {
                int64_t cid = csr.indices[j];
                acc += dense.data[cid * dense.num_cols + k];
            }
            out_off[k] = acc;
        }
    }
}

void coo_spmm(const COOMatrix& coo, const DenseMatrix& dense, DenseMatrix& out) {
    assert(coo.num_cols == dense.num_rows);
    assert(coo.num_rows == out.num_rows);
    assert(dense.num_cols == out.num_cols);
    memset(out.data, 0, out.num_rows * out.num_cols * sizeof(double));
#pragma omp parallel for
    for (int64_t i = 0; i < coo.nnz; ++i) {
        int64_t rid = coo.row[i];
        int64_t cid = coo.col[i];
        double* out_off = out.data + rid * out.num_cols;
        for (int64_t k = 0; k < out.num_cols; ++k) {
#pragma omp atomic
            out_off[k] += dense.data[cid * dense.num_cols + k];
        }
    }
}

void test_small_csr_spmm() {
    int64_t l = 2, m = 3, n = 3;
    CSRMatrix csr(l, m, vec2ptr({0, 1, 3}), vec2ptr({1, 0, 2}));
    DenseMatrix dense(m, n, vec2ptr({{3, 9, 2}, {4, 6, 7}, {5, 8 , 1}}));
    DenseMatrix out(l, n, new double[l * n]());
    csr_spmm(csr, dense, out);
    out.dump();
}

void test_small_coo_spmm() {
    int64_t l = 2, m = 3, n = 3, nnz = 3;
    COOMatrix coo(l, m, nnz, vec2ptr({0, 1, 1}), vec2ptr({1, 0, 2}));
    DenseMatrix dense(m, n, vec2ptr({{3, 9, 2}, {4, 6, 7}, {5, 8 , 1}}));
    DenseMatrix out(l, n, new double[l * n]());
    coo_spmm(coo, dense, out);
    out.dump();
}

void test_large_csr_spmm(int64_t l, int64_t m, int64_t n) {
    CSRMatrix csr = random_csr(l, m);
    DenseMatrix dense = random_dense(m, n);
    DenseMatrix out(l, n, new double[l * n]());
    auto start_time = std::chrono::high_resolution_clock::now();
    csr_spmm(csr, dense, out);
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "csr_spmm time cost: " << duration << "s" << std::endl;
}

void test_large_coo_spmm(int64_t l, int64_t m, int64_t n) {
    COOMatrix coo = random_coo(l, m);
    DenseMatrix dense = random_dense(m, n);
    DenseMatrix out(l, n, new double[l * n]());
    auto start_time = std::chrono::high_resolution_clock::now();
    coo_spmm(coo, dense, out);
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "coo_spmm time cost: " << duration << "s" << std::endl;
}

int main() {
    // test_small_csr_spmm();
    // test_small_coo_spmm();
    test_large_csr_spmm(1000, 1500, 1200);
    test_large_coo_spmm(1000, 1500, 1200);
    
    return 0;
}