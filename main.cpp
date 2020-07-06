#include <stdint.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>
#include <omp.h>

std::mt19937_64 gen(1234);

struct DenseMatrix {
    int64_t num_rows, num_cols;
    double *data;

    DenseMatrix(int64_t rows, int64_t cols, double *p) :
        num_rows(rows), num_cols(cols), data(p) {}

    ~DenseMatrix() {
        delete[] data;
    }

    void dump() {
        for (int64_t i = 0; i < num_rows; ++i) {
            for (int64_t j = 0; j < num_cols; ++j) {
                std::cout << data[i * num_cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

struct CSRMatrix {
    int64_t num_rows, num_cols;
    int64_t *indptr;
    int64_t *indices;

    CSRMatrix(int64_t rows, int64_t cols, int64_t *p1, int64_t *p2) :
        num_rows(rows), num_cols(cols), indptr(p1), indices(p2) {}

    ~CSRMatrix() {
        delete[] indptr;
        delete[] indices;
    }
};

struct COOMatrix {
    int64_t num_rows, num_cols, nnz;
    int64_t *row;
    int64_t *col;

    COOMatrix(int64_t rows, int64_t cols, int64_t nnz, int64_t *r, int64_t *c) :
        num_rows(rows), num_cols(cols), nnz(nnz), row(r), col(c) {}

    ~COOMatrix() {
        delete[] row;
        delete[] col;
    }
};

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

// in order to use initialization list
int64_t* vec2ptr(std::vector<int64_t> v) {
    int64_t *ptr = new int64_t[v.size()]();
    for (int64_t i = 0; i < v.size(); ++i) {
        ptr[i] = v[i];
    }
    return ptr;
}

double* vec2ptr(std::vector<std::vector<double>> v) {
    int64_t num_rows = v.size(), num_cols = v[0].size();
    double *ptr = new double[num_rows * num_cols]();
    for (int64_t i = 0; i < num_rows; ++i) {
        for (int64_t j = 0; j < num_cols; ++j) {
            ptr[i * num_cols + j] = v[i][j];
        }
    }
    return ptr;
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

CSRMatrix random_csr(int64_t m, int64_t n, double p = 0.2) {
    std::uniform_real_distribution<> dist(0, 1);

    int64_t *indptr = new int64_t[m + 1]();
    std::vector<int64_t> indices_vec;
    indptr[0] = 0;
    for (int64_t i = 1; i <= m; ++i) {
        int64_t cnt = 0;
        for (int64_t j = 0; j < n; ++j) {
            if (dist(gen) < p) {
                indices_vec.push_back(j);
                ++cnt;
            }
        }
        indptr[i] = indptr[i - 1] + cnt;
    }

    return CSRMatrix(m, n, indptr, vec2ptr(std::move(indices_vec)));
}

COOMatrix random_coo(int64_t m, int64_t n, double p = 0.2) {
    std::uniform_real_distribution<> dist(0, 1);

    int64_t cnt = 0;
    std::vector<std::pair<int64_t, int64_t>> vec;
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            if (dist(gen) < p) {
                vec.emplace_back(i, j);
                ++cnt;
            }
        }
    }
    std::shuffle(vec.begin(), vec.end(), gen);
    std::vector<int64_t> row_vec, col_vec;
    for (auto&& it : vec) {
        row_vec.push_back(it.first);
        col_vec.push_back(it.second);
    }

    return COOMatrix(m, n, cnt, vec2ptr(std::move(row_vec)), vec2ptr(std::move(col_vec)));
}

DenseMatrix random_dense(int64_t m, int64_t n) {
    std::uniform_real_distribution<> dist(-100, 100);

    int64_t sz = m * n;
    double *ptr = new double[sz]();
    for (int64_t i = 0; i < sz; ++i) {
        ptr[i] = dist(gen);
    }

    return DenseMatrix(m, n, ptr);
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