#include <stdint.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <omp.h>

#include "matrix.h"

std::mt19937_64 gen(1234);

void DenseMatrix::dump() {
    for (int64_t i = 0; i < num_rows; ++i) {
        for (int64_t j = 0; j < num_cols; ++j) {
            std::cout << data[i * num_cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

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

CSRMatrix random_csr(int64_t m, int64_t n, double p) {
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

COOMatrix random_coo(int64_t m, int64_t n, double p) {
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

