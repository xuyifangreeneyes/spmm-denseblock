#include <stdint.h>
#include <vector>


struct DenseMatrix {
    int64_t num_rows, num_cols;
    double *data;

    DenseMatrix(int64_t rows, int64_t cols, double *p) :
        num_rows(rows), num_cols(cols), data(p) {}

    ~DenseMatrix() {
        delete[] data;
    }

    void dump();
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

// in order to use initialization list
int64_t* vec2ptr(std::vector<int64_t> v);

double* vec2ptr(std::vector<std::vector<double>> v);

CSRMatrix random_csr(int64_t m, int64_t n, double p = 0.2);

COOMatrix random_coo(int64_t m, int64_t n, double p = 0.2);

DenseMatrix random_dense(int64_t m, int64_t n);

