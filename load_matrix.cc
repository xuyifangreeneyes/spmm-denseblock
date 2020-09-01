#include <stdlib.h>
#include <fstream>
#include <random>
#include <assert.h>
#include "utility.h"

static std::mt19937_64 gen(1234);

int randomCSRMatrix(int m, int n, float p, int** hostCsrRowPtr, int** hostCsrColInd, float** hostCsrVal, float minVal, float maxVal) {
    std::uniform_real_distribution<float> flip(0, 1), dist(minVal, maxVal);
    *hostCsrRowPtr = (int*) malloc((m + 1) * sizeof(int));
    int cnt = 0;
    (*hostCsrRowPtr)[0] = cnt;
    std::vector<int> indices;
    std::vector<float> vals;
    for (int i = 1; i <= m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (flip(gen) < p) {
                indices.push_back(j);
                vals.push_back(dist(gen));
                ++cnt;
            }
        }
        (*hostCsrRowPtr)[i] = cnt;
        // if (i % 1000 == 0) {
        //     printf("i = %d\n", i);
        // }
    }
    *hostCsrColInd = vec2ptr(indices);
    *hostCsrVal = vec2ptr(vals);

    // Generating random CSR matrix is time-consuming, so we record it for next time use.
    // std::fstream s1("csr_indptr.txt", std::ios::out | std::ios::trunc);
    // std::fstream s2("csr_indices.txt", std::ios::out | std::ios::trunc);

    // s1 << m + 1 << std::endl;
    // for (int i = 0; i <= m; ++i) {
    //     s1 << (*hostCsrRowPtr)[i] << " ";
    // }
    // s1 << std::endl;

    // s2 << cnt << std::endl;
    // for (int i = 0; i < cnt; ++i) {
    //     s2 << (*hostCsrColInd)[i] << " ";
    // }
    // s2 << std::endl;

    return cnt;
}

int readAndFillCSRMatrix(int m, int n, int** hostCsrRowPtr, int** hostCsrColInd, float** hostCsrVal, float minVal, float maxVal) {
    std::fstream s1("csr_indptr.txt", std::ios::in);
    std::fstream s2("csr_indices.txt", std::ios::in);

    int xx;
    s1 >> xx;
    assert(m + 1 == xx);
    *hostCsrRowPtr = (int*) malloc((m + 1) * sizeof(int));
    for (int i = 0; i <= m; ++i) {
        s1 >> (*hostCsrRowPtr)[i];
    }

    int nnz;
    s2 >> nnz;
    *hostCsrColInd = (int*) malloc(nnz * sizeof(int));
    for (int i = 0; i < nnz; ++i) {
        s2 >> (*hostCsrColInd)[i];
    }

    *hostCsrVal = (float*) malloc(nnz * sizeof(float));
    std::uniform_real_distribution<float> dist(minVal, maxVal);
    for (int i = 0; i < nnz; ++i) {
        (*hostCsrVal)[i] = dist(gen);
    }

    return nnz;
}

float* randomDenseMatrix(int n, int dim, float minVal, float maxVal) {
    std::uniform_real_distribution<float> dist(minVal, maxVal);
    int sz = n * dim;
    float* ptr = (float*) malloc(sz * sizeof(float));
    for (int i = 0; i < sz; ++i) {
        ptr[i] = dist(gen);
    }
    return ptr;
}