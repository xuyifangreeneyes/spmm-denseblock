#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <random>
#include <assert.h>
#include <cuda_runtime.h>
#include "cusparse.h"

std::mt19937_64 gen(1234);

std::vector<std::vector<int>> csr2adj(const std::string& indptr_file, 
                                      const std::string& indices_file, int n, int nnz) {

    std::fstream s1(indptr_file, std::ios::in);
    std::fstream s2(indices_file, std::ios::in);

    std::vector<int> indptr, indices;

    int xx;
    s1 >> xx;
    assert(xx == n + 1);
    for (int i = 0; i <= n; ++i) {
        s1 >> xx;
        indptr.push_back(xx);
    }
    assert(indptr[n] - indptr[0] == nnz);
    s2 >> xx;
    assert(xx == nnz);
    for (int i = 0; i < nnz; ++i) {
        s2 >> xx;
        indices.push_back(xx);
    }

    assert(indptr[0] == 0);
    std::vector<std::vector<int>> edges(n);
    for (int i = 0; i < n; ++i) {
        int start = indptr[i], end = indptr[i + 1];
        for (int j = start; j < end; ++j) {
            edges[i].push_back(indices[j]);
        }
    }

    return edges;
}

void divide_matrix(const std::vector<std::vector<int>>& edges,
                   std::vector<int>& csr_row_ptr, std::vector<int>& csr_col_ind,
                   std::vector<int>& bsr_row_ptr, std::vector<int>& bsr_col_ind,
                   std::vector<float>& bsr_val, int n, int bsize, float density) {
    int nb = (n + bsize - 1) / bsize;
    int bnum = bsize * bsize;
    std::vector<int> counts(nb, 0);
    std::vector<int> flags(nb, -1);
    
    csr_row_ptr.push_back(0);
    bsr_row_ptr.push_back(0);

    for (int x1 = 0; x1 < nb; ++x1) {
        
        std::fill(counts.begin(), counts.end(), 0);
        std::fill(flags.begin(), flags.end(), -1);
        
        for (int x2 = 0; x2 < bsize; ++x2) {
            int x = x1 * bsize + x2;
            if (x >= n) {
                break;
            }
            const std::vector<int>& ys = edges[x];
            // for (int y : ys) {
            for (int i = 0; i < ys.size(); ++i) {
                if (i >= 1 && ys[i] == ys[i - 1]) {
                    std::cout << "????" << std::endl;
                }    
                int y = ys[i];
                ++counts[y / bsize];
            }
        }
        
        int bsr_cnt = 0;
        for (int i = 0; i < nb; ++i) {
            float occupy = (counts[i] * 1.0) / bnum;
            if (counts[i] > bnum) {
                std::cout << counts[i] << " oooops" << std::endl;
            }
            if (occupy >= density) {
                bsr_col_ind.push_back(i);
                flags[i] = bsr_cnt;
                ++bsr_cnt;
            }
        }
        
        int bsr_row_val = bsr_row_ptr.back() + bsr_cnt;
        bsr_row_ptr.push_back(bsr_row_val);

        std::vector<float> vals(bsr_cnt * bnum, 0);
        for (int x2 = 0; x2 < bsize; ++x2) {
            int x = x1 * bsize + x2;
            if (x >= n) {
                break;
            }
            int csr_cnt = 0;
            const std::vector<int>& ys = edges[x];
            for (int y : ys) {
                int y1 = y / bsize, y2 = y % bsize;
                int ith = flags[y1];
                if (ith == -1) {
                    csr_col_ind.push_back(y);
                    ++csr_cnt;
                } else {
                    vals[ith * bnum + x2 * bsize + y2] = 1;
                }
            }
            int csr_row_val = csr_row_ptr.back() + csr_cnt;
            csr_row_ptr.push_back(csr_row_val);
        }

        for (float x : vals) {
            bsr_val.push_back(x);
        }
    }
}

template <class T>
void dump_vec(const std::vector<T>& vec, const std::string& name) {
    std::cout << name << ": ";
    for (T x : vec) {
        std::cout << x << " ";
    }
    std::cout << std::endl;
}

void test_divide_matrix() {
    int n = 4, bsize = 2;
    float density = 0.6;
    std::vector<std::vector<int>> edges = { {1, }, {2, }, {1, 3}, {0, 1, 2} };
    std::vector<int> csr_row_ptr, csr_col_ind, bsr_row_ptr, bsr_col_ind;
    std::vector<float> bsr_val;
    
    divide_matrix(edges, csr_row_ptr, csr_col_ind, bsr_row_ptr, bsr_col_ind,
                  bsr_val, n, bsize, density);
    
    dump_vec(csr_row_ptr, "csr_row_ptr");
    dump_vec(csr_col_ind, "csr_col_ind");

    dump_vec(bsr_row_ptr, "bsr_row_ptr");
    dump_vec(bsr_col_ind, "bsr_col_ind");
    dump_vec(bsr_val, "bsr_val");
}

template<typename T>
T* vec2ptr(std::vector<T> v) {
    T* ptr = (T*) malloc(v.size() * sizeof(T));
    for (size_t i = 0; i < v.size(); ++i) {
        ptr[i] = v[i];
    }
    return ptr;
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
    if (hostCsrRowPtr) free(hostCsrRowPtr); \
    if (hostCsrColInd) free(hostCsrColInd); \
    if (hostCsrVal) free(hostCsrVal); \
    if (hostBsrRowPtr) free(hostBsrRowPtr); \
    if (hostBsrColInd) free(hostBsrColInd); \
    if (hostBsrVal) free(hostBsrVal); \
    if (yHostPtr) free(yHostPtr); \
    if (zHostPtr) free(zHostPtr); \
    if (csrRowPtr) cudaFree(csrRowPtr); \
    if (csrColInd) cudaFree(csrColInd); \
    if (csrVal) cudaFree(csrVal); \
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
    CLEANUP(""); \
    exit(-1); \
}

#define HANDLE_CUSPARSE_ERROR( err ) \
if (err != CUSPARSE_STATUS_SUCCESS) { \
    printf("error occurred in %s at line %d\n", __FILE__, __LINE__); \
    CLEANUP(""); \
    exit(-1); \
}

int main() {
    int n = 235868;
    int nnz = 2358104;
    int bsize = 16;
    int bnum = bsize * bsize;
    int nb = (n + bsize - 1) / bsize;
    int n1 = nb * bsize;
    assert(n1 >= n);
    int dim = 64;
    float density = 2.0;
    float alpha = 1.0;
    float beta = 1.0;
    std::string indptr_file = "collab_gpmetis2048_rcmk_indptr.txt";
    std::string indices_file = "collab_gpmetis2048_rcmk_indices.txt";

    std::cout << "csr to adj..." << std::endl;

    std::vector<std::vector<int>> edges = csr2adj(indptr_file, indices_file, n, nnz);

    std::vector<int> csr_row_ptr, csr_col_ind, bsr_row_ptr, bsr_col_ind;
    std::vector<float> bsr_val;

    std::cout << "divide matrix..." << std::endl;

    divide_matrix(edges, csr_row_ptr, csr_col_ind, bsr_row_ptr, bsr_col_ind,
                  bsr_val, n, bsize, density);

    assert(csr_row_ptr.size() == n + 1);
    assert(csr_row_ptr[0] == 0);
    assert(csr_row_ptr[n] == csr_col_ind.size());
    assert(bsr_row_ptr.size() == nb + 1);
    assert(bsr_row_ptr[0] == 0);
    assert(bsr_row_ptr[nb] == bsr_col_ind.size());
    assert(bsr_col_ind.size() * bnum == bsr_val.size());

    int csrNnz = csr_col_ind.size();
    int bsrNnzb = bsr_col_ind.size();

    printf("csr nnz = %d    bsr nnzb = %d\n", csrNnz, bsrNnzb);

    int* hostCsrRowPtr = 0;
    int* hostCsrColInd = 0;
    float* hostCsrVal = 0;
    int* csrRowPtr = 0;
    int* csrColInd = 0;
    float* csrVal = 0;

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

    std::cout << "vec2ptr..." << std::endl;

    hostCsrRowPtr = vec2ptr<int>(std::move(csr_row_ptr));
    hostCsrColInd = vec2ptr<int>(std::move(csr_col_ind));
    hostCsrVal = (float*) malloc(csrNnz * sizeof(float));
    for (int i = 0; i < csrNnz; ++i) {
        hostCsrVal[i] = 1;
    }

    hostBsrRowPtr = vec2ptr<int>(std::move(bsr_row_ptr));
    hostBsrColInd = vec2ptr<int>(std::move(bsr_col_ind));
    hostBsrVal = vec2ptr<float>(std::move(bsr_val));

    std::cout << "gpu memory malloc and memcpy..." << std::endl;

    HANDLE_ERROR( cudaMalloc((void**)&csrRowPtr, (n + 1) * sizeof(int)) );
    HANDLE_ERROR( cudaMalloc((void**)&csrColInd, csrNnz * sizeof(int)) );
    HANDLE_ERROR( cudaMalloc((void**)&csrVal, csrNnz * sizeof(float)) );  

    HANDLE_ERROR( cudaMalloc((void**)&bsrRowPtr, (nb + 1) * sizeof(int)) );
    HANDLE_ERROR( cudaMalloc((void**)&bsrColInd, bsrNnzb * sizeof(int)) );
    HANDLE_ERROR( cudaMalloc((void**)&bsrVal, bsrNnzb * bnum * sizeof(float)) );  

    HANDLE_ERROR( cudaMemcpy(csrRowPtr, hostCsrRowPtr, (size_t)((n + 1) * sizeof(int)), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(csrColInd, hostCsrColInd, (size_t)(csrNnz * sizeof(int)), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(csrVal, hostCsrVal, (size_t)(csrNnz * sizeof(float)), cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMemcpy(bsrRowPtr, hostBsrRowPtr, (size_t)((nb + 1) * sizeof(int)), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(bsrColInd, hostBsrColInd, (size_t)(bsrNnzb * sizeof(int)), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(bsrVal, hostBsrVal, (size_t)(bsrNnzb * bnum * sizeof(float)), cudaMemcpyHostToDevice) );

    std::cout << "prepare y and z..." << std::endl;

    yHostPtr = randomDenseMatrix(n1, dim);
    zHostPtr = (float*) malloc(n1 * dim * sizeof(float));

    HANDLE_ERROR( cudaMalloc((void**)&y, n1 * dim * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void**)&z, n1 * dim * sizeof(float)) );

    HANDLE_ERROR( cudaMemcpy(y, yHostPtr, (size_t)(n1 * dim * sizeof(float)), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemset((void*)z, 0, n1 * dim * sizeof(float)) );

    cusparseHandle_t handle = 0;
    cusparseMatDescr_t csrDescr = 0, bsrDescr = 0;

    HANDLE_CUSPARSE_ERROR( cusparseCreate(&handle) );

    HANDLE_CUSPARSE_ERROR( cusparseCreateMatDescr(&csrDescr) );
    cusparseSetMatType(csrDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(csrDescr, CUSPARSE_INDEX_BASE_ZERO);

    HANDLE_CUSPARSE_ERROR( cusparseCreateMatDescr(&bsrDescr) );
    cusparseSetMatType(bsrDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(bsrDescr, CUSPARSE_INDEX_BASE_ZERO);

    float time1, time2;
    cudaEvent_t start1, stop1, start2, stop2;
    HANDLE_ERROR( cudaEventCreate(&start1) );
    HANDLE_ERROR( cudaEventCreate(&stop1) );
    HANDLE_ERROR( cudaEventRecord(start1, 0) );

    HANDLE_CUSPARSE_ERROR( cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                           CUSPARSE_OPERATION_TRANSPOSE, n, dim, n, csrNnz, 
                                           &alpha, csrDescr, csrVal, csrRowPtr, csrColInd, 
                                           y, dim, &beta, z, n1) );

    HANDLE_ERROR( cudaEventRecord(stop1, 0) );
    HANDLE_ERROR( cudaEventSynchronize(stop1) );
    HANDLE_ERROR( cudaEventElapsedTime(&time1, start1, stop1) ); 
    
    HANDLE_ERROR( cudaEventCreate(&start2) );
    HANDLE_ERROR( cudaEventCreate(&stop2) );
    HANDLE_ERROR( cudaEventRecord(start2, 0) );

    HANDLE_CUSPARSE_ERROR( cusparseSbsrmm(handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          CUSPARSE_OPERATION_TRANSPOSE, nb, dim, nb, bsrNnzb, &alpha, bsrDescr, bsrVal,
                                          bsrRowPtr, bsrColInd, bsize, y, dim, &beta, z, n1) );

    HANDLE_ERROR( cudaEventRecord(stop2, 0) );
    HANDLE_ERROR( cudaEventSynchronize(stop2) );
    HANDLE_ERROR( cudaEventElapsedTime(&time2, start2, stop2) );
    
    printf("csrmm cost time:  %3.10f ms \n", time1);  
    printf("bsrmm cost time:  %3.10f ms \n", time2);   
    printf("total cost time:  %3.10f ms \n", time1 + time2);   

    HANDLE_ERROR( cudaMemcpy(zHostPtr, z, (size_t)(n1 * dim * sizeof(float)), cudaMemcpyDeviceToHost) );

    HANDLE_CUSPARSE_ERROR( cusparseDestroyMatDescr(bsrDescr) );
    bsrDescr = 0;
    HANDLE_CUSPARSE_ERROR( cusparseDestroyMatDescr(csrDescr) );
    csrDescr = 0;
    HANDLE_CUSPARSE_ERROR( cusparseDestroy(handle) );
    handle = 0;

    CLEANUP("end");

    return 0;
}
