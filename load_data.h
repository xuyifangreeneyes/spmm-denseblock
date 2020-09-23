#ifndef SPMM_DENSEBLOCK_LOAD_DATA_H
#define SPMM_DENSEBLOCK_LOAD_DATA_H

#include <string>
#include <utility>
#include <vector>

float *randomArray(int n, float minVal = -1, float maxVal = 1);

float *randomDenseMatrix(int n, int dim, float minVal = -1, float maxVal = 1);

int randomCSRMatrix(int m, int n, float p, int **csrRowPtr, int **csrColInd,
                    float **csrVal, float minVal = -1, float maxVal = 1,
                    bool dump = false);

int readAndFillCSRMatrix(int m, int n, float p, int **csrRowPtr,
                         int **csrColInd, float **csrVal, float minVal = -1,
                         float maxVal = 1);

int randomBSRMatrix(int mb, int nb, int blockDim, float p, int **bsrRowPtr,
                    int **bsrColInd, float **bsrVal, float minVal = -1,
                    float maxVal = 1, bool dump = false);

int readAndFillBSRMatrix(int mb, int nb, int blockDim, float p, int **bsrRowPtr,
                         int **bsrColInd, float **bsrVal, float minVal = -1,
                         float maxVal = 1);

void dumpCSRToFile(const std::string &prefix, int n, int nnz, int *csrRowPtr,
                   int *csrColInd);

std::pair<int, int> loadCSRFromFile(const std::string &prefix, int **csrRowPtr, int **csrColInd);

int loadGraphFromFile(const std::string &filename,
                      std::vector<std::vector<int>> &edges);

std::pair<int *, int *> convertGraphToCSR(
    const std::vector<std::vector<int>> &edges);

#endif