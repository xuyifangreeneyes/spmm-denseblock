#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <algorithm>
#include "load_data.h"
#include "utility.h"

static std::mt19937_64 gen(1234);

static std::string getCSRName(int m, int n, float p) {
  std::stringstream ss;
  int nnz = (int)(m * (n * p));
  ss << "tmp/csr_"
     << "_" << m << "_" << n << "_" << nnz;
  return ss.str();
}

static std::string getBSRName(int mb, int nb, int blockDim, float p) {
  std::stringstream ss;
  int nnzb = (int)(mb * (nb * p));
  ss << "tmp/bsr_" << mb << "_" << nb << "_" << blockDim << "_" << nnzb;
  return ss.str();
}

float *randomArray(int n, float minVal, float maxVal) {
  std::uniform_real_distribution<float> dist(minVal, maxVal);
  float *ptr = (float *)malloc(n * sizeof(float));
  for (int i = 0; i < n; ++i) {
    ptr[i] = dist(gen);
  }
  return ptr;
}

float *randomDenseMatrix(int n, int dim, float minVal, float maxVal) {
  return randomArray(n * dim, minVal, maxVal);
}

int randomCSRMatrix(int m, int n, float p, int **csrRowPtr, int **csrColInd,
                    float **csrVal, float minVal, float maxVal, bool dump) {
  std::uniform_real_distribution<float> flip(0, 1), dist(minVal, maxVal);
  *csrRowPtr = (int *)malloc((m + 1) * sizeof(int));
  int cnt = 0;
  (*csrRowPtr)[0] = cnt;
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
    (*csrRowPtr)[i] = cnt;
  }
  *csrColInd = vec2ptr(indices);
  *csrVal = vec2ptr(vals);

  if (dump) {
    // Generating random CSR matrix is time-consuming, so we record it for next
    // time use.
    dumpCSRToFile(getCSRName(m, n, p), m, cnt, *csrRowPtr, *csrColInd);
  }
  return cnt;
}

int readAndFillCSRMatrix(int m, int n, float p, int **csrRowPtr,
                         int **csrColInd, float **csrVal, float minVal,
                         float maxVal) {
  std::pair<int, int> pair = loadCSRFromFile(getCSRName(m, n, p), csrRowPtr, csrColInd);
  assert(pair.first == m);
  int nnz = pair.second;
  *csrVal = randomArray(nnz);
  return nnz;
}

int randomBSRMatrix(int mb, int nb, int blockDim, float p, int **bsrRowPtr,
                    int **bsrColInd, float **bsrVal, float minVal, float maxVal,
                    bool dump) {
  std::uniform_real_distribution<float> flip(0, 1), dist(minVal, maxVal);
  int blockNum = blockDim * blockDim;
  *bsrRowPtr = (int *)malloc((mb + 1) * sizeof(int));
  int cnt = 0;
  (*bsrRowPtr)[0] = cnt;
  std::vector<int> indices;
  std::vector<float> vals;
  for (int i = 1; i <= mb; ++i) {
    for (int j = 0; j < nb; ++j) {
      if (flip(gen) < p) {
        indices.push_back(j);
        for (int k = 0; k < blockNum; ++k) {
          vals.push_back(dist(gen));
        }
        ++cnt;
      }
    }
    (*bsrRowPtr)[i] = cnt;
  }
  *bsrColInd = vec2ptr(indices);
  *bsrVal = vec2ptr(vals);

  if (dump) {
    // Generating random BSR matrix may be time-consuming, so we record it for
    // next time use.
    dumpCSRToFile(getBSRName(mb, nb, blockDim, p), mb, cnt, *bsrRowPtr,
                  *bsrColInd);
  }
  return cnt;
}

int readAndFillBSRMatrix(int mb, int nb, int blockDim, float p, int **bsrRowPtr,
                         int **bsrColInd, float **bsrVal, float minVal,
                         float maxVal) {
  std::pair<int, int> pair = loadCSRFromFile(getBSRName(mb, nb, blockDim, p), bsrRowPtr, bsrColInd);
  assert(pair.first == mb);
  int nnzb = pair.second;
  *bsrVal = randomArray(nnzb * blockDim * blockDim);
  return nnzb;
}

void dumpCSRToFile(const std::string &prefix, int n, int nnz, int *csrRowPtr,
                   int *csrColInd) {
  std::ofstream s1(prefix + "_indptr.txt");
  std::ofstream s2(prefix + "_indices.txt");

  s1 << n + 1 << std::endl;
  for (int i = 0; i <= n; ++i) {
    s1 << csrRowPtr[i] << " ";
  }
  s1 << std::endl;

  s2 << nnz << std::endl;
  for (int i = 0; i < nnz; ++i) {
    s2 << csrColInd[i] << " ";
  }
  s2 << std::endl;
}

std::pair<int, int> loadCSRFromFile(const std::string &prefix, int **csrRowPtr,
                    int **csrColInd) {
  std::ifstream s1(prefix + "_indptr.txt");
  std::ifstream s2(prefix + "_indices.txt");

  int n_plus_one;
  s1 >> n_plus_one;
  int n = n_plus_one - 1;
  // std::cout << "n = " << n << std::endl;
  *csrRowPtr = (int *)malloc((n + 1) * sizeof(int));
  for (int i = 0; i <= n; ++i) {
    s1 >> (*csrRowPtr)[i];
  }

  int nnz;
  // std::cout << "nnz = " << nnz << std::endl;
  s2 >> nnz;
  *csrColInd = (int *)malloc(nnz * sizeof(int));
  for (int i = 0; i < nnz; ++i) {
    s2 >> (*csrColInd)[i];
  }
  return {n, nnz};
}

int loadGraphFromFile(const std::string &filename,
                      std::vector<std::vector<int>> &edges) {
  std::ifstream fs(filename);
  int n, nnz;
  fs >> n >> nnz;
  edges.clear();
  edges.resize(n);
  for (int i = 0; i < nnz; ++i) {
    int x, y;
    fs >> x >> y;
    edges[x].push_back(y);
  }
  for (int i = 0; i < n; ++i) {
    std::vector<int>& neighbors = edges[i];
    std::sort(neighbors.begin(), neighbors.end());
  }
  return nnz;
}
