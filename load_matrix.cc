#include <assert.h>
#include <fstream>
#include <random>
#include <sstream>
#include <stdlib.h>
#include "load_matrix.h"
#include "utility.h"


static std::mt19937_64 gen(1234);

static std::string getCSRName(const std::string &format, int m, int n,
                              float p) {
  std::stringstream ss;
  int nnz = (int)(m * (n * p));
  ss << "csr_" << format << "_" << m << "_" << n << "_" << nnz << ".txt";
  return ss.str();
}

static std::string getBSRName(const std::string &format, int mb, int nb,
                              int blockDim, float p) {
  std::stringstream ss;
  int nnzb = (int)(mb * (nb * p));
  ss << "bsr_" << format << "_" << mb << "_" << nb << "_" << blockDim << "_"
     << nnzb << ".txt";
  return ss.str();
}

int randomCSRMatrix(int m, int n, float p, int **hostCsrRowPtr,
                    int **hostCsrColInd, float **hostCsrVal, float minVal,
                    float maxVal, bool dump) {
  std::uniform_real_distribution<float> flip(0, 1), dist(minVal, maxVal);
  *hostCsrRowPtr = (int *)malloc((m + 1) * sizeof(int));
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
  }
  *hostCsrColInd = vec2ptr(indices);
  *hostCsrVal = vec2ptr(vals);

  if (dump) {
    // Generating random CSR matrix is time-consuming, so we record it for next
    // time use.
    std::ofstream s1(getCSRName("indptr", m, n, p));
    std::ofstream s2(getCSRName("indices", m, n, p));

    s1 << m + 1 << std::endl;
    for (int i = 0; i <= m; ++i) {
      s1 << (*hostCsrRowPtr)[i] << " ";
    }
    s1 << std::endl;

    s2 << cnt << std::endl;
    for (int i = 0; i < cnt; ++i) {
      s2 << (*hostCsrColInd)[i] << " ";
    }
    s2 << std::endl;
  }
  return cnt;
}

int readAndFillCSRMatrix(int m, int n, float p, int **hostCsrRowPtr,
                         int **hostCsrColInd, float **hostCsrVal, float minVal,
                         float maxVal) {
  std::ifstream s1(getCSRName("indptr", m, n, p));
  std::ifstream s2(getCSRName("indices", m, n, p));

  int m_plus_one;
  s1 >> m_plus_one;
  assert(m + 1 == m_plus_one);
  *hostCsrRowPtr = (int *)malloc((m + 1) * sizeof(int));
  for (int i = 0; i <= m; ++i) {
    s1 >> (*hostCsrRowPtr)[i];
  }

  int nnz;
  s2 >> nnz;
  *hostCsrColInd = (int *)malloc(nnz * sizeof(int));
  for (int i = 0; i < nnz; ++i) {
    s2 >> (*hostCsrColInd)[i];
  }

  *hostCsrVal = (float *)malloc(nnz * sizeof(float));
  std::uniform_real_distribution<float> dist(minVal, maxVal);
  for (int i = 0; i < nnz; ++i) {
    (*hostCsrVal)[i] = dist(gen);
  }

  return nnz;
}

int randomBSRMatrix(int mb, int nb, int blockDim, float p, int **hostBsrRowPtr,
                    int **hostBsrColInd, float **hostBsrVal, float minVal,
                    float maxVal, bool dump) {
  std::uniform_real_distribution<float> flip(0, 1), dist(minVal, maxVal);
  int blockNum = blockDim * blockDim;
  *hostBsrRowPtr = (int *)malloc((mb + 1) * sizeof(int));
  int cnt = 0;
  (*hostBsrRowPtr)[0] = cnt;
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
    (*hostBsrRowPtr)[i] = cnt;
  }
  *hostBsrColInd = vec2ptr(std::move(indices));
  *hostBsrVal = vec2ptr(std::move(vals));

  if (dump) {
    // Generating random BSR matrix may be time-consuming, so we record it for
    // next time use.
    std::string bd = std::to_string(blockDim);
    std::ofstream s1(getBSRName("indptr", mb, nb, blockDim, p));
    std::ofstream s2(getBSRName("indices", mb, nb, blockDim, p));

    s1 << mb + 1 << std::endl;
    for (int i = 0; i <= mb; ++i) {
      s1 << (*hostBsrRowPtr)[i] << " ";
    }
    s1 << std::endl;

    s2 << cnt << std::endl;
    for (int i = 0; i < cnt; ++i) {
      s2 << (*hostBsrColInd)[i] << " ";
    }
    s2 << std::endl;
  }
  return cnt;
}

int readAndFillBSRMatrix(int mb, int nb, int blockDim, float p,
                         int **hostBsrRowPtr, int **hostBsrColInd,
                         float **hostBsrVal, float minVal, float maxVal) {
  std::ifstream s1(getBSRName("indptr", mb, nb, blockDim, p));
  std::ifstream s2(getBSRName("indices", mb, nb, blockDim, p));

  int mb_plus_one;
  s1 >> mb_plus_one;
  assert(mb + 1 == mb_plus_one);
  *hostBsrRowPtr = (int *)malloc((mb + 1) * sizeof(int));
  for (int i = 0; i <= mb; ++i) {
    s1 >> (*hostBsrRowPtr)[i];
  }

  int nnzb;
  s2 >> nnzb;
  *hostBsrColInd = (int *)malloc(nnzb * sizeof(int));
  for (int i = 0; i < nnzb; ++i) {
    s2 >> (*hostBsrColInd)[i];
  }

  int num = nnzb * blockDim * blockDim;
  *hostBsrVal = (float *)malloc(num * sizeof(float));
  std::uniform_real_distribution<float> dist(minVal, maxVal);
  for (int i = 0; i < num; ++i) {
    (*hostBsrVal)[i] = dist(gen);
  }

  return nnzb;
}

float *randomDenseMatrix(int n, int dim, float minVal, float maxVal) {
  std::uniform_real_distribution<float> dist(minVal, maxVal);
  int sz = n * dim;
  float *ptr = (float *)malloc(sz * sizeof(float));
  for (int i = 0; i < sz; ++i) {
    ptr[i] = dist(gen);
  }
  return ptr;
}