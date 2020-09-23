#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "cusparse.h"
#include "utility.h"

bool checkError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line%d\n", cudaGetErrorString(err), file, line);
    return false;
  }
  return true;
}

bool checkCusparseError(cusparseStatus_t status, const char* file, int line) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("%s in %s at line%d\n", cusparseGetErrorString(status), file, line);
    return false;
  }
  return true;
}

std::pair<int *, int *> convertGraphToCSR(
    const std::vector<std::vector<int>> &edges) {
  int n = edges.size();
  std::vector<int> indptr, indices;
  int cnt = 0;
  indptr.push_back(cnt);
  for (const auto &neighbors : edges) {
    cnt += neighbors.size();
    indptr.push_back(cnt);
    for (int cid : neighbors) {
      indices.push_back(cid);
    }
  }
  return {vec2ptr(indptr), vec2ptr(indices)};
}

int calculateNnzb(const std::vector<std::vector<int>>& edges, int blockSize) {
  int n = edges.size();
  int nb = (n + blockSize - 1) / blockSize;
  int nnzb = 0;
  std::vector<int> vec(nb, 0);
  for (int x1 = 0; x1 < nb; ++x1) {
    std::fill(vec.begin(), vec.end(), 0);
    for (int x2 = 0; x2 < blockSize; ++x2) {
      int x = x1 * blockSize + x2;
      if (x >= n) {
        break;
      }
      const std::vector<int>& ys = edges[x];
      for (int y : ys) {
        vec[y / blockSize] = 1;
      }
    }
    for (int z : vec) {
      nnzb += z;
    }
  }
  return nnzb;
}

std::vector<std::vector<int>> getHeatmap(const std::vector<std::vector<int>>& edges, int blockSize) {
  int n = edges.size();
  int nb = (n + blockSize - 1) / blockSize;
  std::vector<std::vector<int>> heatmap(nb, std::vector<int>(nb, 0));
  for (int x1 = 0; x1 < nb; ++x1) {
    for (int x2 = 0; x2 < blockSize; ++x2) {
      int x = x1 * blockSize + x2;
      if (x >= n) {
        break;
      }
      const std::vector<int>& ys = edges[x];
      for (int y : ys) {
        heatmap[x1][y / blockSize] += 1;
      }
    }
  }
  return heatmap;
}

void dumpHeatmap(const std::string& filename, const std::vector<std::vector<int>>& heatmap) {
  std::ofstream fs(filename);
  int nb = heatmap.size();
  assert(nb == heatmap[0].size());
  fs << nb << std::endl;
  for (int i = 0; i < nb; ++i) {
    for (int j = 0; j < nb; ++j) {
      fs << heatmap[i][j] << " ";
    }
    fs << std::endl;
  }
}