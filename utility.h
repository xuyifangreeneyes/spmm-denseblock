#ifndef SPMM_DENSEBLOCK_UTILTITY_H
#define SPMM_DENSEBLOCK_UTILTITY_H

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <stdlib.h>
#include "cusparse.h"

template <typename T>
T* vec2ptr(const std::vector<T>& vec) {
  T* ptr = (T*)malloc(vec.size() * sizeof(T));
  for (int i = 0; i < vec.size(); ++i) {
    ptr[i] = vec[i];
  }
  return ptr;
}

bool checkError(cudaError_t err, const char* file, int line);

bool checkCusparseError(cusparseStatus_t status, const char* file, int line);

std::pair<int *, int *> convertGraphToCSR(const std::vector<std::vector<int>> &edges);

int calculateNnzb(const std::vector<std::vector<int>>& edges, int blockSize);

std::vector<std::vector<int>> getHeatmap(const std::vector<std::vector<int>>& edges, int blockSize);

void dumpHeatmap(const std::string& filename, const std::vector<std::vector<int>>& heatmap);

#endif
