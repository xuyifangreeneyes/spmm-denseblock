#ifndef SPMM_DENSEBLOCK_UTILTITY
#define SPMM_DENSEBLOCK_UTILTITY

#include <vector>
#include <cuda_runtime.h>
#include "cusparse.h"

template<typename T>
T* vec2ptr(const std::vector<T>& vec) {
    T* ptr = (T*) malloc(vec.size() * sizeof(T));
    for (int i = 0; i < vec.size(); ++i) {
        ptr[i] = vec[i];
    }
    return ptr;
}

bool checkError(cudaError_t err, const char* file, int line);

bool checkCusparseError(cusparseStatus_t status, const char* file, int line);


#endif
