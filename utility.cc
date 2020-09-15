#include <stdio.h>
#include <stdlib.h>
#include "utility.h"

bool check_error(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line%d\n", cudaGetErrorString(err), file, line);
        return false;
    }
    return true;
}

bool check_cusparse_error(cusparseStatus_t status, const char* file, int line) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("%s in %s at line%d\n", cusparseGetErrorString(status), file, line);
        return false;
    }
    return true;
}