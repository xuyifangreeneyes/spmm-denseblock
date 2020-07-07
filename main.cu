#include <iostream>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

__global__ void add(float* x, float * y, float* z, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        z[i] = x[i] + y[i];
    }
}

int main()
{
    int N = 1 << 20;
    int nBytes = N * sizeof(float);

    float *x, *y, *z;
    x = (float*)malloc(nBytes);
    y = (float*)malloc(nBytes);
    z = (float*)malloc(nBytes);

    for (int i = 0; i < N; ++i) {
        x[i] = 10.0;
        y[i] = 20.0;
    }

    float *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, nBytes);
    cudaMalloc((void**)&d_y, nBytes);
    cudaMalloc((void**)&d_z, nBytes);

    cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    add << < gridSize, blockSize >> >(d_x, d_y, d_z, N);

    cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyDeviceToHost);

    float maxError = 0.0;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(z[i] - 30.0));
    std::cout << "z0 = " << z[0] << std::endl;
    std::cout << "maximum error: " << maxError << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    free(x);
    free(y);
    free(z);

    return 0;
}