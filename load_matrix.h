int randomCSRMatrix(int m, int n, float p, int **hostCsrRowPtr,
                    int **hostCsrColInd, float **hostCsrVal, float minVal = -1,
                    float maxVal = 1);

int readAndFillCSRMatrix(int m, int n, int **hostCsrRowPtr, int **hostCsrColInd,
                         float **hostCsrVal, float minVal = -1,
                         float maxVal = 1);

int randomBSRMatrix(int mb, int nb, int blockDim, float p, int **hostBsrRowPtr,
                    int **hostBsrColInd, float **hostBsrVal, float minVal = -1,
                    float maxVal = 1);

int readAndFillBSRMatrix(int mb, int nb, int blockDim, int **hostBsrRowPtr,
                         int **hostBsrColInd, float **hostBsrVal,
                         float minVal = -1, float maxVal = 1);

float* randomDenseMatrix(int n, int dim, float minVal=-1, float maxVal=1);
