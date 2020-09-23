#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <queue>
#include <utility>
#include <assert.h>
#include "utility.h"
#include "load_data.h"
#include "reorder_strategy.h"

void analyzeBlockSparseMetrics(const std::vector<std::vector<int>>& edges, int nnz) {
    int n = edges.size();
    std::vector<int> blockSizes = {2, 4, 8, 16, 32, 64};
    for (int blockSize : blockSizes) {
        int nb = (n + blockSize - 1) / blockSize;
        int nnzb = calculateNnzb(edges, blockSize);
        double density = ((double) nnzb) / (((double) nb) * ((double) nb));
        double utilization = ((double) nnz) / (((double) nnzb) * ((double) blockSize) * ((double) blockSize));  
        double average = ((double) nnz) / ((double) nnzb);
        std::cout << "blockSize=" << blockSize << " density=" << density << " utilization=" 
                  << utilization << " average=" << average << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::string dataset = argv[1];
    std::cout << "dataset=" << dataset << std::endl;
    std::vector<std::vector<int>> edges;
    std::cout << "hi1" << std::endl;
    int nnz = loadGraphFromFile("tmp/" + dataset + ".txt", edges);
    int n = edges.size();
    std::cout << "n=" << n << " nnz=" << nnz << std::endl;

    std::string prefix = "tmp/" + dataset + "_original";
    std::pair<int*, int*> csrPtrs = convertGraphToCSR(edges);
    dumpCSRToFile(prefix, n, nnz, csrPtrs.first, csrPtrs.second);
    analyzeBlockSparseMetrics(edges, nnz);
    dumpHeatmap(prefix + "_heatmap.txt", getHeatmap(edges, 256));

    prefix = "tmp/" + dataset + "_rcmk";
    edges = reverseCuthillMcKee(edges);
    csrPtrs = convertGraphToCSR(edges);
    dumpCSRToFile(prefix, n, nnz, csrPtrs.first, csrPtrs.second);
    analyzeBlockSparseMetrics(edges, nnz);
    dumpHeatmap(prefix + "_heatmap.txt", getHeatmap(edges, 256));

    return 0;
}
