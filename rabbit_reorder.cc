#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <utility>
#include "load_data.h"
#include "utility.h"
#include "reorder_strategy.h"

std::vector<int> loadPermutation(const std::string& filename, int n) {
    std::ifstream fs(filename);
    std::vector<int> old2new;
    for (int i = 0; i < n; ++i) {
        int x;
        fs >> x;
        old2new.push_back(x);
    }
    return old2new;
}

int main(int argc, char* argv[]) {
    std::string dataset = argv[1];
    std::cout << "dataset=" << dataset << std::endl;
    std::vector<std::vector<int>> edges;
    int nnz = loadGraphFromFile("tmp/" + dataset + ".txt", edges);
    int n = edges.size();
    std::cout << "n=" << n << " nnz=" << nnz << std::endl;

    std::vector<int> old2new = loadPermutation("tmp/" + dataset + "_rabbit.txt", n);
    edges = permutate(old2new, std::move(edges));

    std::pair<int*, int*> csrPtrs = convertGraphToCSR(edges);
    dumpCSRToFile("tmp/" + dataset + "_rabbit", n, nnz, csrPtrs.first, csrPtrs.second);
    return 0;
}