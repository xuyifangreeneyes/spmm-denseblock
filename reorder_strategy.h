#ifndef SPMM_DENSEBLOCK_REORDER_STRATEGY_H
#define SPMM_DENSEBLOCK_REORDER_STRATEGY_H

#include <vector>

std::vector<std::vector<int>> sortNeighbors(std::vector<std::vector<int>> edges);

std::vector<std::vector<int>> permutate(const std::vector<int>& old2new, std::vector<std::vector<int>> edges);

std::vector<std::vector<int>> maxDegreeSort(std::vector<std::vector<int>> edges);

std::vector<std::vector<int>> reverseCuthillMcKee(std::vector<std::vector<int>> edges);

std::vector<std::vector<int>> BFSTraversal(std::vector<std::vector<int>> edges);


#endif