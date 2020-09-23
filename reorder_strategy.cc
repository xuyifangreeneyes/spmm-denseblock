#include <vector>
#include <queue>
#include <algorithm>
#include <assert.h>
#include "reorder_strategy.h"

namespace {

struct Node {
    int id;
    int val;
    Node(int id, int val) : id(id), val(val) {}
};

}

std::vector<std::vector<int>> sortNeighbors(std::vector<std::vector<int>> edges) {
    int n = edges.size();
    for (int i = 0; i < n; ++i) {
        std::vector<int>& neighbors = edges[i];
        std::sort(neighbors.begin(), neighbors.end());
    }
    return edges;
}

std::vector<std::vector<int>> permutate(const std::vector<int>& old2new, std::vector<std::vector<int>> edges) {
    int n = edges.size();
    for (int i = 0; i < n; ++i) {
        std::vector<int>& neighbors = edges[i];
        for (int j = 0; j < neighbors.size(); ++j) {
            neighbors[j] = old2new[neighbors[j]];
        }
    }
    std::vector<std::vector<int>> new_edges(n);
    for (int i = 0; i < n; ++i) {
        new_edges[old2new[i]] = std::move(edges[i]);
    }
    return sortNeighbors(std::move(new_edges));
}

std::vector<std::vector<int>> maxDegreeSort(std::vector<std::vector<int>> edges) {
    int n = edges.size();
    std::vector<Node> nodes;
    for (int i = 0; i < n; ++i) {
        nodes.emplace_back(i, edges[i].size());
    }
    std::sort(nodes.begin(), nodes.end(), [](const Node& n1, const Node& n2) {
        return n1.val > n2.val;
    });
    std::vector<int> old2new(n, -1);
    for (int i = 0; i < n; ++i) {
        old2new[nodes[i].id] = i;
    } 
    return permutate(old2new, std::move(edges));
}

std::vector<std::vector<int>> reverseCuthillMcKee(std::vector<std::vector<int>> edges) {
    int n = edges.size();
    for (int i = 0; i < n; ++i) {
        std::vector<int>& neighbors = edges[i];
        std::sort(neighbors.begin(), neighbors.end(), [&edges](int x, int y) {
            return edges[x].size() > edges[y].size();
        });
    }
    return BFSTraversal(std::move(edges));
}

std::vector<std::vector<int>> BFSTraversal(std::vector<std::vector<int>> edges) {
    int n = edges.size();
    std::vector<int> old2new(n, -1);
    std::queue<int> que;
    int cnt = 0, pos = 0;
    while (true) {
        for (; pos < n; ++pos) {
            if (old2new[pos] == -1) {
                old2new[pos] = cnt++;
                que.push(pos);
                break;
            }
        }
        if (que.empty()) {
            break;
        }
        while (!que.empty()) {
            int x = que.front();
            que.pop();
            const std::vector<int>& neighbors = edges[x];
            for (int y : neighbors) {
                if (old2new[y] == -1) {
                    old2new[y] = cnt++;
                    que.push(y);
                }
            }
        }
    }
    assert(cnt == n);
    return permutate(old2new, std::move(edges));
}

