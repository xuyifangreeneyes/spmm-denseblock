#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <assert.h>

typedef long long ll;

std::vector<std::vector<int>> csr2adj(const std::string& indptr_file, 
                                      const std::string& indices_file, int n, int nnz) {

    std::fstream s1(indptr_file, std::ios::in);
    std::fstream s2(indices_file, std::ios::in);

    std::vector<int> indptr, indices;

    int xx;
    s1 >> xx;
    assert(xx == n + 1);
    for (int i = 0; i <= n; ++i) {
        s1 >> xx;
        indptr.push_back(xx);
    }
    assert(indptr[n] - indptr[0] == nnz);
    s2 >> xx;
    assert(xx == nnz);
    for (int i = 0; i < nnz; ++i) {
        s2 >> xx;
        indices.push_back(xx);
    }

    assert(indptr[0] == 0);
    std::vector<std::vector<int>> edges(n);
    for (int i = 0; i < n; ++i) {
        int start = indptr[i], end = indptr[i + 1];
        for (int j = start; j < end; ++j) {
            edges[i].push_back(indices[j]);
        }
    }

    return edges;
}


std::array<ll, 10> calculate_block_density_dist(const std::vector<std::vector<int>>& edges, 
                                                   int n, int bsize) {
    std::array<ll, 10> dist;
    dist.fill(0);
    int nb = (n + bsize - 1) / bsize;
    std::vector<int> vec(nb, 0);
    for (int x1 = 0; x1 < nb; ++x1) {
        // std::cout << "x1 = " << x1 << std::endl;
        std::fill(vec.begin(), vec.end(), 0);
        for (int x2 = 0; x2 < bsize; ++x2) {
            // std::cout << "x2 = " << x2 << std::endl;
            int x = x1 * bsize + x2;
            if (x >= n) {
                break;
            }
            // std::cout << "kk1" << std::endl;
            const std::vector<int>& ys = edges[x];
            // std::cout << "kk2" << std::endl;
            for (int y : ys) {
                // std::cout << "y = " << y << std::endl;
                vec[y / bsize]++;
            }
            // std::cout << "kk3" << std::endl;
        }
        // std::cout << "k1" << std::endl;
        for (int num : vec) {
            if (num == 0) {
                continue;
            }
            float occupy = (num * 1.0) / (bsize * bsize);
            int ith = (int) (occupy * 10);
            if (ith >= 10) {
                // std::cout << "large than ten" << std::endl;
                ith = 9;
            }
            dist[ith]++;
        }
    }
    return dist;
}

int main() {
    int n = 235868, nnz = 2358104;
    std::string indptr_file = "collab_gpmetis2048_rcmk_indptr.txt";
    std::string indices_file = "collab_gpmetis2048_rcmk_indices.txt";

    // std::cout << "h1" << std::endl;

    std::vector<std::vector<int>> edges = csr2adj(indptr_file, indices_file, n, nnz);

    // std::cout << "h2" << std::endl;

    std::array<ll, 10> dist = calculate_block_density_dist(edges, n, 32);
    
    // std::cout << "h3" << std::endl;
    
    for (int i = 0; i < 10; ++i) {
        std::cout << i << ": " << dist[i] << std::endl;
    }

    return 0;
}

