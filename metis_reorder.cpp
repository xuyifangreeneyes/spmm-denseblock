#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <assert.h>

const int N = 235868;
const int M = 2358104;

void dump_csr(const std::vector<std::vector<int>>& edges, const std::string& name) {
    std::cout << "dumping " << name << std::endl;
    std::vector<int> indptr, indices;
    int cnt = 0;
    indptr.push_back(cnt);
    for (const auto &neighbors : edges) {
        cnt += neighbors.size();
        indptr.push_back(cnt);
        for (int v : neighbors) {
            indices.push_back(v);
        }
    }
    std::fstream s1(name + "_indptr.txt", std::ios::out | std::ios::trunc);
    std::fstream s2(name + "_indices.txt", std::ios::out | std::ios::trunc);

    s1 << indptr.size() << std::endl;
    for (int x : indptr) {
        s1 << x << " ";
    }
    s1 << std::endl;

    s2 << indices.size() << std::endl;
    for (int x : indices) {
        s2 << x << " ";
    }
    s2 << std::endl;
}

void calculate_density_and_utilization(const std::vector<std::vector<int>>& edges, const std::string& name, int block_size) {
    std::cout << "calculate density and utilization [" << name << "] block_size = " << block_size << std::endl;
    int nb = (N + block_size - 1) / block_size;

    int nnzb = 0;
    std::vector<int> vec(nb, 0);
    for (int x1 = 0; x1 < nb; ++x1) {
        std::fill(vec.begin(), vec.end(), 0);
        for (int x2 = 0; x2 < block_size; ++x2) {
            int x = x1 * block_size + x2;
            if (x >= N) {
                break;
            }
            const std::vector<int>& ys = edges[x];
            for (int y : ys) {
                vec[y / block_size] = 1;
            }
        }
        for (int z : vec) {
            if (z == 1) {
                ++nnzb;
            }
        }
        // if (x1 % 1000 == 0) {
        //     std::cout << "x1 = " << x1 << std::endl;
        //     std::cout << "nonempty = " << nnzb << std::endl;
        // }
    }

    // std::cout << nnzb << std::endl;
    double density = ((double) nnzb) / (((double) nb) * ((double) nb));
    double utilization = ((double) M) / (((double) nnzb) * ((double) block_size) * ((double) block_size));  
    double average = ((double) M) / ((double) nnzb);
    std::cout << "density: " << density << std::endl;
    std::cout << "utilization: " << utilization << std::endl;
    std::cout << "average: " << average << std::endl;
}

int main() {
    std::vector<int> src_vec;
    std::vector<int> dst_vec;
    std::fstream s1("ogbl-collab/src.txt", std::ios_base::in);
    std::fstream s2("ogbl-collab/dst.txt", std::ios_base::in);

    for (int i = 0; i < M; ++i) {
        int x;
        s1 >> x;
        src_vec.push_back(x);
    }

    for (int i = 0; i < M; ++i) {
        int y;
        s2 >> y;
        dst_vec.push_back(y);
    }

    std::vector<std::vector<int>> edges(N);

    int m = 0;

    for (int i = 0; i < M; ++i) {
        int x = src_vec[i];
        int y = dst_vec[i];
        if (x < y) {
            ++m;
        }
        edges[x].push_back(y);
    }

    assert(m * 2 == M);
    std::cout << "M = " << M << std::endl;
    std::cout << "m = " << m << std::endl;

    // std::vector<int> block_sizes = {2, 4, 8, 16, 32, 64};
    // for (int block_size : block_sizes) {
    //     calculate_density_and_utilization(edges, "naive", block_size);
    // }

    std::fstream s3("ogbl_collab_adj.txt.iperm", std::ios_base::in);
    std::vector<int> old2new;
    for (int i = 0; i < N; ++i) {
        int x;
        s3 >> x;
        old2new.push_back(x);
    }

    for (int i = 0; i < N; ++i) {
        std::vector<int>& ys = edges[i];
        for (int j = 0; j < ys.size(); ++j) {
            ys[j] = old2new[ys[j]];
        }
        std::sort(ys.begin(), ys.end());
    }

    std::vector<std::vector<int>> reordered_edges(N);
    for (int i = 0; i < N; ++i) {
        reordered_edges[old2new[i]] = std::move(edges[i]);
    }

    // for (int block_size : block_sizes) {
    //     calculate_density_and_utilization(reordered_edges, "metis", block_size);
    // }

    dump_csr(reordered_edges, "collab_ndmetis");

    return 0;
}