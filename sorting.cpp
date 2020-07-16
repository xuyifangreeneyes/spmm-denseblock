#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <assert.h>


const int N = 169343;
const int M = 1166243;

void dump_csr(const std::vector<std::vector<int>>& edges, const std::string& name) {
    std::cout << "dumping" << std::endl;
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
    std::fstream s1(name + "_indptr.txt");
    std::fstream s2(name + "_indices.txt");

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

struct Vertex {
    int id;
    int degree;

    Vertex(int id, int degree) : id(id), degree(degree) {}
};

int main() {
        std::vector<int> src_vec;
    std::vector<int> dst_vec;
    std::fstream s1("src.txt", std::ios_base::in);
    std::fstream s2("dst.txt", std::ios_base::in);

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

    std::cout << "src num: " << src_vec.size() << std::endl;
    std::cout << "dst num: " << dst_vec.size() << std::endl;
    
    std::vector<std::vector<int>> edges(N);

    for (int i = 0; i < M; ++i) {
        int x = src_vec[i];
        int y = dst_vec[i];
        edges[x].push_back(y);
    }

    for (int i = 0; i < N; ++i) {
        std::sort(edges[i].begin(), edges[i].end());
    }

    for (int x = 0; x < 50; ++x) {
        std::cout << "[" << x << "]: ";
        for (int y : edges[x]) {
            std::cout << y << " ";
        }
        std::cout << std::endl;
    }

    std::vector<Vertex> v_vec;
    for (int i = 0; i < N; ++i) {
        v_vec.emplace_back(i, edges[i].size());
    }

    std::sort(v_vec.begin(), v_vec.end(), [](const Vertex& v1, const Vertex& v2) {
        return v1.degree > v2.degree;
    });

    std::vector<int> old2new(N, -1);
    for (int i = 0; i < N; ++i) {
        old2new[v_vec[i].id] = i;
    }

    std::vector<std::vector<int>> reordered_edges(N);
    for (int i = 0; i < N; ++i) {
        reordered_edges[old2new[i]] = std::move(edges[i]);
    }

    for (int x = 0; x < 50; ++x) {
        std::cout << "[" << x << "]: ";
        for (int y : reordered_edges[x]) {
            std::cout << y << " ";
        }
        std::cout << std::endl;
    }

    dump_csr(reordered_edges, "arxiv_sorting");

    int block_size = 4;
    int nb = (N + block_size - 1) / block_size;

    int nonempty_block_num = 0;
    std::vector<int> vec(nb, 0);
    for (int x1 = 0; x1 < nb; ++x1) {
        std::fill(vec.begin(), vec.end(), 0);
        for (int x2 = 0; x2 < block_size; ++x2) {
            int x = x1 * block_size + x2;
            if (x >= N) {
                break;
            }
            std::vector<int>& ys = reordered_edges[x];
            for (int y : ys) {
                vec[y / block_size] = 1;
            }
        }
        for (int z : vec) {
            if (z == 1) {
                ++nonempty_block_num;
            }
        }
        if (x1 % 1000 == 0) {
            std::cout << "x1 = " << x1 << std::endl;
            std::cout << "nonempty = " << nonempty_block_num << std::endl;
        }
    }

    std::cout << nonempty_block_num << std::endl;
    double density = nonempty_block_num / ((nb * 1.0) * (nb * 1.0));
    std::cout << "block density: " << density << std::endl;

    return 0;
}
