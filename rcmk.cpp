#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <queue>
#include <assert.h>

const int N = 4267;
const int M = 2135822;

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

void dump_heatmap(const std::vector<std::vector<int>>& edges, const std::string& name, int block_size) {
    std::cout << "dump heatmap [" << name << "] block_size = " << block_size << std::endl;
    int nb = (N + block_size - 1) / block_size;

    std::vector<std::vector<int>> heatmap;
    for (int i = 0; i < nb; ++i) {
        heatmap.emplace_back(nb, 0);
    }
    for (int x1 = 0; x1 < nb; ++x1) {
        for (int x2 = 0; x2 < block_size; ++x2) {
            int x = x1 * block_size + x2;
            if (x >= N) {
                break;
            }
            const std::vector<int>& ys = edges[x];
            for (int y : ys) {
                heatmap[x1][y / block_size] += 1;
            }
        }
    }

    std::fstream fs(name + "_heatmap.txt");
    fs << nb << std::endl;
    for (int i = 0; i < nb; ++i) {
        for (int j = 0; j < nb; ++j) {
            fs << heatmap[i][j] << " ";
        }
        fs << std::endl;
    }
}

int main() {
    std::vector<int> src_vec;
    std::vector<int> dst_vec;
    std::fstream s1("ogbl-ddi/src.txt");
    std::fstream s2("ogbl-ddi/dst.txt");

    std::cout << "read src.txt..." << std::endl;

    for (int i = 0; i < M; ++i) {
        int x;
        s1 >> x;
        src_vec.push_back(x);
    }

    std::cout << "read dst.txt..." << std::endl;

    for (int i = 0; i < M; ++i) {
        int y;
        s2 >> y;
        dst_vec.push_back(y);
    }

    std::cout << "src num: " << src_vec.size() << std::endl;
    std::cout << "dst num: " << dst_vec.size() << std::endl;
    
    std::vector<std::vector<int>> edges(N);

    std::cout << "construct edges..." << std::endl;

    for (int i = 0; i < M; ++i) {
        int x = src_vec[i];
        int y = dst_vec[i];
        edges[x].push_back(y);
    }

    std::cout << "sort edges..." << std::endl;

    for (int i = 0; i < N; ++i) {
        std::sort(edges[i].begin(), edges[i].end(), [&edges](int x, int y) {
            return edges[x].size() < edges[y].size();
        });
        // std::sort(edges[i].begin(), edges[i].end());
    }

    for (int x = 0; x < 50; ++x) {
        std::cout << "[" << x << "]: ";
        for (int y : edges[x]) {
            std::cout << y << "(" << edges[y].size() << ") ";
        }
        std::cout << std::endl;
    }


    // dump_csr(edges, "ddi_naive");

    std::vector<int> block_sizes = {2, 4, 8, 16, 32, 64};
    // for (int block_size : block_sizes) {
    //     calculate_density_and_utilization(edges, "naive", block_size);
    // }

    // dump_heatmap(edges, "ddi_naive", 1);

    std::cout << "bfs..." << std::endl;

    std::vector<int> old2new(N, -1);
    std::queue<int> q;

    int cnt = 0;
    int pos = 0;
    while (true) {
        for (; pos < N; ++pos) {
            if (old2new[pos] == -1) {
                old2new[pos] = cnt++; 
                q.push(pos);
                break;
            }
        }

        if (q.empty()) {
            break;
        }

        while (!q.empty()) {
            int x = q.front();
            q.pop();
            for (int y : edges[x]) {
                if (old2new[y] != -1) {
                    continue;
                }
                old2new[y] = cnt++;
                q.push(y);
            }
        }
    }

    assert(cnt == N);

    std::cout << "construct new edges..." << std::endl;

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

    for (int x = 0; x < 50; ++x) {
        std::cout << "[" << x << "]: ";
        for (int y : reordered_edges[x]) {
            std::cout << y << " ";
        }
        std::cout << std::endl;
    }

    dump_csr(reordered_edges, "ddi_rcmk");

    for (int block_size : block_sizes) {
        calculate_density_and_utilization(reordered_edges, "rcmk", block_size);
    }

    dump_heatmap(reordered_edges, "ddi_rcmk", 1);

    return 0;
}
