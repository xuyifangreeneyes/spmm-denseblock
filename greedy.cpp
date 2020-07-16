#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <queue>
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


    dump_csr(edges, "arxiv_1");

    std::vector<int> visit(N, -1);
    std::queue<int> q;

    int cnt = 0;
    int pos = 0;
    while (true) {
        for (; pos < N; ++pos) {
            if (visit[pos] == -1) {
                visit[pos] = cnt++; 
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
                if (visit[y] != -1) {
                    continue;
                }
                visit[y] = cnt++;
                q.push(y);
            }
        }
    }

    assert(cnt == N);

    for (int i = 0; i < N; ++i) {
        std::vector<int>& ys = edges[i];
        for (int j = 0; j < ys.size(); ++j) {
            ys[j] = visit[ys[j]];
        }
        std::sort(ys.begin(), ys.end());
    }

    std::vector<std::vector<int>> reordered_edges(N);
    for (int i = 0; i < N; ++i) {
        reordered_edges[visit[i]] = std::move(edges[i]);
    }

    for (int x = 0; x < 50; ++x) {
        std::cout << "[" << x << "]: ";
        for (int y : reordered_edges[x]) {
            std::cout << y << " ";
        }
        std::cout << std::endl;
    }

    dump_csr(reordered_edges, "arxiv_4");

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
