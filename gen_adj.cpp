#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <assert.h>

const int N = 235868;
const int M = 2358104;

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

    std::fstream s3("ogbl_collab_adj.txt", std::ios_base::out | std::ios_base::trunc);
    s3 << N << " " << m << std::endl;
    for (int i = 0; i < N; ++i) {
        const auto& v = edges[i];
        for (int x : v) {
            s3 << x + 1 << " ";
        }
        s3 << std::endl;
    }

    return 0;
}