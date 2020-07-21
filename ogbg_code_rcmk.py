import queue
from ogb.graphproppred import GraphPropPredDataset


def rcmk(graph):
    n = graph['num_nodes']
    adj = [[] for _ in range(n)]
    edge_index = graph['edge_index']
    nnz = edge_index.shape[1]
    for i in range(nnz):
        src, dst = edge_index[0][i], edge_index[1][i]
        adj[src].append(dst)
    for i in range(n):
        pairs = [(x, len(adj[x])) for x in adj[i]]
        adj[i] = [x for x, _ in sorted(pairs, key=lambda pair : pair[1])]
    
    q = queue.Queue()
    old2new = [-1 for _ in range(n)]
    cnt = 0
    
    q.put(0)
    old2new[0] = cnt
    cnt += 1        
    while not q.empty():
        x = q.get()
        for y in adj[x]:
            if old2new[y] != -1:
                continue
            old2new[y] = cnt
            cnt += 1
            q.put(y)
    assert cnt == n

    for i in range(n):
        l = len(adj[i])
        for j in range(l):
            adj[i][j] = old2new[adj[i][j]]
        adj[i] = sorted(adj[i])

    new_adj = [None] * n
    for i in range(n):
        new_adj[old2new[i]] = adj[i]
    
    graph['reordered'] = new_adj
    

def cal_util(graph, block_size):
    adj = graph['reordered']
    n = graph['num_nodes']
    nnz = graph['edge_index'].shape[1]
    nb = (n + block_size - 1) // block_size
    nnzb = 0
    for x1 in range(nb):
        flags = [False] * nb
        for x2 in range(block_size):
            x = x1 * block_size + x2
            if x >= n:
                break
            for y in adj[x]:
                flags[y // block_size] = True
        for flag in flags:
            if flag:
                nnzb += 1
    return nnz / (nnzb * block_size * block_size)

if __name__ == '__main__':
    dataset = GraphPropPredDataset(name='ogbg-code')
    util_sum = 0
    cnt = 0
    for i, (graph, _) in enumerate(dataset):
        rcmk(graph)
        util = cal_util(graph, 4)
        util_sum += util
        print("i = {}, utilization = {}".format(i, util))
        cnt += 1
        if cnt >= 100:
            break
    print("average utilization:", util_sum / cnt)