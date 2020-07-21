from ogb.graphproppred import GraphPropPredDataset

N = 100000

def num_same_elements(v1, v2):
    l1, l2 = len(v1), len(v2)
    i, j, cnt = 0, 0, 0
    while i < l1 and j < l2:
        if v1[i] == v2[j]:
            cnt += 1
            i += 1
            j += 1
        elif v1[i] > v2[j]:
            j += 1
        else:
            i += 1
    return cnt

def closest_reorder(graph):
    n = graph['num_nodes']
    edge_index = graph['edge_index']
    nnz = edge_index.shape[1]
    adj = [[] for _ in range(n)]
    for i in range(nnz):
        x, y = edge_index[0][i], edge_index[1][i]
        adj[x].append(y)
    for i in range(n):
        adj[i] = sorted(adj[i])
    new2old = [-1 for _ in range(n)]
    old2new = [-1 for _ in range(n)]
    new2old[0] = 0
    old2new[0] = 0
    for i in range(1, n):
        prev = new2old[i - 1]
        max_val, pos = 0, -1
        for j in range(n):
            if old2new[j] != -1 or i == j:
                continue
            val = num_same_elements(adj[prev], adj[j])
            if val >= max_val:
                max_val = val
                pos = j
        new2old[i] = pos
        old2new[pos] = i
    for i in range(n):
        l = len(adj[i])
        for j in range(l):
            adj[i][j] = old2new[adj[i][j]]
        adj[i] = sorted(adj[i])
    new_adj = [None] * n
    for i in range(n):
        new_adj[old2new[i]] = adj[i]
    return new_adj


if __name__ == '__main__':
    dataset = GraphPropPredDataset(name='ogbg-molhiv')
    num_sub, num_nodes = 0, 0
    
    # edges = []
    # while num_nodes < N:
    #     graph, _ = dataset[num_sub]
    #     n = graph['num_nodes']
    #     edge_index = graph['edge_index']
    #     nnz = edge_index.shape[1]
    #     for i in range(nnz):
    #         edges.append((edge_index[0][i] + num_nodes, edge_index[1][i] + num_nodes))
    #     num_sub += 1
    #     num_nodes += n
    
    # print("number of subgraphs: {}, number of nodes: {}, number of edges: {}".format(num_sub, num_nodes, len(edges)))
    
    # adj = [[] for _ in range(num_nodes)]
    # for x, y in edges:
    #     adj[x].append(y)
    # for i in range(num_nodes):
    #     adj[i] = sorted(adj[i])
    
    adj = []
    while num_nodes < N:
        graph, _ = dataset[num_sub]
        n = graph['num_nodes']
        sub_adj = closest_reorder(graph)
        for i in range(n):
            adj.append([num_nodes + y for y in sub_adj[i]])
        num_sub += 1
        num_nodes += n

    print("number of subgraphs: {}, number of nodes: {}".format(num_sub, num_nodes))

    indptr, indices = [0], []
    for i in range(num_nodes):
        x = indptr[-1] + len(adj[i])
        indptr.append(x)
        indices.extend(adj[i])

    with open('ogbg_molhiv_2_indptr.txt', 'w') as f:
        f.write(str(len(indptr)) + '\n')
        for x in indptr:
            f.write(str(x) + ' ')
        f.write('\n')
    with open('ogbg_molhiv_2_indices.txt', 'w') as f:
        f.write(str(len(indices)) + '\n')
        for x in indices:
            f.write(str(x) + ' ')
        f.write('\n')