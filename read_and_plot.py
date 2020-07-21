from ogb.nodeproppred import NodePropPredDataset
from ogb.linkproppred import LinkPropPredDataset


if __name__ == '__main__':
    dataset = NodePropPredDataset(name='ogbn-arxiv')
    # dataset = LinkPropPredDataset(name='ogbl-collab')
    graph, label = dataset[0]
    # graph = dataset[0]
    
    # rcmk(graph)
    # print("rcmk end")
    # util = cal_util(graph, 4)
    # print(util)
    
    edge_index = graph['edge_index']
    nnz = edge_index.shape[1]
    print(graph['num_nodes'])
    print(nnz)
    with open('ogbn-arxiv/src.txt', 'w') as f:
        for i in range(nnz):
            f.write(str(edge_index[0][i]) + '\n')
            if i % 100000 == 0:
                print("src.txt i = {}".format(i))
    with open('ogbn-arxiv/dst.txt', 'w') as f:
        for i in range(nnz):
            f.write(str(edge_index[1][i]) + '\n')
            if i % 100000 == 0:
                print("dst.txt i = {}".format(i))