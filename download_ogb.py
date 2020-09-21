from ogb.nodeproppred import NodePropPredDataset
from ogb.linkproppred import LinkPropPredDataset

def load_and_dump(gtype, gname):
    dataset_name = '{}-{}'.format(gtype, gname)
    if gtype == 'ogbn':
        dataset = NodePropPredDataset(name=dataset_name)
        graph, _ = dataset[0]
    elif gtype == 'ogbl':
        dataset = LinkPropPredDataset(name=dataset_name)
        graph = dataset[0]
    else:
        assert False
    edge_index = graph['edge_index']
    n = graph['num_nodes']
    nnz = edge_index.shape[1]
    assert 0 in edge_index and not n in edge_index # check 0-base
    with open('tmp/{}_{}.txt'.format(gtype, gname), 'w') as f:
        f.write('{} {}\n'.format(n, nnz))
        for i in range(nnz):
            f.write('{} {}\n'.format(edge_index[0][i], edge_index[1][i]))
            if i % 100000 == 0:
                print("{} i={}".format(dataset_name, i))

if __name__ == '__main__':
    ogbn_datasets = ['products', 'proteins', 'arxiv']
    ogbl_datasets = ['ppa', 'collab', 'ddi', 'citation']
    for dataset in ogbn_datasets:
        load_and_dump('ogbn', dataset)
    for dataset in ogbl_datasets:
        load_and_dump('ogbl', dataset)