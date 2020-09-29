import subprocess

def rabbit_reorder():
    exe_file = './rabbit_order/demo/reorder'
    dataset_list = ['ogbn_arxiv', 'ogbn_proteins', 'ogbl_ppa', 'ogbl_collab', 'ogbl_ddi']
    for dataset in dataset_list:
        with open('tmp/' + dataset + '_rabbit.txt', 'w') as f:
            subprocess.run([exe_file, 'tmp/' + dataset + '_1.txt'], stdout=f)

if __name__ == '__main__':
    rabbit_reorder()