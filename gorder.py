import subprocess

def gorder():
    exe_file = './Gorder/Gorder'
    dataset_list = ['ogbn_arxiv', 'ogbn_proteins', 'ogbl_ppa', 'ogbl_collab', 'ogbl_ddi']
    for dataset in dataset_list:
        subprocess.run([exe_file, 'tmp/' + dataset + '_1.txt'])

if __name__ == '__main__':
    gorder()
