import subprocess

def test_bsrmm():
    density_list = [0.0002, 0.002, 0.02]
    block_dim_list = [2, 4, 8, 16, 32, 64]
    # feat_dim_list = [64,]
    feat_dim_list = [64, 128, 256, 512]
    # bsrmm_impl_list = ["rocsparse",]
    bsrmm_impl_list = ["rocsparse", "cusparse"]
    # trans_b_list = [0, ]
    trans_b_list = [0, 1]
    with open('test_bsrmm_result.txt', 'w') as f:
        for density in density_list:
            for block_dim in block_dim_list:
                for feat_dim in feat_dim_list:
                    for bsrmm_impl in bsrmm_impl_list:
                        for trans_b in trans_b_list:
                            print("hi")
                            subprocess.run(['./test_bsrmm', str(density), str(block_dim), str(feat_dim), bsrmm_impl, str(trans_b)], stdout=f, stderr=f)

def test_csrmm():
    density_list = [0.0002, 0.002, 0.02]
    # feat_dim_list = [512,]
    feat_dim_list = [64, 128, 256, 512]
    csrmm_impl_list = ["gespmm", "cusparse"]
    with open('test_csrmm_result.txt', 'w') as f:
        for density in density_list:
            for feat_dim in feat_dim_list:
                for csrmm_impl in csrmm_impl_list:
                    print("hi2")
                    subprocess.run(['./test_csrmm', str(density), str(feat_dim), csrmm_impl], stdout=f, stderr=f)


def run_csrmm():
    # dataset_list =  ['ogbn_arxiv', 'ogbn_proteins', 'ogbl_ppa', 'ogbl_collab', 'ogbl_ddi']
    dataset_list =  ['ogbl_ddi',]
    reordering_list = ['original', 'rcmk', 'rabbit']
    feat_dim_list = [16, 32, 64, 128]
    spmm_list = [('cusparseScsrmm', 0), ('cusparseScsrmm2', 1), ('gespmm', 0)]
    for dataset in dataset_list:
        for feat_dim in feat_dim_list:
            for spmm, trans in spmm_list:
                for reordering in reordering_list:
                    subprocess.run(['./run_csrmm', dataset + '_' + reordering, str(feat_dim), spmm, str(trans)])

if __name__ == '__main__':
    # test_bsrmm()
    # test_csrmm()
    run_csrmm()