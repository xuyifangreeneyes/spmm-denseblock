import subprocess

def test_bsrmm():
    density_list = [0.0002, 0.002, 0.02]
    block_dim_list = [2, 4, 8, 16, 32, 64]
    feat_dim_list = [64, 128, 256]
    bsrmm_impl_list = ["rocsparse", "cusparse"]
    trans_b_list = [0, 1]
    with open('test_bsrmm_result.txt', 'w') as f:
        for density in density_list:
            for block_dim in block_dim_list:
                for feat_dim in feat_dim_list:
                    for bsrmm_impl in bsrmm_impl_list:
                        for trans_b in trans_b_list:
                            subprocess.run(['./test_bsrmm', str(density), str(block_dim), str(feat_dim), bsrmm_impl, str(trans_b)], stdout=f, stderr=f)

def test_csrmm():
    density_list = [0.0002, 0.002, 0.02]
    feat_dim_list = [64, 128, 256]
    csrmm_impl_list = ["gespmm", "cusparse"]
    with open('test_csrmm_result.txt', 'w') as f:
        for density in density_list:
            for feat_dim in feat_dim_list:
                for csrmm_impl in csrmm_impl_list:
                    subprocess.run(['./test_csrmm', str(density), str(feat_dim), csrmm_impl], stdout=f, stderr=f)

if __name__ == '__main__':
    # test_bsrmm()
    test_csrmm()