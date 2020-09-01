import tvm
from tvm import te

n = te.var("n")
nnz = te.var("nnz")
dim = te.var("dim")

csr_row_ptr = te.placeholder((n + 1), name="csr_row_ptr")
csr_col_ind = te.placeholder((nnz,), name="csr_col_ind")
csr_val = te.placeholder((nnz,), name="csr_val")
B = te.placeholder((n, dim), name="B")

def _compute_csrmm(i, j):
    row_start = csr_row_ptr[i]
    row_end = csr_row_ptr[i + 1]
    k = te.reduce_axis((row_start, row_end), name="k")
    cid = csr_col_ind[k]
    val1 = csr_val[k]
    val2 = B[cid, j]
    return te.sum(val1 * val2, axis=k)

C = te.compute((n, dim), _compute_csrmm, name="C")

s = te.create_schedule(C.op)
print(tvm.lower(s, [csr_row_ptr, csr_col_ind, csr_val, B, C], simple_mode=True))

