import tvm
from tvm import te
import numpy as np

tgt_host = "llvm"
tgt = "cuda"

n = te.var("n")
nnz = te.var("nnz")
dim = te.var("dim")

coo_rows = te.placeholder((nnz,), name="coo_rows")
coo_cols = te.placeholder((nnz,), name="coo_cols")
coo_vals = te.placeholder((nnz,), name="csr_val")
B = te.placeholder((n, dim), name="B")