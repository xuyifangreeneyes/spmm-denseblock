import tvm
from tvm import te

tgt_host = "llvm"
tgt = "cuda"

n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i : A[i] + B[i], name="C")

s = te.create_schedule(C.op)

bx, tx = s[C].split(C.op.axis[0], factor=64)
s[C].bind(bx, te.thread_axis("blockIdx.x"))
s[C].bind(tx, te.thread_axis("threadIdx.x"))

my_add = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="my_add")
dev_module = my_add.imported_modules[0]
print(dev_module.get_source())