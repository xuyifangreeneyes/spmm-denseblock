import numpy as np
import tensorflow as tf

from blocksparse.matmul import BlocksparseMatMul

n = 1024
bsize = 8
nb = n // bsize
p = 0.02
dim = 128

conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)

with tf.Session(config=conf) as sess, tf.device("/gpu:0"):
    layout = np.random.binomial(1, 0.02, size=(nb, nb))
    bsmm = BlocksparseMatMul(layout, block_size=bsize)
    W = np.random.uniform(-1.0, 1.0, bsmm.w_shape).astype(np.float32)
    w = tf.constant(W) 
    x = tf.placeholder(tf.float32, shape=[None, n])
    y = bsmm(x, w)
    result = sess.run([y], feed_dict={x: np.random.uniform(-1.0, 1.0, (dim, n)).astype(np.float32)})
    print(result)