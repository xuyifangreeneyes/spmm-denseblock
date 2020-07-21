from blocksparse.matmul import BlocksparseMatMul
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np

n = 2 << 14
bsize = 32
dim = 128
nb = n // bsize

sparsity = np.random.binomial(1, 0.02, size=(nb, nb))

bsmm = BlocksparseMatMul(sparsity, block_size=bsize)

x = tf.placeholder(tf.float32, shape=[dim, n])

print("bsmm.w_shape = {}".format(bsmm.w_shape))
W = np.random.uniform(-1.0, 1.0, bsmm.w_shape).astype(np.float32)
w = tf.constant(W) 

y = bsmm(x, w)

with tf.Session() as sess:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(tf.global_variables_initializer())
    result = sess.run([y], feed_dict={x: np.random.uniform(-0.1, 0.1, (dim, n)).astype(np.float32)}, options=run_options, run_metadata=run_metadata)
    print(result)

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
        f.write(ctf)