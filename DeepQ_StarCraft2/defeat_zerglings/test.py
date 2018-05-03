import tensorflow as tf
import numpy as np



with tf.device("/gpu:0"):
        hello = tf.constant("my test")
        var1 = tf.constant(dtype=tf.float32, value=1)
        var2 = tf.constant(dtype=tf.float32, value=2)
        var3 = tf.multiply(var1, var2)
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(hello))
    print(sess.run(var3))


