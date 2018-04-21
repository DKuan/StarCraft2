# from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat
# from baselines import logger
# import numpy as np
# logdir = "./tensorboard"
# Logger.DEFAULT \
#       = Logger.CURRENT \
#       = Logger(dir=None,
#                output_formats=[TensorBoardOutputFormat(logdir)])
#
# for i in range(10):
#     logger.record_tabular("steps", i)
#     logger.record_tabular("episodes", i)
#     logger.record_tabular("reward", i)
#     logger.record_tabular("mean 100 episode reward",
#                           i)
#     logger.dump_tabular()
# a =[[1,2], [3,4], [4,5]]
# b=np.array(a)
# length = b.size//2
# c = b[:,0]
#
# print(c)

import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()
x =tf.constant([[1,1],
                [2,3]])
y = tf.constant([[2,3],
                [4,5]])
floatx = tf.constant([
        [1., 2.],
        [2., 3.]
    ])
b=tf.transpose(x)
