import tensorflow as tf
from tensorflow.contrib.layers.python.layers import *

class DeepQ(object):
    def __init__(self, convs, hiddens, dueling, inpt, num_actions, scope, reuse=False, layer_norm=False):
        with tf.variable_scope("cnn1"):
            out = inpt
            with tf.variable_scope("convnet"):
                for num_outputs, kernel_size, stride in convs:
                    out = layers.convolution2d(out,
                                               num_outputs=num_outputs,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               activation_fn=tf.nn.relu)
            conv_out = layers.flatten(out)
            with tf.variable_scope("action_value"):
                action_out = conv_out
                for hidden in hiddens:
                    action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        action_out = layers.layer_norm(action_out, center=True, scale=True)
                    action_out = tf.nn.relu(action_out)
                action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

            with tf.variable_scope("state_value"):
                state_out = conv_out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        state_out = layers.layer_norm(state_out, center=True, scale=True)
                    state_out = tf.nn.relu(state_out)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered