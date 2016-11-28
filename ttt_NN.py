# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:29:07 2016

@author: Mike
"""

import tensorflow as tf

class ttt_NN():
    def __init__(self, params):
        self.params = params
        self.label_placeholder = tf.placeholder(tf.float32,
                                                shape=[None, params.LABELLEN])
        self.labels = tf.reshape(self.label_placeholder, [-1, params.LABELLEN])
        self.input_placeholder = tf.placeholder(tf.float32,
                                                shape=[None, params.INPUTLEN])
        
        with tf.variable_scope("Layer") as scope:
            W1 = tf.get_variable("W1", [params.INPUTLEN, params.HIDDENSIZE])
            b1 = tf.get_variable("b1", [params.HIDDENSIZE])
            layer_1 = tf.add(tf.matmul(self.input_placeholder, W1), b1)
            self.lastoutput = tf.nn.relu(layer_1)

        with tf.variable_scope("linclassifier") as scope:
            self.V = tf.get_variable("Voutput",
                                     [params.HIDDENSIZE, params.LABELLEN])
            self.b = tf.get_variable("boutput", [params.LABELLEN])

            self.logits = tf.nn.xw_plus_b(self.lastoutput, self.V, self.b)

        with tf.variable_scope("reinforcement_score") as scope:
            self.assignedscore = tf.reduce_sum(
                                            tf.mul(self.labels, self.logits),
                                            reduction_indices=1)
            self.score_placeholder = tf.placeholder(tf.float32, shape=[None])
            self.reinforcementloss = tf.reduce_sum(tf.square(
                                                    self.score_placeholder -
                                                    self.assignedscore))

            rein_optimizer = tf.train.GradientDescentOptimizer(
                                                        params.LEARNING_RATE)
            rein_global_step = tf.Variable(0, name='rein_global_step',
                                           trainable=False)
            self.reinforcement_train_op = rein_optimizer.minimize(
                                                self.reinforcementloss,
                                                global_step=rein_global_step)
