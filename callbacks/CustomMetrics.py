"""
Code to permit the use of custom metrics in Keras (where the metrics are
obtained from some other module).

Author: 1st Lt Ian McQuaid
Date: 22 May 2019
"""
import tensorflow as tf
from tensorflow.keras import backend as K


class CustomMetrics(object):
    def __init__(self):
        # Things get mad if you don't initialize these...
        sess = K.get_session()
        self.max_f1_tensor = tf.Variable(0.0, dtype=tf.float32)
        sess.run(tf.global_variables_initializer())

    def max_f1(self, y_true, y_pred):
        return self.max_f1_tensor
