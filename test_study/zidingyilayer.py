#-*-coding:utf8-*-
#user:brian
#created_at:2019/6/27 8:58
# file: zidingyilayer.py
#location: china chengdu 610000
import tensorflow as tf
from tensorflow.keras import layers
print(tf.__version__)
print(tf.keras.__version__)
class MyLayer(layers.Layer):
    def __init__(self, input_dim=5, unit=32):
        super(MyLayer, self).__init__()

        w_init = tf.random_normal_initializer()
        self.weight = tf.Variable(initial_value=w_init(
            shape=(input_dim, unit), dtype=tf.float32), trainable=True)
        print(input_dim,unit)
        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=b_init(
            shape=(unit,), dtype=tf.float32), trainable=True)

    def call(self, inputs1):
        print(inputs1)
        return tf.matmul(inputs1, self.weight) + self.bias


x = tf.ones((3, 5))
my_layer = MyLayer(5, 4)
out = my_layer(x)
print(out)
