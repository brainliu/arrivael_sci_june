#-*- coding:utf-8 -*-
#created by brian
# create time :2019/8/27-21:05 
#location: sichuan chengdu
from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.keras.backend.clear_session()
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class MyLayer(layers.Layer):
    def __init__(self, input_dim=32, unit=32):
        super(MyLayer, self).__init__()

        w_init = tf.random_normal_initializer()
        self.weight = tf.Variable(initial_value=w_init(
            shape=(input_dim, unit), dtype=tf.float32), trainable=True)

        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=b_init(
            shape=(unit,), dtype=tf.float32), trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias

x = tf.ones((3,5))
my_layer = MyLayer(5, 4)
out = my_layer(x)
print(out)


print('Is GPU available:')
print(tf.test.is_gpu_available())
print('Is the Tensor on gpu #0:')
print(x.device.endswith('GPU:0'))

import time
def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)
    result = time.time() - start
    print('10 loops: {:0.2}ms'.format(1000*result))

# 强制使用CPU
print('On CPU:')
with tf.device('CPU:0'):
    x = tf.random.uniform([1000, 1000])
    # 使用断言验证当前是否为CPU0
    assert x.device.endswith('CPU:0')
    time_matmul(x)

# 如果存在GPU,强制使用GPU
if tf.test.is_gpu_available():
    print('On GPU:')
    with tf.device.endswith('GPU:0'):
        x = tf.random.uniform([1000, 1000])
    # 使用断言验证当前是否为GPU0
    assert x.device.endswith('GPU:0')
    time_matmul(x)


# num_words = 30000
# maxlen = 200
# (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_words)
# print(x_train.shape, ' ', y_train.shape)
# print(x_test.shape, ' ', y_test.shape)
# x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen, padding='post')
# x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen, padding='post')
# print(x_train.shape, ' ', y_train.shape)
# print(x_test.shape, ' ', y_test.shape)
#
# def lstm_model():
#     model = keras.Sequential([
#         layers.Embedding(input_dim=30000, output_dim=32, input_length=maxlen),
#         layers.LSTM(32, return_sequences=True),
#         layers.LSTM(1, activation='sigmoid', return_sequences=False)
#     ])
#     model.compile(optimizer=keras.optimizers.Adam(),
#                  loss=keras.losses.BinaryCrossentropy(),
#                  metrics=['accuracy'])
#     return model
# model = lstm_model()
# model.summary()

"""
history = model.fit(x_train, y_train, batch_size=64, epochs=5,validation_split=0.1)
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()
"""

class MyDense(tf.keras.layers.Layer):
    def __init__(self, n_outputs):
        super(MyDense, self).__init__()
        self.n_outputs = n_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable('kernel',
                                       shape=[int(input_shape[-1]),
                                             self.n_outputs])
    def call(self, input):
        return tf.matmul(input, self.kernel)

layer = MyDense(10)
print(layer(tf.ones([6, 5])))
print(layer.trainable_variables)