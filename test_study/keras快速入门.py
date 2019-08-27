#-*-coding:utf8-*-
#user:brian
#created_at:2019/6/26 10:29
# file: keras快速入门.py
#location: china chengdu 610000
import tensorflow as tf
from tensorflow.keras import layers
print(tf.__version__)
print(tf.keras.__version__)
model = tf.keras.Sequential()
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

import numpy as np

train_x = np.random.random((1000, 72))
train_y = np.random.random((1000, 10))

val_x = np.random.random((200, 72))
val_y = np.random.random((200, 10))

inputs = tf.keras.Input(shape=(784,), name='img')
h1 = layers.Dense(32, activation='relu')(inputs)
h2 = layers.Dense(32, activation='relu')(h1)
outputs = layers.Dense(10, activation='softmax')(h2)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mnist model')

model.summary()

x = tf.ones((3, 5))
inputdata={}
x = [[3.]]
m = tf.matmul(x, x)
print(m.numpy())

a = tf.constant([[1,9],[3,6]])
print(a)

import tensorflow as tf
my_var = tf.Variable(tf.ones([2,3]))
print(my_var)
try:
    with tf.device("/device:GPU:0"):
        v = tf.Variable(tf.zeros([10, 10]))
        print(v)
except:
    print('no gpu')

