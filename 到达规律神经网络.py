#-*-coding:utf8-*-
#user:brian
#created_at:2019/6/27 9:56
# file: 到达规律神经网络.py
#location: china chengdu 610000
import tensorflow as tf
from tensorflow.keras import layers
print(tf.__version__)
print(tf.keras.__version__)
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
