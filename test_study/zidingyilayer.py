#-*-coding:utf8-*-
#user:brian
#created_at:2019/6/27 8:58
# file: zidingyilayer.py
#location: china chengdu 610000
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras

print(tf.__version__)
print(tf.keras.__version__)
# class MyLayer(layers.Layer):
#     def __init__(self, input_dim=5, unit=32):
#         super(MyLayer, self).__init__()
#
#         w_init = tf.ones_initializer()
#
#         self.weight = tf.Variable(initial_value=w_init(
#             shape=(input_dim, unit), dtype=tf.float32), trainable=True)
#
#         print(input_dim,unit)
#         b_init = tf.zeros_initializer()
#         self.bias = tf.Variable(initial_value=b_init(
#             shape=(unit,), dtype=tf.float32), trainable=True)
#
#     def call(self, inputs1):
#         #print(inputs1)
#         print(inputs1,self.weight,self.bias)
#         return tf.matmul(inputs1, self.weight) + self.bias
#
#
# x = tf.ones((3, 5))
# my_layer = MyLayer(5, 4)
# out = my_layer(x)
# print(out)


class MyLayer(layers.Layer):
    def __init__(self, unit=32):
        super(MyLayer, self).__init__()
        self.unit = unit

    def build(self, input_shape):
        self.weight = self.add_weight(shape=(input_shape[-1], self.unit),
                                      initializer=keras.initializers.RandomNormal(),
                                      trainable=True)
        self.bias = self.add_weight(shape=(self.unit,),
                                    initializer=keras.initializers.Zeros(),
                                    trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias

class MyBlock(layers.Layer):
    def __init__(self):
        super(MyBlock, self).__init__()
        self.layer1 = MyLayer(32)
        self.layer2 = MyLayer(16)
        self.layer3 = MyLayer(2)

    def call(self, inputs):
        h1 = self.layer1(inputs)
        h1 = tf.nn.relu(h1)
        h2 = self.layer2(h1)
        h2 = tf.nn.relu(h2)
        return self.layer3(h2)


my_block = MyBlock()
print('trainable weights:', len(my_block.trainable_weights))
y = my_block(tf.ones(shape=(3, 64)))
# 构建网络在build()里面，所以执行了才有网络
print('trainable weights:', len(my_block.trainable_weights))
可以通过构建网络层的方法来收集loss


class LossLayer(layers.Layer):
    def __init__(self, rate=1e-2):
        super(LossLayer, self).__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs

​
class OutLayer(layers.Layer):
    def __init__(self):
        super(OutLayer, self).__init__()
        self.loss_fun = LossLayer(1e-2)

    def call(self, inputs):
        return self.loss_fun(inputs)


my_layer = OutLayer()
print(len(my_layer.losses))  # 还未call
y = my_layer(tf.zeros(1, 1))
print(len(my_layer.losses))  # 执行call之后
y = my_layer(tf.zeros(1, 1))
print(len(my_layer.losses))  # call之前会重新置0