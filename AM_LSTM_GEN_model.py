#-*-coding:utf8-*-
#user:brian
#created_at:2019/8/9 17:13
# file: AM_LSTM_GEN_model.py
#location: china chengdu 610000
####公式部分已经梳理完毕，下周之内完成文献综述和引言部分8.12-19
####下下周8.20-27完成实验部分
###预计八月底完成初稿，带着论文去开学啦！！！
###安装配置好了mendeley和为知笔记的同步工具！
##本周开始高效的写引言和文献综述了！
###怎么昨天写的没有上传上去呢

##吸引力模型的类
import pydot_ng
print(pydot_ng.find_graphviz())

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
import tensorflow.keras.backend  as K
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, concatenate, Input, Conv2D, Reshape, Flatten, Dropout, BatchNormalization, Concatenate, LSTM
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
import pydot
print(tf.__version__)
print(tf.keras.__version__)

class data_generator(layers.Layer):
    def __init__(self,n_outputs):
        super(data_generator,self).__init_()
        self.n_outputs=n_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable('kernel',
                                        shape=[int(input_shape[-1]),
                                               self.n_outputs])
    def call(self, input):
        return tf.matmul(input, self.kernel)



class Attention(layers.Layer):
    def __init__(self, method=None, **kwargs):
        self.supports_masking = True
        if method != 'lba' and method !='ga' and method != 'cba' and method is not None:
            raise ValueError('attention method is not supported')
        self.method = method
        super(Attention, self).__init__(**kwargs)

    #定义自定义的权重
    def build(self, input_shape):
        if isinstance(input_shape, list):
            self.att_size = input_shape[0][-1]
            self.query_dim = input_shape[1][-1]
            if self.method == 'ga' or self.method == 'cba':
                self.Wq = self.add_weight(name='kernal_query_features', shape=(self.query_dim, self.att_size), initializer='glorot_normal', trainable=True)
        else:
            self.att_size = input_shape[-1]

        if self.method == 'cba':
            self.Wh = self.add_weight(name='kernal_hidden_features', shape=(self.att_size,self.att_size), initializer='glorot_normal', trainable=True)
        if self.method == 'lba' or self.method == 'cba':
            self.v = self.add_weight(name='query_vector', shape=(self.att_size, 1), initializer='zeros', trainable=True)

        super(Attention, self).build(input_shape)##必须要有的


    def call(self, inputs, mask=None):
        '''
        :param inputs: a list of tensor of length not larger than 2, or a memory tensor of size BxTXD1.
        If a list, the first entry is memory, and the second one is query tensor of size BxD2 if any
        :param mask: the masking entry will be directly discarded
        :return: a tensor of size BxD1, weighted summing along the sequence dimension
        '''
        ##模型中对应的记忆单元为过去7天的
        ##输入为memory和query 就是记忆单元和查询单元，分别计算出最后的值，
        ###相当于每一个记忆单元和查询值最后得到一个输出
        ###一共有7天的记忆单元，那么就有7个输出在值
        ###然后再进行加权？？7天的结果得到不同的值
        if isinstance(inputs, list) and len(inputs) == 2:
            memory, query = inputs
            if self.method is None:
                return memory[:,-1,:]
            ###本文用的cba conten_base_attention
            elif self.method == 'cba':
                ##expand_dims 表示维度变换
                hidden = K.dot(memory, self.Wh) + K.expand_dims(K.dot(query, self.Wq), 1)
                hidden = K.tanh(hidden)
                #squeeze表示移除一个维度
                s = K.squeeze(K.dot(hidden, self.v), -1)
            elif self.method == 'ga':
                s = K.sum(K.expand_dims(K.dot(query, self.Wq), 1) * memory, axis=-1)
            else:
                s = K.squeeze(K.dot(memory, self.v), -1)
            if mask is not None:
                mask = mask[0]
        else:
            if isinstance(inputs, list):
                if len(inputs) != 1:
                    raise ValueError('inputs length should not be larger than 2')
                memory = inputs[0]
            else:
                memory = inputs
            if self.method is None:
                return memory[:,-1,:]
            elif self.method == 'cba':
                hidden = K.dot(memory, self.Wh)
                hidden = K.tanh(hidden)
                s = K.squeeze(K.dot(hidden, self.v), -1)
            elif self.method == 'ga':
                raise ValueError('general attention needs the second input')
            else:
                s = K.squeeze(K.dot(memory, self.v), -1)

        ##在利用sofmax函数进行输出

        s = K.softmax(s)
        if mask is not None:
            s *= K.cast(mask, dtype='float32')
            sum_by_time = K.sum(s, axis=-1, keepdims=True)
            s = s / (sum_by_time + K.epsilon())

        ##返回基于概率的输出
        return K.sum(memory * K.expand_dims(s), axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            att_size = input_shape[0][-1]
            batch = input_shape[0][0]
        else:
            att_size = input_shape[-1]
            batch = input_shape[0]
        return (batch, att_size)

class models:
    def __init__(self):
        pass
    #stdn函数
    #需要定义的输入参数：
    ##按照5分钟划分间隔 一共有288个点   10分钟划分间隔 144个点 预测6点--24点  5分钟就是 18*12=216个点 10分钟就是108个点  15分钟就是18*4=72个点
    ##5分钟 每个航班时间区间跨度假设2小时=24，那就是24个短期特征长度 sita=24
    ## att_lstm_num:吸引力的长度 也就是一共有多少天 长期吸引力天 w=7天
    ##att_lstm_seg_len:每个吸引子的序列长度 短期吸引力短期特征长度 sitar=24个
    ##吸引力个数：也就是要预测的个数  φ=216个点，表示预测未来的216个点
    ##长期时间效应趋势用lstm来捕捉，得到一个h值，短期则用h跟他相关的过去再来捕捉一次sita=24，表示与短期的24个相关

    ###参数指标组合（时间间隔为5分钟==> w=7 sita=24 φ=216）
    ###           （时间间隔为10分钟==> w=7 sita=12 φ=108）
    ###                       15        w=7 sita=8  φ=72
    ###                       30        w=7 sita=4  φ=36

    def stdn(self, att_lstm_day, att_lstm_seq_sita, lstm_num_fai, lstm_out_size=268,output_shape=268,optimizer = 'adagrad', loss = 'mse', metrics=["accuracy"]):
        """
        :param att_lstm_day: 一共过去XXX天的数据
        :param att_lstm_seq_sita: 吸引力力序列的长度 这里就是sita的值 20
        :param lstm_num_fai: 每一天的数据的长度   268
        :param optimizer:
        :param loss:
        :param metrics:
        :return:
        """

        ##第一个输入为吸引力 inputs
        att_lstm_inputs = [Input(shape = ( lstm_num_fai,att_lstm_seq_sita,), name = "att_lstm_input_{0}".format(att+1)) for att in range(att_lstm_day)]
        lstm_inputs = Input(shape = (lstm_num_fai,1), name = "generator_data")

        #对产生数据进行一个lstm
        lstm = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(lstm_inputs)


        ####对过去的每一天进行了一个lstm
        att_lstms = [LSTM(units=lstm_out_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, name="att_lstm_{0}".format(att + 1))(att_lstm_inputs[att]) for att in range(att_lstm_day)]

        #用的attention中compare这一个方法
        #attention 模型的输入为 attlsems[0] ,lstm

        #吸引力模型，分别得到若干个
        att_low_level=[Attention(method='cba')([att_lstms[att], lstm]) for att in range(att_lstm_day)]
        ##计算得到low_level 也就是每一个memory与要查询的单元的吸引力的结果，对应的是一个预测值

        att_low_level=Concatenate(axis=-1)(att_low_level)
        ##然后再把这些值连接在一起，维度得到了增加
        att_low_level=Reshape(target_shape=( lstm_out_size,att_lstm_day))(att_low_level)
        ######最后再进行了一次lstm，相当于把7填的汇总到一起，进行了一个大的lstm
        att_high_level = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(att_low_level)

        ##相当于有8个lstm？？，8个一起求权重
        lstm_all = Concatenate(axis=-1)([att_high_level, lstm])
        ###
        lstm_all = Dense(units = output_shape)(lstm_all)
        pred_volume = Activation('tanh')(lstm_all)

        inputs =  att_lstm_inputs  + [lstm_inputs,]
        model = Model(inputs = inputs, outputs = pred_volume)
        model.compile(optimizer = optimizer, loss = loss, metrics=metrics)
        return model


# nbhd_inputs = [Input(shape = (3, 3, 2,), name = "nbhd_volume_input_time_{0}".format(ts+1)) for ts in range(20)]
# flatten_att_nbhd_inputs = [
#     Input(shape=(3, 3, 2,), name="att_nbhd_volume_input_time_{0}_{1}".format(att + 1, ts + 1))
#     for ts in range(20) for att in range(7)]
# z=[5 for i in range(2) for j in range(3)]
# print(1)
# model.fit(x=att_cnnx + att_flow + att_x + cnnx + flow + [x, ],
#           y=y,atch_size=128, , epochs=max_epochs, callbacks=[EarlyStopping()])
if __name__ == '__main__':
    model = models().stdn(att_lstm_day=7, att_lstm_seq_sita=20, lstm_num_fai=268)
    keras.utils.plot_model(model, 'model_info_V33.png', show_shapes=True)
    model.summary()