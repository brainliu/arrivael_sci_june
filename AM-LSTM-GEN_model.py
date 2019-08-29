#-*-coding:utf8-*-
#user:brian
#created_at:2019/8/9 17:13
# file: AM-LSTM-GEN_model.py
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
    def stdn(self, att_lstm_num, att_lstm_seq_len, lstm_seq_len, feature_vec_len, cnn_flat_size = 128, lstm_out_size = 128,\
    nbhd_size = 3, nbhd_type = 2, map_x_num = 10, map_y_num = 20, flow_type = 4, output_shape = 2, optimizer = 'adagrad', loss = 'mse', metrics=[]):
        """
        :param att_lstm_num:
        :param att_lstm_seq_len:
        :param lstm_seq_len:
        :param feature_vec_len:
        :param cnn_flat_size:
        :param lstm_out_size:
        :param nbhd_size:
        :param nbhd_type:
        :param map_x_num:
        :param map_y_num:
        :param flow_type:
        :param output_shape:
        :param optimizer:
        :param loss:
        :param metrics:
        :return:
        """
        ##吸引力的输入层：att_lstm_num，也就是过去的时间个数w，每个输入的长度为att_lstm_seq_len

        ##每一个序列的长度为att_lstm_seq_len 也就是48个时间刻度 一共有7天的lstm数据att_lstm_num
        ##假设有20个时间段相关，然后有7天，这个输入相当于就有140个

        #假设为7，表示由7个吸引能力的输入，每一个输入长度为att_lstm_seq_len
        att_lstm_inputs = [Input(shape = (att_lstm_seq_len, feature_vec_len,), name = "att_lstm_input_{0}".format(att+1)) for att in range(att_lstm_num)]

        nbhd_inputs = [Input(shape = (nbhd_size, nbhd_size, nbhd_type,), name = "nbhd_volume_input_time_{0}".format(ts+1)) for ts in range(lstm_seq_len)]
        flow_inputs = [Input(shape = (nbhd_size, nbhd_size, flow_type,), name = "flow_volume_input_time_{0}".format(ts+1)) for ts in range(lstm_seq_len)]
        lstm_inputs = Input(shape = (lstm_seq_len, feature_vec_len,), name = "lstm_input")

        lstm_input = lstm_inputs
        #lstm
        lstm = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(lstm_input)

        att_lstm_input = att_lstm_inputs


        ####对过去的每一天进行了一个lstm
        att_lstms = [LSTM(units=lstm_out_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, name="att_lstm_{0}".format(att + 1))(att_lstm_input[att]) for att in range(att_lstm_num)]

        #用的attention中compare这一个方法
        #attention 模型的输入为 attlsems[0] ,lstm
        #compare
        att_low_level=[Attention(method='cba')([att_lstms[att], lstm]) for att in range(att_lstm_num)]
        ##计算得到low_level 也就是每一个memory与要查询的单元的吸引力的结果，对应的是一个预测值

        att_low_level=Concatenate(axis=-1)(att_low_level)
        ##然后再把这些值连接在一起，维度得到了增加
        att_low_level=Reshape(target_shape=(att_lstm_num, lstm_out_size))(att_low_level)

        ######最后再进行了一次lstm，相当于把7填的汇总到一起，进行了一个大的lstm
        att_high_level = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(att_low_level)


        ##相当于有8个lstm？？，8个一起求权重
        lstm_all = Concatenate(axis=-1)([att_high_level, lstm])
        ###
        # lstm_all = Dropout(rate = .3)(lstm_all)
        lstm_all = Dense(units = output_shape)(lstm_all)
        pred_volume = Activation('tanh')(lstm_all)


        inputs =  att_lstm_inputs + nbhd_inputs + flow_inputs + [lstm_inputs,]
        # print("Model input length: {0}".format(len(inputs)))
        # ipdb.set_trace()
        model = Model(inputs = inputs, outputs = pred_volume)
        model.compile(optimizer = optimizer, loss = loss, metrics=metrics)
        return model

x=models().stdn(att_lstm_num=7, att_lstm_seq_len=20, lstm_seq_len=48, feature_vec_len=3, cnn_flat_size = 128, lstm_out_size = 128,\
    nbhd_size = 3, nbhd_type = 2)
keras.utils.plot_model(x, 'model_info.png', show_shapes=True)
print(x.summary())
# nbhd_inputs = [Input(shape = (3, 3, 2,), name = "nbhd_volume_input_time_{0}".format(ts+1)) for ts in range(20)]
# flatten_att_nbhd_inputs = [
#     Input(shape=(3, 3, 2,), name="att_nbhd_volume_input_time_{0}_{1}".format(att + 1, ts + 1))
#     for ts in range(20) for att in range(7)]
# z=[5 for i in range(2) for j in range(3)]
# print(1)

