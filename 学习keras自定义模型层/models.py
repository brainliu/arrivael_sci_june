import pickle
import numpy as np
import json
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Activation, concatenate, Input, Conv2D, Reshape, Flatten, Dropout, BatchNormalization, Concatenate, LSTM
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
import ipdb
import attention

class baselines:
    def __init__(self):
        pass


###主模型函数
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
        flatten_att_nbhd_inputs = [Input(shape = (nbhd_size, nbhd_size, nbhd_type,), name = "att_nbhd_volume_input_time_{0}_{1}".format(att+1, ts+1)) for ts in range(att_lstm_seq_len) for att in range(att_lstm_num)]
        #对应的流动门限个数flow gate 机制
        flatten_att_flow_inputs = [Input(shape = (nbhd_size, nbhd_size, flow_type,), name = "att_flow_volume_input_time_{0}_{1}".format(att+1, ts+1)) for ts in range(att_lstm_seq_len) for att in range(att_lstm_num)]
        ##每一个序列的长度为att_lstm_seq_len 也就是48个时间刻度 一共有7天的lstm数据att_lstm_num

        att_nbhd_inputs = []
        att_flow_inputs = []
        for att in range(att_lstm_num):
            att_nbhd_inputs.append(flatten_att_nbhd_inputs[att*att_lstm_seq_len:(att+1)*att_lstm_seq_len])
            att_flow_inputs.append(flatten_att_flow_inputs[att*att_lstm_seq_len:(att+1)*att_lstm_seq_len])

        #假设为7，表示由7个吸引能力的输入，每一个输入长度为att_lstm_seq_len
        att_lstm_inputs = [Input(shape = (att_lstm_seq_len, feature_vec_len,), name = "att_lstm_input_{0}".format(att+1)) for att in range(att_lstm_num)]

        nbhd_inputs = [Input(shape = (nbhd_size, nbhd_size, nbhd_type,), name = "nbhd_volume_input_time_{0}".format(ts+1)) for ts in range(lstm_seq_len)]
        flow_inputs = [Input(shape = (nbhd_size, nbhd_size, flow_type,), name = "flow_volume_input_time_{0}".format(ts+1)) for ts in range(lstm_seq_len)]
        lstm_inputs = Input(shape = (lstm_seq_len, feature_vec_len,), name = "lstm_input")

        #short-term part
        #1st level gate
        #nbhd cnn
        nbhd_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "nbhd_convs_time0_{0}".format(ts+1))(nbhd_inputs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu", name = "nbhd_convs_activation_time0_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        #flow cnn
        flow_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "flow_convs_time0_{0}".format(ts+1))(flow_inputs[ts]) for ts in range(lstm_seq_len)]
        flow_convs = [Activation("relu", name = "flow_convs_activation_time0_{0}".format(ts+1))(flow_convs[ts]) for ts in range(lstm_seq_len)]
        #flow gate
        flow_gates = [Activation("sigmoid", name = "flow_gate0_{0}".format(ts+1))(flow_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(lstm_seq_len)]


        #2nd level gate
        nbhd_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "nbhd_convs_time1_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu", name = "nbhd_convs_activation_time1_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        flow_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "flow_convs_time1_{0}".format(ts+1))(flow_inputs[ts]) for ts in range(lstm_seq_len)]
        flow_convs = [Activation("relu", name = "flow_convs_activation_time1_{0}".format(ts+1))(flow_convs[ts]) for ts in range(lstm_seq_len)]
        flow_gates = [Activation("sigmoid", name = "flow_gate1_{0}".format(ts+1))(flow_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(lstm_seq_len)]

        #3rd level gate
        nbhd_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "nbhd_convs_time2_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu", name = "nbhd_convs_activation_time2_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        flow_convs = [Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "flow_convs_time2_{0}".format(ts+1))(flow_inputs[ts]) for ts in range(lstm_seq_len)]
        flow_convs = [Activation("relu", name = "flow_convs_activation_time2_{0}".format(ts+1))(flow_convs[ts]) for ts in range(lstm_seq_len)]
        flow_gates = [Activation("sigmoid", name = "flow_gate2_{0}".format(ts+1))(flow_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(lstm_seq_len)]


        #dense part
        nbhd_vecs = [Flatten(name = "nbhd_flatten_time_{0}".format(ts+1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_vecs = [Dense(units = cnn_flat_size, name = "nbhd_dense_time_{0}".format(ts+1))(nbhd_vecs[ts]) for ts in range(lstm_seq_len)]
        nbhd_vecs = [Activation("relu", name = "nbhd_dense_activation_time_{0}".format(ts+1))(nbhd_vecs[ts]) for ts in range(lstm_seq_len)]

        #feature concatenate
        nbhd_vec = Concatenate(axis=-1)(nbhd_vecs)
        nbhd_vec = Reshape(target_shape = (lstm_seq_len, cnn_flat_size))(nbhd_vec)
        ###吸引力模型的一个输入.................
        lstm_input = Concatenate(axis=-1)([lstm_inputs, nbhd_vec])

        #lstm
        lstm = LSTM(units=lstm_out_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(lstm_input)

        #attention part
        att_nbhd_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_nbhd_convs_time0_{0}_{1}".format(att+1,ts+1))(att_nbhd_inputs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[Activation("relu", name = "att_nbhd_convs_activation_time0_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_flow_convs_time0_{0}_{1}".format(att+1,ts+1))(att_flow_inputs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Activation("relu", name = "att_flow_convs_activation_time0_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate0_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]

        att_nbhd_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_nbhd_convs_time1_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[Activation("relu", name = "att_nbhd_convs_activation_time1_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_flow_convs_time1_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Activation("relu", name = "att_flow_convs_activation_time1_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate1_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]

        att_nbhd_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_nbhd_convs_time2_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[Activation("relu", name = "att_nbhd_convs_activation_time2_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Conv2D(filters = 64, kernel_size = (3,3), padding="same", name = "att_flow_convs_time2_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_convs = [[Activation("relu", name = "att_flow_convs_activation_time2_{0}_{1}".format(att+1,ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_flow_gates = [[Activation("sigmoid", name = "att_flow_gate2_{0}_{1}".format(att+1, ts+1))(att_flow_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_convs = [[keras.layers.Multiply()([att_nbhd_convs[att][ts], att_flow_gates[att][ts]]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]

        att_nbhd_vecs = [[Flatten(name = "att_nbhd_flatten_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_convs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_vecs = [[Dense(units = cnn_flat_size, name = "att_nbhd_dense_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_vecs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]
        att_nbhd_vecs = [[Activation("relu", name = "att_nbhd_dense_activation_time_{0}_{1}".format(att+1,ts+1))(att_nbhd_vecs[att][ts]) for ts in range(att_lstm_seq_len)] for att in range(att_lstm_num)]


        att_nbhd_vec = [Concatenate(axis=-1)(att_nbhd_vecs[att]) for att in range(att_lstm_num)]
        att_nbhd_vec = [Reshape(target_shape = (att_lstm_seq_len, cnn_flat_size))(att_nbhd_vec[att]) for att in range(att_lstm_num)]
        att_lstm_input = [Concatenate(axis=-1)([att_lstm_inputs[att], att_nbhd_vec[att]]) for att in range(att_lstm_num)]


        ####对过去的每一天进行了一个lstm
        att_lstms = [LSTM(units=lstm_out_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, name="att_lstm_{0}".format(att + 1))(att_lstm_input[att]) for att in range(att_lstm_num)]

        #用的attention中compare这一个方法
        #attention 模型的输入为 attlsems[0] ,lstm
        #compare
        att_low_level=[attention.Attention(method='cba')([att_lstms[att], lstm]) for att in range(att_lstm_num)]
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

        inputs = flatten_att_nbhd_inputs + flatten_att_flow_inputs + att_lstm_inputs + nbhd_inputs + flow_inputs + [lstm_inputs,]
        # print("Model input length: {0}".format(len(inputs)))
        # ipdb.set_trace()
        model = Model(inputs = inputs, outputs = pred_volume)
        model.compile(optimizer = optimizer, loss = loss, metrics=metrics)
        return model