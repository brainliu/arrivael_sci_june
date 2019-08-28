#-*- coding:utf-8 -*-
#created by brian
# create time :2019/8/28-22:17 
#location: sichuan chengdu
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
import tensorflow.keras.backend  as K
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, concatenate, Input, Conv2D, Reshape, Flatten, Dropout, BatchNormalization, Concatenate, LSTM
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
input_x = tf.keras.Input(shape=(1,72))
att_lstms = LSTM(units=48, return_sequences=False, dropout=0.1, recurrent_dropout=0.1,
                  name="att_lstm_")(input_x)

att_lstm2 = LSTM(units=48, return_sequences=False, dropout=0.1, recurrent_dropout=0.1,
                  name="att_lstm_")(input_x)

