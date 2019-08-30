#-*-coding:utf8-*-
#user:brian
#created_at:2019/8/30 21:53
# file: tf_simple_lstm.py
#location: china chengdu 610000
# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
import tensorflow.keras.backend  as K
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.python.keras.layers import Dense, Activation, concatenate, Input, Conv2D, Reshape, Flatten, Dropout, BatchNormalization, Concatenate, LSTM
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
