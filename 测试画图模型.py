#-*- coding:utf-8 -*-
#created by brian
# create time :2019/8/28-20:09 
#location: sichuan chengdu
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
import tensorflow.keras as keras
from tensorflow.python.keras.layers import LSTM
from numpy import array
from numpy import hstack
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input,Dense,concatenate


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
# separate input data
X1 = X[:, :, 0]
X2 = X[:, :, 1]
# first input model
visible1 = Input(shape=(n_steps,))
dense1 = Dense(100, activation='relu')(visible1)
# second input model
sess=tf.Session()
sess.run(tf.global_variables_initializer())
visible2 = Input(shape=(n_steps,))
dense2 = Dense(100, activation='relu')(visible2)
dense3_1=tf.convert_to_tensor(dense1[0:50].eval())
dense3_2=tf.convert_to_tensor(dense2[1:51])
dense3_3=tf.convert_to_tensor(dense1[20:70])
# merge input models
merge = concatenate([dense1, dense2])

merge1=concatenate([dense3_1,dense3_2])
merge2=concatenate([dense3_1,dense3_3])
merge3=concatenate([merge1,merge2])

output = Dense(1)(merge1)
model = Model(inputs=[visible1, visible2], outputs=output)
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit([X1, X2], y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([[80, 85], [90, 95], [100, 105]])
x1 = x_input[:, 0].reshape((1, n_steps))
x2 = x_input[:, 1].reshape((1, n_steps))
yhat = model.predict([x1, x2], verbose=0)
keras.utils.plot_model(model, 'model_info3.png', show_shapes=True)
print(model.summary())
print(yhat)

