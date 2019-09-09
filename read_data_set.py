#-*-coding:utf8-*-
#user:brian
#created_at:2019/9/6 15:15
# file: read_data_set.py
#location: china chengdu 610000
##从处理好的数据中读取模型的训练数据和测试数据
##前面的处理程序以月份为单位进程处理结果
##主要用3个数
import AM_LSTM_GEN_model as AM_MODEL
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data_x1=pd.read_csv("./data/6_dataset_x1.csv")
data_x2=pd.read_csv("./data/6_dataset_x2.csv")
data_y=pd.read_csv("./data/6_dataset_y1.csv")


def plot_result(data_ori,data_pred):
    y = range(0,268)
    plt.plot(y, data_ori, ls='dashed',
             lw=2, c='r', label='true_y')
    plt.plot(data_pred, ls='dashed',
             lw=2, c='b', label='data_pred')
    plt.savefig("test2.png")
    plt.show()

# att_lstm_inputs = [Input(shape=(att_lstm_seq_sita, lstm_num_fai,), name="att_lstm_input_{0}".format(att + 1)) for att in
#                    range(att_lstm_day)]

#多个input 但是shape都是一样的
# name = "w_%d%2d" % (target_day_index-past_day_index, i) #命名规则，后面会用到这个
# w_1 0 ===w_1 20  为一天的 20个  一共过去7天
#用np.array() 来实现输入
flow_att_features=[]
generator_features=np.array(data_x2["hb_people"].values) #.reshape(8,1,268)##这里不能用简单的Reshape
temp_gen=[]
for i in range(8):
    temp_day =[]
    for j in range(268):
        temp_day.append([generator_features[i*268+j]])
    temp_gen.append(temp_day)

generator_features=np.array(temp_gen) #生成结果shanpe= 8,268,1

for day in range(1,8):
    target_lists=[]##每一天数据对应的索引lists
    for past_time in range(20):
        name="w_%d%2d"%(day,past_time)
        target_lists.append(name)
    print(target_lists)
    past_Day_temp=data_x1[target_lists]
    past_Day_temp=past_Day_temp.as_matrix().reshape(8, 268, 20)#.reshape(8,20,268) ##这里读取数据有点问题
    ##this is 8 day  20 col and 268 lines
    flow_att_features.append(past_Day_temp)


target_y=data_y["0"].as_matrix().reshape(8,268)

##测试数据
test_x1=[[flow_att_features[i][7]] for i in range(7)]
test_x2=generator_features[7]
true_y=target_y[7]
# target_y=target_y.reshape(1,len(target_y))
# print(target_y)
##目前只有8天的数据，前面7天训练，预测第8天
model=AM_MODEL.models().stdn(att_lstm_day=7, att_lstm_seq_sita=20, lstm_num_fai=268)
# AM_MODEL.keras.utils.plot_model(model, 'model_info_V2.png', show_shapes=True)
model.summary()
model.fit(x=flow_att_features+[generator_features,],y=target_y,batch_size=77,epochs=1)

y_pred = model.predict(x=test_x1+[test_x2,],batch_size=77)

plot_result(true_y,y_pred)

###输入有两个，一个是past_w_day 个 lstm  一个是 generator生成的
#inputs =  att_lstm_inputs  + [lstm_inputs,]
# model = Model(inputs = inputs, outputs = pred_volume)

# model.fit( \
#     x=att_cnnx + att_flow + att_x + cnnx + flow + [x, ], \
#     y=y, \
#     batch_size=batch_size, validation_split=validation_split, epochs=max_epochs, callbacks=[early_stop])
#
# att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(datatype="test", nbhd_size=args.nbhd_size,
#                                                                   cnn_nbhd_size=args.cnn_nbhd_size)
# y_pred = model.predict( \
#     x=att_cnnx + att_flow + att_x + cnnx + flow + [x, ], )