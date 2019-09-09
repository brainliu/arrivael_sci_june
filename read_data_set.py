#-*-coding:utf8-*-
#user:brian
#created_at:2019/9/6 15:15
# file: read_data_set.py
#location: china chengdu 610000
##从处理好的数据中读取模型的训练数据和测试数据
##前面的处理程序以月份为单位进程处理结果
##主要用3个数
import AM_LSTM_GEN_model as AM_MODEL
import pandas as pd
data_x1=pd.read_csv("./data/6_dataset_x1.csv")
data_x2=pd.read_csv("./data/6_dataset_x2.csv")
data_y=pd.read_csv("./data/6_dataset_y1.csv")

# att_lstm_inputs = [Input(shape=(att_lstm_seq_sita, lstm_num_fai,), name="att_lstm_input_{0}".format(att + 1)) for att in
#                    range(att_lstm_day)]

#多个input 但是shape都是一样的
# name = "w_%d%2d" % (target_day_index-past_day_index, i) #命名规则，后面会用到这个
# w_1 0 ===w_1 20  为一天的 20个  一共过去7天
#用np.array() 来实现输入
flow_att_features=[]
generator_features=[]
for day in range(1,8):
    target_lists=[]##每一天数据对应的索引lists
    for past_time in range(20):
        name="w_%d%2d"%(day,past_time)
        target_lists.append(name)
    print(target_lists)

##目前只有8天的数据，前面7天训练，预测第8天
# model=AM_MODEL.models().stdn(att_lstm_day=7, att_lstm_seq_sita=20, lstm_num_fai=266)
# AM_MODEL.keras.utils.plot_model(model, 'model_info_V2.png', show_shapes=True)
# print(model.summary())
# model.fit()


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