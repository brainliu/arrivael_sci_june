#-*-coding:utf8-*-
#user:brian
#created_at:2019/7/22 17:16
# file: LSTM-源码学习.py
#location: china chengdu 610000
import numpy as np

class  SigmoidActivator(object):
    def forward(self,weighted_input):
        return 1.0/(1.0+np.exp(-weighted_input))
    def backward(self,output):
        return output*(1-output)

class Tanhactivator(object):
    def forward(self,weighted_input):
        return 2.0/(1.0+np.exp(-2*weighted_input))-1.0
    def backward(self,output):
        return 1-output*output
class LstmLayer(object):
    def __init__(self,input_width,state_with,learning_rate):
        self.input_with=input_width
        self.state_with=state_with
        self.learning_rate=learning_rate
        #门的激活函数
        self.gate_activator=SigmoidActivator()
        #输出的激活函数
        self.output_activator=Tanhactivator()

        #当前时刻初始化为t0
        self.times=0
        #各个时刻单元状态向量C
        self.c_list=self.init_state_vec()

    def init_state_vec(self):
        '''
        初始化保存状态的向量嗯
        '''
        state_vec_list = []
        state_vec_list.append(np.zeros(
            (self.state_with, 1)))
        return state_vec_list

    def init_weight_mat(self):
        '''
        初始化权重矩阵
        '''
        Wh = np.random.uniform(-1e-4, 1e-4,
                               (self.state_with, self.state_with))
        Wx = np.random.uniform(-1e-4, 1e-4,
                               (self.state_with, self.input_with))
        b = np.zeros((self.state_with, 1))
        return Wh, Wx, b