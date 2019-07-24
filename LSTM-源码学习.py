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
        #各时刻的输出向量h
        self.h_list=self.init_state_vec()
        #各时刻的遗忘门f
        self.f_list=self.init_state_vec()
        #各时刻的输入门
        self.i_list=self.init_state_vec()
        #各时刻的输出门
        self.o_list=self.init_state_vec()
        #各时刻的即时状态
        self.ct_list=self.init_state_vec()

        #遗忘门权重矩阵wfh，wfx，偏置项bf
        self.wfh,self.wfx,self.bf=(self.init_weight_mat())
        #输入门权重矩阵wif，wix，偏置项bi
        self.wih,self.wix,self.bi=(self.init_weight_mat())
        #输出门权重矩阵wof，wox，偏置项bo
        self.woh,self.wox,self.bo=(self.init_weight_mat())

        #单元状态权重矩阵wch,wcx偏置项bc
        self.wch,self.wcx,self.wbc=(self.init_weight_mat())

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

    #forward 方法实lstm的前向计算
    def forward(self,x):
        """
        根据1-式-6式进行前向计算
        :param self:
        :param x:
        :return:
        """
        self.times+=1
        #遗忘门
        fg=self.calc_gate(x,self.wfx,self.wfh,self.bf,self.gate_activator)
        #输入门
        ig=self.calc_gate(x,self.wix,self.wih,self.bi,self.gate_activator)
        #输出门
        og=self.calc_gate(x,self.wox,self.woh,self.bo,self.gate_activator)

        #各时刻的输出门添加到list中去记录下来
        self.o_list.append(og)

        #即时状态
        ct=self.calc_gate(x,self.wcx,self.wch,self.bc,self.output_activator)
        self.ct_list.append(ct)

        #单元状态
        c=fg*self.ct_list[self.times-1]+ig*ct
        self.c_list.append(c)
        #输出
        h=og*self.output_activator.forward(c)
        self.h_list.append(h)


    def calc_gate(self,x,wx,wh,b,activator):
        """计算门"""
        h=self.h_list[self.times-1]#上次lstm的输出
        net=np.dot(wh,h)+np.dot(wx,x)+b
        gate=activator.forward(net)
        return gate
