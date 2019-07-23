#-*-coding:utf8-*-
#user:brian
#created_at:2019/7/22 17:16
# file: LSTM-源码学习.py
#location: china chengdu 610000
import numpy as np
class IdentityActivator(object):
    def forward(self,weighted_input):
        return weighted_input
    def backward(self,output):
        return 1
class RuleActivator(object):
    def forward(self,weighted_input):
        return max(0,weighted_input)
    def backwward(self,output):
        return 1 if output>0 else 0


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
        self.wch,self.wcx,self.bc=(self.init_weight_mat())

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
        self.f_list.append(fg)
        #输入门
        ig=self.calc_gate(x,self.wix,self.wih,self.bi,self.gate_activator)
        self.i_list.append(ig)
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
    #反向传播法的实现
    def backwward(self,x,delta_h,activator):
        self.calc_delta(delta_h,activator)
        self.calc_gradient(x)
    #算法分为两个部分，一部分是计算误差项
    def calc_delta(self,delta_h,activator):
        #初始化各个时刻的误差项
        self.delta_h_list=self.init_delta()#输出误差项
        self.delta_o_list=self.init_delta()#输出门误差项
        self.delta_i_list=self.init_delta()#输入门误差项
        self.delta_f_list=self.init_delta()#遗忘门误差项
        self.delta_Ct_list=self.init_delta()#即时输出误差项
        #保存从上一层传递下来的当前时刻的误差项
        self.delta_h_list[-1]=delta_h
        #迭代计算每个时刻的误差
        for k in range(self.times,0,-1):
            self.calc_delta_k(k)

    #初始化误差项
    def init_delta(self):
        delta_list=[]
        for i in range(self.times+1):
            delta_list.append(np.zeros((
                self.state_with,1)))
        return delta_list
    def calc_delta_k(self,k):
        """
        根据K时刻的delta_h,计算K时刻的delta_f
        delta_i,delta_o,delta_Ct,以及K-1时刻的delta_h
        """
        #获得K时刻的前向计算的值
        ig=self.i_list[k]
        og=self.o_list[k]
        fg=self.f_list[k]
        ct=self.ct_list[k]
        c=self.c_list[k]
        tanh_c=self.output_activator.forward(c)
        delta_k=self.delta_h_list[k]
        #根据式9计算delta_o
        delta_o=(delta_k*tanh_c*self.gate_activator.backward(og))

        delta_f=(delta_k*og*(1-tanh_c*tanh_c)*ct*self.gate_activator.backward(ig))

        delta_i=(delta_k*og*(1-tanh_c*tanh_c)*ct*self.gate_activator.backward(ig))

        delta_ct=(delta_k*og*(1-tanh_c*tanh_c)*ig*self.output_activator.backward(ct))

        delta_h_prev=(
            np.dot(delta_o.transpose(),self.woh)+
            np.dot(delta_i.transpose(),self.wih)+
            np.dot(delta_f.transpose(),self.wfh)+
            np.dot(delta_ct.transpose(),self.wch)
        ).transpose()

        #保存全部delta的值
        self.delta_h_list[k-1]=delta_h_prev
        self.delta_f_list[k]=delta_f
        self.delta_i_list[k]=delta_i
        self.delta_o_list[k]=delta_o
        self.delta_Ct_list[k]=delta_ct
    #另一部分是计算梯度
    def calc_gradient(self,x):
        #初始化遗忘门权重梯度矩阵和偏置项
        self.wfh_grad,self.wfx_grad,self.bf_grad=(self.init_weight_gradient_mat())
        #初始化输入们权重梯度矩阵和偏置向量
        self.wih_grad,self.wix_grad,self.bi_grad=(self.init_weight_gradient_mat())
        #初始化输出门权梯度矩阵和偏置向量
        self.woh_grad,self.wox_grad,self.bo_grad=(self.init_weight_gradient_mat())
        #初始化单元状态太权重梯度矩阵和偏置向量
        self.wch_grad,self.wcx_grad,self.bc_grad=(self.init_weight_gradient_mat())
        #计算对上一次输出h的权重梯度


    def init_weight_gradient_mat(self):
        #初始化权重矩阵
        wh_grad=np.zeros((self.state_with,self.state_with))
        wx_grad=np.zeros((  self.state_with,self.input_with))
        b_grad=np.zeros((self.state_with,1))
        return wh_grad,wx_grad,b_grad
    def calc_gradient_t(self,t):
        #计算每时刻t权重的梯度
        h_prew=self.h_list[t-1].transpose()
        wfh_grad=np.dot(self.delta_f_list[t],h_prew)
        bf_grad=self.delta_f_list[t]
        wih_grad=np.dot(self.delta_i_list[t],h_prew)
        bi_grad=self.delta_f_list[t]
        Woh_grad=np.dot(self.delta_o_list[t],h_prew)
        bo_grad=self.delta_f_list[t]
        wch_grad=np.dot(self.delta_Ct_list[t],h_prew)
        bc_grad=self.delta_Ct_list[t]
        return wfh_grad,bf_grad,wih_grad,bi_grad,Woh_grad,bo_grad,wch_grad,bc_grad
    #梯度下降算法实现,更新权重
    def update(self):
        self.wfh=self.learning_rate*self.wch_grad
        self.wfx=self.learning_rate*self.wfh_grad
        self.bf=self.learning_rate*self.bf_grad
        self.wih=self.learning_rate*self.wih_grad
        self.bi=self.learning_rate*self.bi_grad
        self.woh=self.learning_rate*self.woh_grad
        self.wox=self.learning_rate*self.wox_grad
        self.bo=self.learning_rate*self.bo_grad
        self.wch=self.learning_rate*self.wch_grad
        self.wcx=self.learning_rate*self.wcx_grad
    def reset_state(self):
        #当前时刻初始化为0
        self.times=0
        #各个时刻单元状态向量C
        self.c_list=self.init_state_vec()
        self.h_list=self.init_state_vec()
        self.f_list=self.init_state_vec()
        self.i_list=self.init_state_vec()
        self.o_list=self.init_state_vec()
        self.ct_list=self.init_state_vec()
#最后是梯度检查的代码
def data_Set():
    x = [np.array([[1], [2], [3]]),
         np.array([[2], [3], [4]])]
    d = np.array([[1], [2]])
    return x,d
def gradient_check():
    error_function=lambda o:o.sum()
    lstm=LstmLayer(3,2,1e-3)
    #计算forward值
    x,d=data_Set()
    lstm.forward(x[0])
    lstm.forward(x[1])
    #求取senstivity map
    sensitivity_Array=np.ones(lstm.h_list[-1].shape,dtype=np.float64)
    #计算梯度
    lstm.backwward(x[1],sensitivity_Array,IdentityActivator())
    #检查梯度
    epsilon=10e-4
    for i in range(lstm.wfh.shape[0]):
        for j in range(lstm.wfh.shape[1]):
            lstm.wfh[i,j]+=epsilon
            lstm.reset_state()
            lstm.forward(x[0])
            lstm.forward(x[1])
            err1=error_function(lstm.h_list[-1])
            lstm.wfh[i,j]-=2*epsilon
            lstm.reset_state()
            lstm.forward(x[0])
            lstm.forward(x[1])
            err2=error_function(lstm.h_list[-1])
            expect_grad=(err1-err2)/(2*epsilon)
            lstm.wfh[i,j]+=epsilon
            print('weights(%d,%d): expected - actural %.4e - %.4e' % (i, j, expect_grad, lstm.wfh_grad[i, j]))
    return lstm

def test():
    l = LstmLayer(3, 2, 1e-3)
    x, d = data_Set()
    l.forward(x[0])
    l.forward(x[1])
    l.backwward(x[1], d, IdentityActivator())
    return l

gradient_check()

