#-*- coding:utf-8 -*-
#created by brian
# create time :2019/7/24-21:04 
#location: sichuan chengdu
#处理成1440的数据
#先用一分钟的数据来进行测试，然后再考虑5分钟的模型，生成器方面也可以进行优化，泊松分布叠加比如这种
#再去优化泊松分布的参数w
#以及引入attention机制模型，截取过去7天周期性变化的特征
#最后得到预测结果


#输入数据有 航班时刻，航班人数，先只做国内航班
#两种思路，一种是只输入航班时刻，不输入人数，人数是根据学习规则来计算的，另一种是输入人数，推荐第一种方法

#################第一步处理数据###########
###########first step deal orinignal data which contains flight time/yeater1-7day people
################第二步构建模型###########
#2019-0728 进行公式推导和思路的明确
# 主要公式推导：
#A 生成器模型
#B LSTM模型1
#C 吸引力模型
#Temporal dynamic periodic similarity: periodically shifted attention mechanism

#joint trainning 联合训练
# optimization 优化器
###2019-0806 本周务必要把公式推导写完
##下周8月12日开始写程序来实现这个算法
##fighting