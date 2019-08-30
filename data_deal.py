#-*- coding:utf-8 -*-
#created by brian
# create time :2019/7/24-21:04 
#location: sichuan chengdu
#####1.泊松分布生成器
#####2.不同的航班时刻能否单独设置为不同的参数，一共有多少个固定的参数
####3，梯度下降法去修改这个参数######能够加速训练，捕获结果

########################先不考虑变化参数的问题###############################

######1.历史数据解析以及对过去的数据进行统计################################
######1.统计得到每一天每分钟的航班人数
def get_minute_one_count():
    ##实现一个参数，也就是5分钟、10分钟、15分钟、30分钟 多种组合规则的提取程序
    #3输出字段：一个是时间、一个是人数、一个是日期
    pass

def get_inputs_past_w_day_data():
    ##得到过去w天的数据
    ##根据不同的分钟数作为输入数据
    ##配置：单个序列的长度、以及每个吸引力的长度
    pass






##############2.航班规律到达函数##########################
#单航班到达规律生成器
def possion_generator(hb_time,hb_people):
    result=[0 for i in range(1440)]
    ##
    return result
###多航班叠加
def multi_flight(hb_time_list,hb_people_list):
    result2=[0 for i in range(1440)]
    for i in range(len(hb_time_list)):
        temp=possion_generator(hb_time_list[i],hb_people_list[i])
        for j in range(1440):
            result2[j]+=temp[j]
    return result2











