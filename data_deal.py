#-*- coding:utf-8 -*-
#created by brian
# create time :2019/7/24-21:04 
#location: sichuan chengdu
#####1.泊松分布生成器
#####2.不同的航班时刻能否单独设置为不同的参数，一共有多少个固定的参数
####3，梯度下降法去修改这个参数######能够加速训练，捕获结果

import numpy as np
import matplotlib.pyplot as plt
########################！！！！先不考虑变化参数的问题###############################

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

    time_people=[0 for i in range(1440)]
    ##分成三段叠加
    #时间上分段45-210分钟，1==》45-70  2===》 70-150  3===》150-210  三段之间的比例计算为
    ##                       0.25         0.5           0.25
    ##先生成到达时间，从第一段低密度开始，最远的地方，距离航班起飞最远到最近45分钟，截至人数全部到达
    # 初始化三个时间区间长度
    ##整体的时间长度，后面通过泊松分布算出来的时间分布可能高于这个值呢
    ##从第45分钟开始
    time_interval_low1=25
    time_interval_high2=80
    time_interval_low3=60
    ###上面三个为时间的跨度
    time_scale_last=[0,0,0]   #记录最后一个值为多少
    times_scale_chen=[0,0,0]
    time_arrival_all=[]       #记录每个人的到达时间间隔时间，一共有多少个人人就有多少个数据，最后再把这些数据映射到1440上面去！
    #高密度和低密度的比例
    low_percent=0.25
    high_percent=0.5
    low_people_1=int(low_percent*hb_people)
    high_peole_2 = int(high_percent * hb_people)
    low_people_3=low_people_1

    ##计算每个人的到达时间  距离起飞还有45分钟-80分钟的人数的到达时间
    temp_arrival_time1=hb_time-210+45 #每一段的开始时间 （第一段）
    temp_arrival_time2=temp_arrival_time1+time_interval_low1  #（第二段 高密度，开始时间）
    temp_arrival_time3=temp_arrival_time2+time_interval_high2  #（第三段， 低密度，开始时间)
    result=[]

    for i in range(low_people_1):
        next_times_Arrival=np.random.exponential(1)
        time_arrival_all.append(next_times_Arrival)
        time_scale_last[0]+=next_times_Arrival
    for i in range(high_peole_2):
        next_times_Arrival = np.random.exponential(1)
        time_arrival_all.append(next_times_Arrival)
        time_scale_last[1] += next_times_Arrival
    for i in range(low_people_3):
        next_times_Arrival = np.random.exponential(1)
        time_arrival_all.append(next_times_Arrival)
        time_scale_last[2] += next_times_Arrival
    ##将得到的规模进行转换
    times_scale_chen[0]=float(time_interval_low1/time_scale_last[0])
    times_scale_chen[1] = float(time_interval_high2 / time_scale_last[1])
    times_scale_chen[2] = float(time_interval_low3 / time_scale_last[2])
    for j in range(3):
        print(times_scale_chen[j])

    for i in range(low_people_1):
        temp_arrival_time1+=time_arrival_all[i]*times_scale_chen[0]
        result.append(temp_arrival_time1)
    for i in range(low_people_1,high_peole_2+low_people_1):
        temp_arrival_time2+=time_arrival_all[i]*times_scale_chen[1]
        result.append(temp_arrival_time2)
    for i in range(high_peole_2+low_people_1,hb_people):
        temp_arrival_time3+=time_arrival_all[i]*times_scale_chen[2]
        result.append(temp_arrival_time3)
    ##现在的result就是每个人的到达时间，下一步就是进行排序和统计每分钟的人数：
    for k in result:
        time_people[int(k)]+=1
    return time_people
###多航班叠加
def multi_flight(hb_time_list,hb_people_list):
    result2=[0 for i in range(1440)]
    for i in range(len(hb_time_list)):
        temp=possion_generator(hb_time_list[i],hb_people_list[i])
        for j in range(1440):
            result2[j]+=temp[j]
    return result2


#######3.画图对比函数#################################################
def plot(data,name):
    y = range(1440)
    plt.plot(y, data, ls='dashed',
             lw=2, c='r', label='Poisson distribution\n$(\lambda=%s)$'%name)
    #plt.xlim(200,400)
    plt.savefig("test.png")
    plt.show()


test_peoples=possion_generator(400,500)
print(test_peoples)
plot(test_peoples,"500ren")






