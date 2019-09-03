#-*- coding:utf-8 -*-
#created by brian
# create time :2019/7/24-21:04 
#location: sichuan chengdu
#############################################################################
#####1.泊松分布生成器                                                 #######
#####2.不同的航班时刻能否单独设置为不同的参数，一共有多少个固定的参数 #######
####3，梯度下降法去修改这个参数######能够加速训练，捕获结果           #######
#############################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
########################！！！！先不考虑变化参数的问题####################################

######1.航班计划的解析以及得到每一天各航班时刻下的人数#####################################
#1.1解析一天的航班计划，到标准长度的数组中
def get_hb_schedule(hb_data_one_Day,day,hb_time_all):
    ##处理数据得到所有的航班计划
    sums = hb_data_one_Day["CHECKINTIME"].groupby([hb_data_one_Day["SCHETIME"]]).count()
    index_time = hb_data_one_Day.drop_duplicates("SCHETIME")["SCHETIME"].sort_values()
    index_time = list(index_time)
    index_people = list(sums)
    resut_hb_people = [0 for i in hb_time_all]
    for index in range(len(index_time)):
        time_temp = index_time[index]
        people_temp = index_people[index]
        resut_hb_people[hb_time_all.index(time_temp)] += people_temp
    hb_date = [day for i in hb_time_all ]
    hb_time_and_people = pd.DataFrame({"hb_people": resut_hb_people, "hb_time": hb_time_all,"hb_date":hb_date})
    #print(hb_time_and_people)
    return hb_time_and_people
######1.2.统计得到所有的航班的人数到标准数组长度中去
def get_all_hb_time_people(hb_time_all,filename="./data_ori/flight_filted6.csv"):
    """
    :param hb_time_all:所有的航班时间，去除重复了的
    :param filename: 数据文件，月份
    :return: 返回当月的航班计划，对齐了的航班时间，保证统计输入
    """
    data = pd.read_csv(filename)
    ##将所有数据汇总得到407个航班时间和航班人数，对应不同的航空公司
    #初始化  13日
    hb_data_day_13 = data[data["SCHETIME_date"] == 13]
    result=get_hb_schedule(hb_data_day_13, 13, hb_time_all)
    for day in range(14,28):
        try:

            data_day_temp=get_hb_schedule(data[data["SCHETIME_date"] == day], day, hb_time_all)
            result=result.append(data_day_temp)
            print(result)
        except:
            print("day ==>%s not exist!"%day)
            continue
    result.to_csv("6.csv")
    return result
#####2.历史数据解析，每一天个时刻的人数1440分钟
#2.1统计每分钟的人数，先出来一分钟的人数
def get_minute_one_count(data_temp_one_day,day):
    ##得到某一天的1440的每分钟的人数
    #3输出字段：一个是时间、一个是人数、一个是日期
    result = [0 for i in range(1440)]
    arrival_time = list(data_temp_one_day["CHECKINTIME"].values)
    for i in arrival_time:
        result[i] += 1
    hb_date=[day for i in range(1440)]
    result_all=pd.DataFrame({"hb_date":hb_date,"time":[i for i in range(1440)],"hb_people":result})
    return result_all
#2.2得到所有天的X分钟到达人数,并聚合在一起，汇总成一张表
def get_minute_one_count_all(filename="./data_ori/flight_filted6.csv"):
    data = pd.read_csv(filename)
    data_temp_one_day = data[data["SCHETIME_date"] == 13]
    result=get_minute_one_count(data_temp_one_day,13)
    for day_index in range(14,28):
        try:
            result=result.append(get_minute_one_count(data[data["SCHETIME_date"] == day_index],day_index))
        except:
            continue
    return result

######3，数据聚合，按照5分钟，10分钟，15分钟，30分钟 4个级别来进行聚合######################
def get_aggravte_minute():
    ##得到每分钟的聚合人数
    pass

######4.计算过去w天的数据，整合在一起#######################################################
def get_inputs_past_w_day_data():
    ##得到过去w天的数据
    ##根据不同的分钟数作为输入数据
    ##配置：单个序列的长度、以及每个吸引力的长度
    pass

##############5.航班规律到达函数############################################################
#单航班到达规律生成器
def possion_generator(hb_time,hb_people):
    ###可能存在需要训练的参数
    ###1） 时间比例和人数比例 一共165分钟，w1+w2+w3=1  w2变化  w1+w3=1-w2    *165
    ###2) 人数比例                         w1+w2+w3=1  w2变化   w1=w3=（1-w2）/2  *hb_people
    time_people=[0 for i in range(1440)]
    ##分成三段叠加
    #时间上分段45-210分钟，1==》45-70  2===》 70-150  3===》150-210  三段之间的比例计算为
    ##                       0.25         0.5           0.25
    ##先生成到达时间，从第一段低密度开始，最远的地方，距离航班起飞最远到最近45分钟，截至人数全部到达
    # 初始化三个时间区间长度
    ##整体的时间长度，后面通过泊松分布算出来的时间分布可能高于这个值呢
    if hb_people>0:
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
    else:
        return time_people
###多航班叠加 统计每一天的所有航班的总人数
def multi_flight(hb_time_list,hb_people_list):
    """
    :param hb_time_list: 某一天的所有航班的航班时间
    :param hb_people_list: 某一天的所有的航班的人数
    :return:某一天的航班计划得到的航班人数  1440*1 维度为list
    """
    result2=[0 for i in range(1440)]
    for i in range(len(hb_time_list)):
        temp=possion_generator(hb_time_list[i],hb_people_list[i])
        for j in range(1440):
            result2[j]+=temp[j]
    return result2


#######6.画图对比函数#######################################################################
def plot(data,name):
    y = range(1440)
    plt.plot(y, data, ls='dashed',
             lw=2, c='r', label='Poisson distribution\n$(\lambda=%s)$'%name)
    plt.xlim(200,400)
    plt.savefig("test.png")
    plt.show()

############7.测试###########################################################################
def Tst():
    data = pd.read_csv("./data_ori/flight_filted6.csv")
    hb_schedule = data.drop_duplicates(subset=["SCHETIME"])
    hb_schedule_all = hb_schedule[["SCHETIME"]]
    # .to_csv("./data_ori/hb_all_table.csv")
    ##6月份有407个航班 里面可能有重复的时间 200个时间有航班，只计算航班时间对应人数就行了，简化计算一下
    # 对每一天生成一个list 长度为407的 代表这407个航班是否存在的序列
    hb_time_all = sorted(list(hb_schedule_all["SCHETIME"]))  ##拍好序的所有航班时间的数据
    get_all_hb_time_people(hb_time_all,"./data_ori/flight_filted6.csv")
    # test_peoples=possion_generator(400,500)
    # print(test_peoples)
    # plot(test_peoples,"500ren")
    result = get_minute_one_count_all(filename="./data_ori/flight_filted6.csv")
    result.to_csv("./data_ori/everyday.csv")
#Tst()




