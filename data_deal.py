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
#####1.1解析一天的航班计划，到标准长度的数组中
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
           # print(result)
        except:
            print("day ==>%s not exist!"%day)
            continue
    result.to_csv("./data/6_hb_flight.csv")
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


######3，数据聚合，按照5分钟，10分钟get_arrtavte_minutes_one，15分钟，30分钟 4个级别来进行聚合######################
#得到了在day_index下的航班人数聚合，下一步是将每一天的数据叠加在一起
def get_arrtavte_minutes_one(hb_people,interval,day_index):
    hb_people_result= [0 for i in range(int(1440 / interval))]
    hb_date_result=[day_index for i in range(int(1440/ interval))]
    time_result=[i for i in range(int(1440 / interval))]
    for i in range(len(hb_people)):
        index=int(i/interval) #获取聚合后的新的时间长度
        hb_people_result[index]+=hb_people[i]
    #返回一个dataframe
    return  pd.DataFrame({"hb_date":hb_date_result,"hb_people":hb_people_result,"time":time_result})
#得到每一天再不同的时间间隔内的各个时刻的人数
def get_arrtavte_minutes_all(data_file="./data/everyday.csv",interval=5):
    ##得到每分钟的聚合人数
    data = pd.read_csv(data_file)
    # 一共有多少天的数据需要处理
    day_lists = sorted(list(data.drop_duplicates("hb_date")["hb_date"].values))
    result_all = get_arrtavte_minutes_one(list(data[data["hb_date"] == day_lists[0]]["hb_people"].values), interval,
                                          day_lists[0])
    for day_index in day_lists[1:]:
        print(day_index)
        # 处理成结果的数据的时间区间
        result_all = result_all.append(
            get_arrtavte_minutes_one(list(data[data["hb_date"] == day_index]["hb_people"].values), interval, day_index))
    result_all.to_csv("./data/everyday_interval_%d.csv" % interval)

######4.计算过去w天的数据，整合在一起#######################################################
##按照5分钟划分间隔 一共有288个点   10分钟划分间隔 144个点 预测6点--24点  5分钟就是 18*12=216个点 10分钟就是108个点  15分钟就是18*4=72个点
    ##5分钟 每个航班时间区间跨度假设2小时=24，那就是24个短期特征长度 sita=24
    ## att_lstm_num:吸引力的长度 也就是一共有多少天 长期吸引力天 w=7天
    ##att_lstm_seg_len:每个吸引子的序列长度 短期吸引力短期特征长度 sitar=24个
    ##吸引力个数：也就是要预测的个数  φ=216个点，表示预测未来的216个点
    ##长期时间效应趋势用lstm来捕捉，得到一个h值，短期则用h跟他相关的过去再来捕捉一次sita=24，表示与短期的24个相关

    ###参数指标组合（时间间隔为5分钟==> w=7 sita=24 φ=216）
    ###           （时间间隔为10分钟==> w=7 sita=12 φ=108）
    ###                       15        w=7 sita=8  φ=72
    ###                       30        w=7 sita=4  φ=36
def get_inputs_past_w_day_data_one(data,target_day_index,past_w_day,past_q_time,orignial_length):
    """
    :param target_day_index 目标日的取值，也就是y变量
    :param data:包含所有数据的DataFrame，而且是包含了目标日的过去7天的
    :param past_w_day: 过去的某几天，这里数据比较少，13-28，那就只能用20-28这几天，如果选取7天的话
    :param past_q_time: 过去的q个时刻，也就是288个时间间隔的时候，选取5
    :param orignial_length:
    :return: result 过去w天的提取好的数据,target_day_result 目标数据的值
    #####name = "w_%s_%d" % (past_day_index, i) #命名规则，后面会用到这个
    """
    day_lists = sorted(set(list(data['hb_date'].values)))  # all days
    #last_day_index = len(day_lists)
    # 直接指定第8天，来看过去的7天
    target_data=data[data["hb_date"] == day_lists[target_day_index]] ["hb_people"].values
    target_day_result = []
    #加入指定长度的
    for i in range(past_q_time,orignial_length):
        target_day_result.append(target_data[i])
    result = dict()
    for past_day_index in range(target_day_index-past_w_day, target_day_index): #循环过去的7天，距离目标日的
        data_temp = data[data["hb_date"] == day_lists[past_day_index]]  # 取出当天的data计算
        hb_people_temp = list(data_temp["hb_people"].values)
        for i in range(past_q_time):  # 过去的几个时刻
            name = "w_%d%2d" % (target_day_index-past_day_index, i) #命名规则，后面会用到这个
            result[name] = []
            # 从第0个开始计算，0就是0-267 也就是 i->orignial_length-past_q_time-1+i 长度均为orignial_length-past_q_time
            for j in range(i, orignial_length - past_q_time + i):
                (result[name]).append(hb_people_temp[j])
    #pd.DataFrame(result).to_csv("./data/past_w.csv")
    return pd.DataFrame(result),pd.DataFrame(target_day_result)

def get_inputs_past_w_day_data_all(data_file="./data/everyday_interval_5.csv",past_w_day=7,past_q_time=20,orignial_length=288):
    ##得到过去w天的数据，根据不同的分钟数作为输入数据，配置：单个序列的长度、以及每个吸引力的长度
    ###根据past_q_time和总的长度，判断，整体的数据size=（orignial_length-past_q_time，past_q_time)
    #先处理一天的数据
    data = pd.read_csv(data_file)
    data_x1, data_y = get_inputs_past_w_day_data_one(data, 7, past_w_day, past_q_time, orignial_length)  # 得到某天的结果数据
    day_lists = sorted(set(list(data['hb_date'].values)))  # all days 所有的天
    for target_day_index in range(8,len(day_lists)):
        print("DEALING====》%s"%day_lists[target_day_index])
        data_x_temp,data_y_temp=get_inputs_past_w_day_data_one(data,target_day_index,past_w_day,past_q_time,orignial_length)
        data_x1=data_x1.append(data_x_temp)
        data_y=data_y.append(data_y_temp)
    data_x1.to_csv("./data/6_dataset_x1.csv")
    data_y.to_csv("./data/6_dataset_y1.csv")
    return data_x1,data_y

# get_inputs_past_w_day_data_all()
##############5.航班规律到达函数############################################################
#5.1单航班到达规律生成器
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
    if hb_people>10:
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
        for i in range(high_peole_2+low_people_1,high_peole_2+2*low_people_1):
            temp_arrival_time3+=time_arrival_all[i]*times_scale_chen[2]
            result.append(temp_arrival_time3)
        ##现在的result就是每个人的到达时间，下一步就是进行排序和统计每分钟的人数：
        for k in result:
            time_people[int(k)]+=1
        return time_people
    else:
        return time_people
###5.2多航班叠加 统计每一天的所有航班的总人数
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
def get_arrtavte_minutes_one_hb(hb_people,interval,day_index,past_q_time):
    hb_people_result= [0 for i in range(int(1440 / interval))]
    hb_date_result=[day_index for i in range(past_q_time,int(1440/ interval))]
    time_result=[i for i in range(past_q_time,int(1440 / interval))]
    for i in range(len(hb_people)):
        index=int(i/interval) #获取聚合后的新的时间长度
        hb_people_result[index]+=hb_people[i]
    #提取20-288之间的数据 除掉前面20个时刻的数据
    hb_people_result_f=[]
    for j in range(past_q_time,int(1440/ interval)):
        hb_people_result_f.append(hb_people_result[j])
    #返回一个dataframe
    return  pd.DataFrame({"hb_date":hb_date_result,"hb_people":hb_people_result_f,"time":time_result})

##5.3得到数据的所有航班的总人数，并均分到每时刻
def get_all_day_possion_flight_people():

    pass

target_Day=7
past_q_time=20
data=pd.read_csv("./data/6_hb_flight.csv")
days_list=sorted(list(set(data["hb_date"].values)))
hb_flight_Data=data[data["hb_date"]==days_list[target_Day]] #这里选择第七天开始计算
data_x2=multi_flight(list(hb_flight_Data["hb_time"].values),list(hb_flight_Data["hb_people"].values))
###转化为时间间隔5分钟
data_x2=get_arrtavte_minutes_one_hb(data_x2,5,days_list[target_Day],past_q_time)
##德奥所有天的
for hb_date_index in range(target_Day+1,len(days_list)):
    hb_flight_Data_temp = data[data["hb_date"] == days_list[hb_date_index]]  # 这里选择第七天开始计算
    data_x2_temp=multi_flight(list(hb_flight_Data_temp["hb_time"].values),list(hb_flight_Data_temp["hb_people"].values))
    ##聚合成5分钟,并去除前面20个
    data_x2_temp=get_arrtavte_minutes_one_hb(data_x2_temp,5,days_list[hb_date_index],past_q_time)
    data_x2=data_x2.append(data_x2_temp)

data_x2.to_csv("./data/6_dataset_x2.csv")

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
    ##6月份有407个航班 里面可能有重复的时间 200个时间有航班，只计算航班时间对应人数就行了，简化计算一下
    # 对每一天生成一个list 长度为407的 代表这407个航班是否存在的序列
    hb_time_all = sorted(list(hb_schedule_all["SCHETIME"]))  ##拍好序的所有航班时间的数据

    get_all_hb_time_people(hb_time_all,"./data_ori/flight_filted6.csv")
    # test_peoples=possion_generator(400,500)
    # print(test_peoples)
    # plot(test_peoples,"500ren")
    # result = get_minute_one_count_all(filename="./data_ori/flight_filted6.csv")
    # result.to_csv("./data_ori/everyday.csv")
    # get_arrtavte_minutes_all()
# Tst()




