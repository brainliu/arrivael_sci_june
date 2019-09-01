#-*-coding:utf8-*-
#user:brian
#created_at:2019/9/1 15:25
# file: possion_distributn.py
#location: china chengdu 610000
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import scipy.stats as st

rv = st.poisson(4)
num_years = [4, 10, 7, 5, 4, 0, 0, 1]
x = range(8)
plt.bar(np.array(x)-.4, num_years, label='Observed instances')
plt.plot(x, sum(num_years)*rv.pmf(x), ls='dashed',
        lw=2, c='r', label='Poisson distribution\n$(\lambda=3.0)$')
plt.xlim([-1, 8])
plt.ylim([0, 11])
plt.xlabel('# of mass shootings in a year')
plt.ylabel('# of years')
plt.legend(loc='best')
plt.show()


##三段泊松分布，第一段和第三段为低密度区域，中间段为高密度区域
##调节各个分段之间的比例关系w1，w2，w3，以及每一段之间的参数lambda
hb_time=120
hb_people=180

rv_high=st.poisson(50)

y=range(hb_time)

plt.plot(y, hb_people*rv_high.pmf(y), ls='dashed',
        lw=2, c='r', label='Poisson distribution\n$(\lambda=2.0)$')
plt.show()

###到达时间间隔服从指数分布

from scipy.stats import binom
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(1, 1)

lambdaUse = 2
loc = 0
scale = 1.0 / lambdaUse


##LOC 为均值，SCAL为方差，size为抽取样本的size
# 平均值, 方差, 偏度, 峰度
mean, var, skew, kurt = st.expon.stats(loc, scale, moments='mvsk')
print(mean, var, skew, kurt)
# ppf:累积分布函数的反函数。q=0.01时，ppf就是p(X<x)=0.01时的x值。
x = np.linspace(st.expon.ppf(0.01, loc, scale), st.expon.ppf(0.99, loc, scale), 100)
ax.plot(x, st.expon.pdf(x, loc, scale), 'b-', label='expon')



###模拟泊松分布生成下一个旅客的到达时间（假设服从1）
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import scipy.stats as st

plt.title(u'指数分布概率密度函数')
plt.show()
for i in range(20):
        pp= np.random.poisson(20)
        print(pp)

