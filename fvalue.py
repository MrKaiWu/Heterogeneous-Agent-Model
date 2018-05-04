# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 10:17:56 2018

这个是用来生成随机过程的
我的架构是先生成一串随机fv！而不是每循环一次生成一次！
我模型里fv并不是几何布朗运动！几何布朗运动的话基本面很可能一直增长！
就是围绕原价值100元波动
所以用Ornstein-Uhlenbeck Process
X(t+dt) = X(t) + k(mu - X(t))*dt + sigma*Y(t)
Y(t) ~ N(0,dt)
但是我这样规定，可能会导致市场流动性很差！因为没有大起大落啊。
或者说我这个本来就是一个基本面很稳定的资产
A股是T + 1的，必须得嵌入到模型里！
而且不同horizon的投资者 预期的fundamental value是一样的 很尬啊

也可以试试c的那两篇文章的定义。
@author: CountryOld
"""

from parameters import parameters
import numpy as np
import logging

init_fv = parameters['init_fv']
fv_adj = parameters['fv_adj']
fv_vol = parameters['fv_vol']
time_unit = parameters['time_unit']

# 4小时 240分钟 以内 布朗运动的volatility是fv_vol
# time_unit 5分钟
# 则dt为 5 / 240 

class fvalue():
    
#    fvalue应该和timeline产生关联
    
    def __init__(self, timeline, mu = init_fv, k = fv_adj, sigma = fv_vol):
        self.TIMELINE = timeline
        self.mu = mu
        self.k = k
        self.sigma = sigma
        self.dt = time_unit / 240
        self.dt_sqrt = np.sqrt(self.dt)
        self.FV = self.mu
        logging.info('FFFFFF the initial FV is {} FFFFFF'.format(self.FV))
#        self.time = self.TIMELINE.TIME
        self.timestream = [] #目前看来是没啥用的，以后再想办法解决！
        self.fvstream = []
        self.__snapshot()
        
    def __snapshot(self):
        self.timestream.append(self.TIMELINE.TIME)
        self.fvstream.append(self.FV)

    def new_fv(self):
        self.innovation = self.k * (self.mu - self.FV) * self.dt + \
            self.sigma * self.dt_sqrt * np.random.randn()
        self.FV += self.innovation
        logging.info('FFFFFF the FV for this ROUND is {} '.format(self.FV))
        self.__snapshot()
        
    def forward_fv(self, step = 100):
        FV = self.mu
        forward_FVlist = [FV]
        for i in range(step):
            forward_FVlist.append((FV - self.sigma * self.dt_sqrt * np.random.randn() - self.k * self.mu * self.dt) / (1 - self.k * self.dt))
        forward_FVlist.reverse()
        return forward_FVlist
        
    def new_shock(self,shock_size=None,shock_ratio=None):
        if shock_size and not shock_ratio:
            self.FV += shock_size
            self.mu += shock_size
        elif shock_ratio:
            self.FV *= 1 + shock_ratio
            self.mu *= 1 + shock_ratio
        self.__snapshot()


if __name__ == '__main__':
    from TimeLine import TimeLine
    
    fv = fvalue(TimeLine())
    plt.plot(fv.forward_fv())

#    plt.figure(figsize=(15,3))
#    fv = fvalue(100,0.1,0.5)
#    for i in range(500):
##        if i == 20:
##            fv.new_shock(5)
##        else:
#        fv.new_fv() 
#    plt.plot(fv.timestream,fv.fvstream)


#def brownian_path(N):
#    Δt_sqrt = math.sqrt(1 / N)
#    Z = np.random.randn(N)
#    Z[0] = 0
#    B = np.cumsum(Δt_sqrt * Z)
#    return B


#def gen_paths(S0, r, sigma, T, M, I):
#    dt = float(T) / M
#    paths = np.zeros((M + 1, I), np.float64)
#    paths[0] = S0
#    for t in range(1, M + 1):
#        rand = np.random.standard_normal(I)
#        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
#                                         sigma * np.sqrt(dt) * rand)
#    return paths
#
#S0 = 100
#r = 0.05
#sigma = 0.050
#T = 1
#N = 252
#i = 1000
#
#paths = gen_paths(S0, r, sigma, T, N, i)
#
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#
#mu=1
#n=50
#dt=0.1
#x0=100
#x=pd.DataFrame()
#
#for sigma in np.arange(0.8,2,0.2):
#    step=np.exp((mu-sigma**2/2)*dt)*np.exp(sigma*np.random.normal(0,np.sqrt(dt),(1,n)))
#    temp=pd.DataFrame(x0*step.cumprod())
#    x=pd.concat([x,temp],axis=1)
#
#x.columns=np.arange(0.8,2,0.2)
#plt.plot(x)
#plt.legend(x.columns)
#plt.xlabel('t')
#plt.ylabel('X')
#plt.show()