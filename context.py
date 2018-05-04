# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 12:18:54 2018

in charge of communication

@author: CountryOld
"""
from parameters import parameters
from TimeLine import TimeLine
from OrderBook import OrderBook
from Trader import AgentPool
from fvalue import fvalue
import logging
import numpy as np

price_limit = parameters['price_limit']
up_limit = 1 + price_limit
down_limit = 1 - price_limit
#有哪些类是外生的（随着时间流逝，自动变化的，模式在一开始就可以确定的）？
#timeline
#fvalue
#可以放到context里面

#内生的
#agentpool
#orderbook
#内生的类之间会互相交互，可以通过context进行通信

#调用顺序，先 定义外生的实体，再定义context 后定义需要利用context的实体如order_book 和 agentpool



class context():
    
    #Context应该也具备记录交易价格、交易量的功能吧
    
    def __init__(self, timeline, fv):
        self.TIMELINE = timeline
        self.FVALUE = fv
        self.latest_quote = None
        self.deal_info = {}
        self.latest_price = self.FVALUE.mu
        self.openprice = None
        self.upper_limit = None
        self.lower_limit = None
        self.__open_market()
        self.price_stream = self.FVALUE.forward_fv() #用于储存历史中间价 前101个价格提前给定！用fv模拟！
        self.bestbid = None
        self.bestask = None

    def receive_order(self, price, quantity, direction, agentid, duration, mkt = False):
        self.latest_quote = {'price':price, 'quantity':quantity, 
                             'direction':direction, 'agentid':agentid, 'duration':duration, 'mkt':mkt}        

    def receive_deal_info(self, agentid1, agentid2, price, quantity, agentid1_direction):
        agentid2_direction = 'ask' if agentid1_direction == 'bid' else 'bid'
        
        if agentid1 in self.deal_info.keys():
            self.deal_info[agentid1].append((price, quantity, agentid1_direction, self.TIMELINE.DAY))
        else:
            self.deal_info[agentid1] = [(price, quantity, agentid1_direction, self.TIMELINE.DAY)]
            
        if agentid2 in self.deal_info.keys():
            self.deal_info[agentid2].append((price, quantity, agentid2_direction, self.TIMELINE.DAY))
        else:
            self.deal_info[agentid2] = [(price, quantity, agentid2_direction, self.TIMELINE.DAY)]
    # 怎么实现，一成交就告知？一成交就刷新财富值？我觉得可以放一放，因为如果只提交一次指令且不撤销指令的话，
    # 在下一次要交易的时候更新财富值就行。
    def __open_market(self):
        self.openprice = round(self.latest_price, 2)
        self.upper_limit = round(self.openprice * up_limit, 2)
        self.lower_limit = round(self.openprice * down_limit, 2)
        logging.info('$$$$$$ the market opening price is {} $$$$$$'.format(self.openprice))

    def proceed(self):
        self.TIMELINE.roll()
        self.FVALUE.new_fv()
        if self.TIMELINE.lastDAY != self.TIMELINE.DAY:
            self.TIMELINE.newDay = True
        if self.TIMELINE.newDay and self.TIMELINE.inTrade:
            self.__open_market()
            self.TIMELINE.newDay = False
            
    def update_price(self, mid_price):
        #应该用索引就可以！
        #接下去修改orderbook 加入中间价的计算！
        self.latest_price = mid_price
        self.price_stream.append(mid_price)
       
if __name__ == '__main__':
    logging.basicConfig(filename='ham.log', filemode = 'w+', level=logging.INFO)
    timeline = TimeLine()
    fv = fvalue(timeline)
    CONTEXT = context(timeline, fv)
    orderbook = OrderBook(CONTEXT)
    agentpool = AgentPool(5000, CONTEXT)
    
#    count = 0
#    for i in range():
#        agentpool.select_an_agent()
#        if CONTEXT.latest_quote:
#            print(i)
#        orderbook.scan_quote()
    agentidlist = list(range(5000))
    for rd in range(1,5000):
        if CONTEXT.TIMELINE.inTrade:
            np.random.shuffle(agentidlist)
            for agentid in agentidlist:
                if agentpool.agent_activate(agentid):
#                    print(orderbook.bookset['bid'])
                    orderbook.delete_order(agentid) #不管最后的决策是什么，先撤单
#                    print(orderbook.bookset['bid'])
                    agentpool.agent_trade(agentid)
#                    print(orderbook.bookset['bid'])
                    orderbook.scan_quote()
            orderbook.calculate_price()
            orderbook.update_duration()
        if CONTEXT.TIMELINE.dayEnd:
            orderbook.clear()
        CONTEXT.proceed()
#        if CONTEXT.TIMELINE.inTrade and rd > 100:
#            fv.new_shock(shock_size=20)
#        agentpool.update_activated_agents()
    
    #清book ， 以及重返交易后自动清book
    
    #解决通信问题！
    
#    print(timeline.DAY,timeline.RD,timeline.TIME)
#     
#    for i in range(1000):
#        CONTEXT.proceed()
#        
#    agentpool.pool[19].send_order(10,55,'bid')
#    orderbook.scan_quote()
#    agentpool.pool[29].send_order(11,5,'bid')
#    orderbook.scan_quote()
#    agentpool.pool[39].send_order(9,15,'bid')
#    orderbook.scan_quote()
#    agentpool.pool[11].send_order(13,22,'ask')
#    orderbook.scan_quote()
#    agentpool.pool[21].send_order(15,32,'ask')
#    orderbook.scan_quote()
#    agentpool.pool[31].send_order(14,12,'ask')
#    orderbook.scan_quote()
#    print(orderbook.bookset)
#    print(CONTEXT.deal_info)
#    agentpool.pool[32].send_order(15,100,'bid',True)
#    orderbook.scan_quote()
#    for i in [19,29,39,11,21,31,32]:
#        agentpool.pool[i].scan_deal_info()
#    
#    for i in range(500):
#        timeline.roll(noprint=True)
    
#    print(timeline.DAY,timeline.RD,timeline.TIME)
#    print(orderbook.bookset)
#    print()
#    print(CONTEXT.deal_info)
#    print()
#    print(agentpool.pool[32].account.cash, agentpool.pool[32].account.holding)
#    agentpool.pool[32].scan_deal_info()
#    print(agentpool.pool[32].account.cash, agentpool.pool[32].account.holding)
#    print()
#    print(agentpool.pool[29].account.cash, agentpool.pool[29].account.holding)
#    agentpool.pool[29].scan_deal_info()
#    print(agentpool.pool[29].account.cash, agentpool.pool[29].account.holding)
#    print()
#    print(agentpool.pool[19].account.cash, agentpool.pool[19].account.holding)
#    agentpool.pool[19].scan_deal_info()
#    print(agentpool.pool[19].account.cash, agentpool.pool[19].account.holding)
#    print()
#    print(agentpool.pool[11].account.cash, agentpool.pool[11].account.holding)
#    agentpool.pool[11].scan_deal_info()
#    print(agentpool.pool[11].account.cash, agentpool.pool[11].account.holding)
#    print()
#    print(agentpool.pool[21].account.cash, agentpool.pool[21].account.holding)
#    agentpool.pool[21].scan_deal_info()
#    print(agentpool.pool[21].account.cash, agentpool.pool[21].account.holding)    

    

#方法！要么把涨跌停带进去！
#我的随机选交易量是错误的！因为交易量和价格的关系是非线性的！，所以肯定会导致价格非常靠近p_asterik
#INFO:root:after transformation,  lower_p 2.5399999999999996, upper_p 98.04
#INFO:root:before transformation, lower_q 2.0734279933478255e-05, upper_q 50
#INFO:root:after transformation,  lower_q 1, upper_q 50
#INFO:root:SELL 4900 at the price 35.4
#继续完善交易日志吧！

#还有这个问题，看着很成问题！
#35 * 0.01 == 0.35
#Out[9]: False

#直接把涨跌停板放进去！


## 暂时强行搞成paradox并略过了
#INFO:root:p* <= p_L < p_H <= pM
#INFO:root:lower_q 0.013136117329244539, upper_q 0.1510609568413745
#INFO:root:lower_q 1, upper_q 0

#今天要解决小数点后两位数的问题！
#为什么涨跌停限值不对！

#round(75.3449999999999994,2)
#Out[44]: 75.34
#round(75.3549999999999994,2)
#Out[45]: 75.36

#今天要改进的！
#参数
#交易决策的方式
#随机过程运动的方式

