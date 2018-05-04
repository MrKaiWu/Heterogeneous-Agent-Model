# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 23:28:51 2018
争取在吃饭之前完成order book!
@author: CountryOld
"""

import numpy as np
#rd
#rank

import logging

class OrderBook():
# 集成了交易功能
    def __init__(self, context):
#        self.bidbook = []
#        self.askbook = []
        self.CONTEXT = context
        self.bookset = {'bid':[],'ask':[]}
        self.last_rd = self.CONTEXT.TIMELINE.RD
        self.rank = 0 #limit order排序
        self.hasTrade = False #这一轮有木有成交！
        self.mid_price = self.CONTEXT.FVALUE.mu
        self.latest_trade_price = None
        self.hasOrder = set()
        # 瞬时的价格也是需要记录的！因为这是agent在make_order的时候看到的价格！
        # 这里 lattest_trade_price 代表本轮最新成交价。
#        self.RDstream = []
#        self.mid_price_stream = []

    def update_duration(self):
        num_orders1 = len(self.bookset['bid']) + len(self.bookset['ask'])
        for book in ['bid', 'ask']:
            tmp = []
            for order in self.bookset[book]:
                order = order.copy()
                order[-1] -= 1
                if order[-1] > 0:
                    tmp.append(order)
            self.bookset[book] = tmp
        num_orders2 = len(self.bookset['bid']) + len(self.bookset['ask'])
        if num_orders1 != num_orders2:
            logging.info('D{}R{}$$$$$$ after this ROUND {} orders are due and dropped'.format(self.CONTEXT.TIMELINE.DAY, self.CONTEXT.TIMELINE.RD, num_orders1 - num_orders2))
        self.__update_agentids()
        self.__sort_and_update_best_quote('all', rank = False)
        
    def scan_quote(self):

        if self.CONTEXT.TIMELINE.RD != self.last_rd:
            self.last_rd = self.CONTEXT.TIMELINE.RD 
            self.rank = 0
            self.hasTrade = False
            self.latest_trade_price = None

        if self.CONTEXT.latest_quote:
#            print(self.CONTEXT.latest_quote)
            self.__update_book(**self.CONTEXT.latest_quote)            
            self.__sort_and_update_best_quote('all')
            self.__update_agentids()
            self.CONTEXT.latest_quote = None
        else:
            pass
    
    def delete_order(self, agentid):
        if not agentid in self.hasOrder:
            pass
        for book in ['bid', 'ask']:
            tmp = []
            for order in self.bookset[book]:
                if order[-2] != agentid:
                    tmp.append(order)
            self.bookset[book] = tmp
        logging.info('$$$$$$ delete previous orders of the agent')
        self.__sort_and_update_best_quote('all', rank='False')
        
    def __update_book(self,price,quantity,direction,agentid,duration,mkt=False):
        
        if not mkt:
            if direction == 'bid':
                self.bookset['bid'].append([price, quantity, self.CONTEXT.TIMELINE.RD, self.rank, agentid, duration])
#                self.__sort_and_update_best_quote('all')  #惊天大bug 同下
            elif direction == 'ask':
                self.bookset['ask'].append([price, quantity, self.CONTEXT.TIMELINE.RD, self.rank, agentid, duration])
#                self.__sort_and_update_best_quote('all') #惊天大bug 以前是bid 这样的话如果是因为market ask order执行完毕仍有剩余，bid book已清空，但CONTEXT.bestbid不会自动调整！卧槽
            self.rank += 1
            
        else:
            self.__mkt_deal(price, quantity, direction, agentid, duration)    
    
    def __mkt_deal(self,price,quantity,direction,agentid,duration): #实现2009年那篇文章的撮合方式，market order 一直用limit order去撮合直到达到设定的价格以后，转化为limit order
#      尽最大限度消化一个market order，剩下的转化为limit order, 然后把成交的结果传送给相应的agent

        if direction == 'bid':
            bookdirection = 'ask'
        else:
            bookdirection = 'bid'
#        print(AGENTPOOL.pool[agentid].account.holding)
        while quantity > 0:
#            print('q',quantity)
#            print(agentid, quantity, bookdirection)
            best_quote = self.bookset[bookdirection][0]
            best_quote_price = best_quote[0]
            best_quote_quant = best_quote[1]
            best_quote_agentid = best_quote[4]
#            print(best_quote)
            if ((best_quote_price >= price and  bookdirection == 'bid') \
                    or (best_quote_price <= price and bookdirection == 'ask')):
                if best_quote_quant > quantity:    
                    tmp = best_quote
                    tmp[1] = best_quote_quant - quantity
                    self.bookset[bookdirection][0] = tmp
                    self.CONTEXT.receive_deal_info(agentid, best_quote_agentid, best_quote_price, quantity, direction)
                    quantity = 0
                else:
                    quantity = quantity - best_quote_quant
                    del self.bookset[bookdirection][0]
                    self.CONTEXT.receive_deal_info(agentid, best_quote_agentid, best_quote_price, best_quote_quant, direction)
                
                self.latest_trade_price = best_quote_price
                self.CONTEXT.latest_price = best_quote_price
                
                if len(self.bookset[bookdirection]) == 0:
                    logging.info('D{}R{}$$$$$$ all the {} book is cleared! '.format(self.CONTEXT.TIMELINE.DAY, self.CONTEXT.TIMELINE.RD, bookdirection))
                    break
            else:
                break
        
        logging.info('D{}R{}$$$$$$ the latest trade price is {} '.format(self.CONTEXT.TIMELINE.DAY, self.CONTEXT.TIMELINE.RD, self.latest_trade_price))
        if quantity > 0:
            logging.info('D{}R{}$$$$$$ but there is still {} of the market order remaining '.format(self.CONTEXT.TIMELINE.DAY, self.CONTEXT.TIMELINE.RD, quantity))
            self.__update_book(best_quote_price, quantity, direction, agentid, duration)  #草，居然现在才发现， 以前是price + bookdirection！

        elif bookdirection == 'ask':
            logging.info('D{}R{}$$$$$$ the market order is completely absorbed '.format(self.CONTEXT.TIMELINE.DAY, self.CONTEXT.TIMELINE.RD))
#            if len(self.bookset['ask']) == 0:
#                self.CONTEXT.bestask = None
#            else:
#                self.CONTEXT.bestask = self.bookset['ask'][0]
        else:
            logging.info('D{}R{}$$$$$$ the market order is completely absorbed '.format(self.CONTEXT.TIMELINE.DAY, self.CONTEXT.TIMELINE.RD))
#            if len(self.bookset['bid']) == 0:
#                self.CONTEXT.bestbid = None
#            else:
#                self.CONTEXT.bestbid = self.bookset['bid'][0]
            
        self.hasTrade = True
        
    def __update_agentids(self):
        self.hasOrder = set()
        for order in self.bookset['bid']:
            self.hasOrder.add(order[-2])
        for order in self.bookset['ask']:
            self.hasOrder.add(order[-2])
            
    def __sort_and_update_best_quote(self,booktype='bid', rank=True):
        if booktype == 'bid':
            if rank:
                self.bookset['bid'] = sorted(self.bookset['bid'], key = lambda x: (-x[0],x[2],x[3]))
            if self.bookset['bid']:
                self.CONTEXT.bestbid = self.bookset['bid'][0]
            else:
                self.CONTEXT.bestbid = None
        elif booktype == 'ask':
            if rank:
                self.bookset['ask'] = sorted(self.bookset['ask'], key = lambda x: (x[0],x[2],x[3]))
            if self.bookset['ask']:
                self.CONTEXT.bestask = self.bookset['ask'][0]
            else: #惊天大bug！以前是没有else的。。。。
                self.CONTEXT.bestask = None                
        elif booktype == 'all':
            self.__sort_and_update_best_quote('bid', rank)
            self.__sort_and_update_best_quote('ask', rank)
        
    def calculate_price(self):
        #每轮RD结束后计算中间价
        #如果发出的是限价指令，那一定是因为当前quote不满足自己的要求。
        #如果发出市价指令，当前的quote一定部分或全部满足自己的要求。
        #所以，不存在两个限价指令撮合成交的情况！
        # 逻辑： 
        # 如果在这一轮有成交，且轮末 bidbook askbook都有深度，则计算新的中间价
        # 如果在这一轮有成交，且轮末 bidbook askbook至少有一方是空的，则以上一次成交的价格作为中间价
        # 如果在这一轮中，没有任何撮合成交，则mid_price等于上一轮的值，依此类推
        if self.hasTrade:
            if self.bookset['ask'] and self.bookset['bid']:
                self.mid_price = (self.bookset['ask'][0][0] + self.bookset['bid'][0][0]) / 2
                logging.info('D{}R{}$$$$$$ there is trade at this round, and both sides of order book is not empty '.format(self.CONTEXT.TIMELINE.DAY, self.CONTEXT.TIMELINE.RD))
                logging.info('D{}R{}$$$$$$ so price for this round is {}, which is calulated as the mid-point price '.format(self.CONTEXT.TIMELINE.DAY, self.CONTEXT.TIMELINE.RD, self.mid_price))
            else:
                self.mid_price = self.latest_trade_price
                logging.info('D{}R{}$$$$$$ there is trade at this round, but at least one side of the book is empty '.format(self.CONTEXT.TIMELINE.DAY, self.CONTEXT.TIMELINE.RD))
                logging.info('D{}R{}$$$$$$ so price for this round is {}, which is equal to the latest trade price '.format(self.CONTEXT.TIMELINE.DAY, self.CONTEXT.TIMELINE.RD, self.mid_price))

            self.CONTEXT.update_price(self.mid_price)
        else:
            #不变啊
            logging.info('D{}R{}$$$$$$ there is no trade at this round, so price for this round remain unchanged at {} '.format(self.CONTEXT.TIMELINE.DAY, self.CONTEXT.TIMELINE.RD, self.mid_price))
            self.CONTEXT.update_price(self.mid_price)
    
    def clear(self):
        self.bookset['bid'] = []
        self.bookset['ask'] = []
        self.CONTEXT.bestbid = None
        self.CONTEXT.bestask = None
        logging.info('D{}R{}$$$$$$ all the orders previous stored are cleared!'.format(self.CONTEXT.TIMELINE.DAY, self.CONTEXT.TIMELINE.RD))


if __name__ == '__main__':
    pass

#    from pprint import pprint

#测试通信
#    rd = 0
#    AGENTPOOL = AgentPool(200)
#    ORDERBOOK = OrderBook()    
#    AGENTPOOL.pool[19].limit_order(10,55,'bid',ORDERBOOK)
#    AGENTPOOL.pool[29].limit_order(11,5,'bid',ORDERBOOK)
#    AGENTPOOL.pool[39].limit_order(9,15,'bid',ORDERBOOK)
#    
#    AGENTPOOL.pool[11].limit_order(13,22,'ask',ORDERBOOK)
#    AGENTPOOL.pool[21].limit_order(15,32,'ask',ORDERBOOK)
#    AGENTPOOL.pool[31].limit_order(14,12,'ask',ORDERBOOK)
#
#    pprint(ORDERBOOK.bookset['ask'])
#    print()
#    pprint(ORDERBOOK.bookset['bid'])
#    
#    print()
#    print('before:')
#    print('agent',AGENTPOOL.pool[32].agentid, AGENTPOOL.pool[32].account.holding, AGENTPOOL.pool[32].account.cash)
#    print('agent',AGENTPOOL.pool[29].agentid, AGENTPOOL.pool[29].account.holding, AGENTPOOL.pool[29].account.cash)
#    print('agent',AGENTPOOL.pool[19].agentid, AGENTPOOL.pool[19].account.holding, AGENTPOOL.pool[19].account.cash)
#    print('agent',AGENTPOOL.pool[39].agentid, AGENTPOOL.pool[39].account.holding, AGENTPOOL.pool[39].account.cash)
#    
#    AGENTPOOL.pool[32].market_order(9,100,'ask',ORDERBOOK)
#    print('after:')
#    print('agent',AGENTPOOL.pool[32].agentid, AGENTPOOL.pool[32].account.holding, AGENTPOOL.pool[32].account.cash)
#    print('agent',AGENTPOOL.pool[29].agentid, AGENTPOOL.pool[29].account.holding, AGENTPOOL.pool[29].account.cash)
#    print('agent',AGENTPOOL.pool[19].agentid, AGENTPOOL.pool[19].account.holding, AGENTPOOL.pool[19].account.cash)
#    print('agent',AGENTPOOL.pool[39].agentid, AGENTPOOL.pool[39].account.holding, AGENTPOOL.pool[39].account.cash)
#
#    pprint(ORDERBOOK.bookset['ask'])
#    print()
#    pprint(ORDERBOOK.bookset['bid'])
#
#    print()
#    print('before:')
#    print('agent',AGENTPOOL.pool[101].agentid, AGENTPOOL.pool[101].account.holding, AGENTPOOL.pool[101].account.cash)
#    print('agent',AGENTPOOL.pool[11].agentid, AGENTPOOL.pool[11].account.holding, AGENTPOOL.pool[11].account.cash)
#    print('agent',AGENTPOOL.pool[31].agentid, AGENTPOOL.pool[31].account.holding, AGENTPOOL.pool[31].account.cash)
#    print('agent',AGENTPOOL.pool[21].agentid, AGENTPOOL.pool[21].account.holding, AGENTPOOL.pool[21].account.cash)
#    
#    AGENTPOOL.pool[101].market_order(15.001,230,'bid',ORDERBOOK)
#    print('after:')
#    print('agent',AGENTPOOL.pool[101].agentid, AGENTPOOL.pool[101].account.holding, AGENTPOOL.pool[101].account.cash)
#    print('agent',AGENTPOOL.pool[11].agentid, AGENTPOOL.pool[11].account.holding, AGENTPOOL.pool[11].account.cash)
#    print('agent',AGENTPOOL.pool[31].agentid, AGENTPOOL.pool[31].account.holding, AGENTPOOL.pool[31].account.cash)
#    print('agent',AGENTPOOL.pool[21].agentid, AGENTPOOL.pool[21].account.holding, AGENTPOOL.pool[21].account.cash)
#
#    pprint(ORDERBOOK.bookset['ask'])
#    print()
#    pprint(ORDERBOOK.bookset['bid'])


#测试orderbook
#    ob = OrderBook()    
#    rd = 0
#    ob.update_book(int(10 * np.random.uniform()),10,'bid','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'ask','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'bid','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'bid','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'bid','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'bid','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'ask','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'ask','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'ask','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'ask','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'ask','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'ask','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'bid','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'bid','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'bid','unknown')
#    rd = 1
#    ob.update_book(int(10 * np.random.uniform()),10,'ask','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'bid','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'ask','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'ask','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'ask','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'bid','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'bid','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'bid','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'bid','unknown')
#    rd = 2
#    ob.update_book(int(10 * np.random.uniform()),10,'bid','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'ask','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'ask','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'ask','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'ask','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'ask','unknown')
#    ob.update_book(int(10 * np.random.uniform()),10,'bid','unknown')
#    
#    
#    pprint.pprint(ob.bookset['bid'])
#    print()
#    pprint.pprint(ob.bookset['ask'])
#    
#    
#    ob.update_book(price=10.5,quantity=161,direction='bid',agentid='wukai',mkt=True)
#    print()
#    pprint.pprint(ob.bookset['ask'])
    
    
    
    
