# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 09:31:18 2018
Define the characteristics of a typical agent in the HAM experiment.
@author: CountryOld
"""

from parameters import parameters
import numpy as np
from scipy.optimize import newton, brentq
from collections import namedtuple
import logging

GF = parameters['GF']
GC = parameters['GC']
NT = parameters['NT']
epsilon = parameters['noise']
tau = parameters['tau']
baseline_gamma = parameters['baseline_gamma']
init_precision = parameters['init_precision']
init_cash_max = parameters['init_cash_max']
init_holding_max = parameters['init_holding_max']
init_fv =  parameters['init_fv']
min_unit = parameters['min_unit']

class AgentPool():
    
    def __init__(self, numbers, context):
        
        self.pool = {}
        self.CONTEXT = context
        self.activated_agents = {}
        self.agentid_max = numbers
        self.agentarray = range(self.agentid_max)
        for i in range(numbers):
            self.pool[i] = agent(i, context)
        logging.info('{} agents ready'.format(numbers))

    def agent_activate(self, agentid):
        #从不位于activated_agents的id中按顺序挑选出一个！然后再按概率看该agent有没有被激活
        current_agent = self.pool[agentid]
        activated = current_agent.activate()
        if activated:
            logging.info('D{}R{}'.format(self.CONTEXT.TIMELINE.DAY, self.CONTEXT.TIMELINE.RD) + '*' * 90)
            logging.info('agent {} activated.'.format(agentid))
        return activated
    
    def agent_trade(self, agentid):
        current_agent = self.pool[agentid]
#        logging.info('D{}R{}'.format(self.CONTEXT.TIMELINE.DAY, self.CONTEXT.TIMELINE.RD) + '*' * 90)
#        logging.info('agent {} activated.'.format(agentid))
        current_agent.trading_decision()        
        
# 这个是持有期内不再次入场的版本，然而，既然以持有期倒数为概率入场，那平均而言入场间隔就是持有期！。        
#    def select_an_agent(self, agentid):
#        #从不位于activated_agents的id中按顺序挑选出一个！然后再按概率看该agent有没有被激活
#        if agentid in self.activated_agents.keys():
#            pass
#        else:
#            current_agent = self.pool[agentid]
#            activated = current_agent.activate()
#            if activated:
#                logging.info('D{}R{}'.format(self.CONTEXT.TIMELINE.DAY, self.CONTEXT.TIMELINE.RD) + '*' * 90)
#                logging.info('agent {} activated.'.format(agentid))
#                make_trade = current_agent.trading_decision()
#                if make_trade:
#                    self.activated_agents[agentid] = current_agent.horizon
    
    def update_activated_agents(self):
        aa_copy = self.activated_agents.copy()
        for key in self.activated_agents.keys():
            if self.activated_agents[key] >= 2:
                aa_copy[key] -= 1
            else:
                del aa_copy[key]
        self.activated_agents = aa_copy

#    def randomly_pick_and_trade(self):
#        np.random.shuffle(self.agentarray)
#        for agentid in self.agentarray:
#            self.__select_an_agent(agentid)

class agent():
    
    def __init__(self, agentid, context):
        
        self.agentid = agentid
        self.CONTEXT = context
        
        self.gf = np.random.exponential(GF)
        self.gc = np.random.exponential(GC)
        self.nt = np.random.exponential(NT)
        
        ratio = tau * (1 + self.gf) / (1 + self.gc)
        self.horizon = round(ratio)
        if self.horizon == 0:
            self.horizon = 1
            
#        self.gamma = 10 * np.tanh(baseline_gamma * ratio)
        self.gamma = baseline_gamma * ratio
        
        self.precision = init_precision
        
        holding = round(init_holding_max * np.random.uniform()) * min_unit  #init_holding_max是手数！
        cash = round(init_cash_max * np.random.uniform())
        self.account = account(cash, holding, context)
        
    def __fundamental_return(self):
        exp_FV = self.CONTEXT.FVALUE.FV + np.random.normal(scale = 1 / self.precision)
        logging.info('expected fundamental value is {}'.format(exp_FV))
        if self.CONTEXT.latest_price is None:
            return None
        logging.info('investment horizon is {}'.format(self.horizon))
        fv_ret = 1 / self.horizon * np.log(exp_FV / self.CONTEXT.latest_price)
        logging.info('expected fundamental return is {}'.format(fv_ret))
        return fv_ret
        
#    def __trend_return(self):
#        price_history = self.CONTEXT.price_stream[- self.horizon - 1:]
#        if len(price_history) <= 2:
#            return None
#        else:
#            price_history.reverse()
#            return np.mean(np.log(a/b) for a, b in zip(price_history, price_history[1:]))

    def __trend_volatility_and_return(self):
        price_history = self.CONTEXT.price_stream[- self.horizon - 1:]
        if len(price_history) <= 2:
            self.trend_volatility = epsilon**2  #如果，就用epsion的平方吧 ！留意一下，到时候看看平均的volatility到底多少 
            logging.info('history variance of return is {}'.format(self.trend_volatility))
            return None #返回的是return, 过去的return是中间值，所以没必要作为单独的属性，但trend_volatility会在不同的地方用到。
        else:
            price_history.reverse()
            array = [np.log(a/b) for a, b in zip(price_history, price_history[1:])]
            self.trend_volatility = np.var(array)
            if self.trend_volatility == 0:
                self.trend_volatility = epsilon**2 #这个也是值得争议的！
            logging.info('history variance of return is {}'.format(self.trend_volatility))
            trend_ret = np.mean(array)
            logging.info('history average return is {}'.format(trend_ret))
            return trend_ret

#这里也是要跟时间联系在一起的！然后要先在context里定义price！
    def __expect_price(self):
        f_ret = self.__fundamental_return()
        trend_ret = self.__trend_volatility_and_return()
        noise_ret = np.random.normal(scale = epsilon)
        logging.info('noise return is {}'.format(noise_ret))
        if f_ret is not None and trend_ret is not None:
            exp_ret =  1 / (self.gc + self.gf + self.nt) * (self.gf * f_ret + self.gc * trend_ret + self.nt * noise_ret)
        elif f_ret is None and trend_ret is None:
            exp_ret = noise_ret
        elif f_ret is None and trend_ret is not None:
            exp_ret = 1 / (self.gc + self.nt) * (self.gc * trend_ret + self.nt * noise_ret)
        else:
            exp_ret = 1 / (self.gf + self.nt) * (self.gf * f_ret  + self.nt * noise_ret)
        logging.info('expect future return is {}'.format(exp_ret))
        self.exp_price = self.CONTEXT.latest_price * np.exp(exp_ret * self.horizon)
        logging.info('expect future price is {}'.format(self.exp_price))
    
    def __desirable_holding(self, price):
        return np.log(self.exp_price / price) / (self.gamma * self.trend_volatility * price)
        
    def __gen_equation(self, CONSTANT, ADDITIONAL_TERM = 0):
        def func(p):
            return np.log(self.exp_price / p) - self.gamma * self.trend_volatility * p * CONSTANT - ADDITIONAL_TERM
        return func
    
    def __solve_for_p(self):
        self.__expect_price() #这一步已经把volatililaty也更新了
#        self.__trend_volatility_and_return()
        self.func1 = self.__gen_equation(self.account.unsellable)
#        self.pM = brentq(self.func1, 1e-100, self.exp_price + 1, maxiter = 1000)
        try:
            self.pM = newton(self.func1, self.exp_price, maxiter = 1000)
        except RuntimeError:
            self.pM = None
        
        self.func2 = self.__gen_equation(self.account.holding)
        try:
            self.p_asterik = newton(self.func2, self.exp_price / 2, maxiter = 1000)
        except RuntimeError:
            self.p_asterik = None
#        self.p_asterik = brentq(self.func2, 1e-100, self.exp_price + 1, maxiter = 1000)
        
        self.func3 = self.__gen_equation(self.account.holding, self.gamma * self.trend_volatility * self.account.cash)
#        self.pm = brentq(self.func3, 1e-100, self.exp_price + 1, maxiter = 1000)
        try:        
            self.pm = newton(self.func3, 1, maxiter = 1000)
        except RuntimeError:
            self.pm = 0.01
        #这里加几行修正的代码！
        logging.info('originally pm is {}, p_asterik is {}, pM is {}.'.format(self.pm, self.p_asterik, self.pM))
        if self.pM is None or self.p_asterik is None:
            self.pm = None
        elif abs(self.pm - self.p_asterik) < 1e-6 and abs(self.p_asterik - self.pM) < 1e-6:
            self.pm = self.p_asterik = self.pM = None
        logging.info('afterwards pm is {}, p_asterik is {}, pM is {}.'.format(self.pm, self.p_asterik, self.pM))

    def __round_to_int(self, number, tol=1e-4, nosmallerthan = True):
        if abs(number - round(number)) < tol:
            return int(round(number))
        else:
            if nosmallerthan:
                return int(number) + 1
            else:
                return int(number)         
    
    def __quantity_and_price(self, lower_p, upper_p):
#        print(lower_p, upper_p)
        lower_q = self.__desirable_holding(upper_p) / min_unit
        upper_q = self.__desirable_holding(lower_p) / min_unit
        logging.info('lower_q {}, upper_q {}'.format(lower_q, upper_q))                                                        
        lower_q = self.__round_to_int(self.__desirable_holding(upper_p) / min_unit)
        upper_q = self.__round_to_int(self.__desirable_holding(lower_p) / min_unit, nosmallerthan=False)
        logging.info('lower_q {}, upper_q {}'.format(lower_q, upper_q))              
        if lower_q > upper_q:
            logging.info('paradox appers that lower_q > upper_q')
            q = self.account.holding
            return None, q
        else:
            q = np.random.choice(range(lower_q, upper_q + 1)) * min_unit #在100的整数倍里面，随机选择一个持有量
            func = self.__gen_equation(q) 
            p = newton(func, (lower_p + upper_p) / 2, maxiter=1000) #解出相对应的价格
            p = round(p, 2) #转化为合法的价格
            if p < lower_p:
                p = lower_p
            elif p > upper_p:
                p = upper_p
            return p, q  
    
    def __book_strategy(self, p, quantity):
        if quantity > 0:
            logging.info('BUY {} at the price {}'.format(quantity, p))
            if not self.CONTEXT.bestask:
                logging.info('no opposite order in the current askbook, thus send a limit order')
                self.send_order(p, quantity, 'bid')
            elif p >= self.CONTEXT.bestask[0]:
                logging.info('accept the best quote {} in the current askbook, thus send a market order'.format(self.CONTEXT.bestask))
                self.send_order(p, quantity, 'bid', True)
            else:
                logging.info('expect a better price, thus send a limit order')
                self.send_order(p, quantity, 'bid')
        else:
            quantity = -quantity
            logging.info('SELL {} at the price {}'.format(quantity, p))
            if not self.CONTEXT.bestbid:
                logging.info('no opposite order in the current bidbook, thus send a limit order')
                self.send_order(p, quantity, 'ask')
            elif p <= self.CONTEXT.bestbid[0]:
                logging.info('accept the best quote {} in the current bidbook, thus send a market order'.format(self.CONTEXT.bestbid))
                self.send_order(p, quantity, 'ask', True)
            else:
                logging.info('expect a better price, thus send a limit order')
                self.send_order(p, quantity, 'ask')        

    def trading_decision(self):
        self.scan_deal_info()        
        self.__solve_for_p()
        if self.pm is None:
            logging.info('peculiar roots for p are encountered')
            choice = (None, self.account.holding)
        elif self.account.holding - self.account.unsellable == 0: #就是当前没有持仓！ pM = p*
            logging.info('no holding at the moment')
            direction = 1
            if self.pm < self.CONTEXT.lower_limit <= self.pM:
                if self.pM < self.CONTEXT.upper_limit :
                    logging.info('pm < p_L <= p* = pM < p_H ')
                    choice = self.__quantity_and_price(self.CONTEXT.lower_limit, self.pM) 
                else:
                    logging.info('pm < p_L < p_H <= p* = pM')
                    choice = self.__quantity_and_price(self.CONTEXT.lower_limit, self.CONTEXT.upper_limit)
            elif self.CONTEXT.lower_limit <= self.pm < self.pM < self.CONTEXT.upper_limit:
                logging.info('p_L < pm < p* = pM < p_H')
                choice = self.__quantity_and_price(self.pm, self.pM)
            elif self.CONTEXT.lower_limit <= self.pm < self.CONTEXT.upper_limit <= self.pM:
                logging.info('p_L < pm < p_H <= p* = pM')
                choice = self.__quantity_and_price(self.pm, self.CONTEXT.upper_limit)
            elif self.CONTEXT.upper_limit <= self.pm: #全买
                logging.info('p_H <= pm < p* = pM')
                holding = self.__round_to_int(self.account.cash / self.CONTEXT.upper_limit / min_unit, nosmallerthan=False) * min_unit
                choice = (self.CONTEXT.upper_limit, holding) 
            else: #
                logging.info('pM < p_L')
                choice = (None, self.account.holding)    
        else:
            if self.pM <= self.CONTEXT.lower_limit:
                logging.info('the zero-holding price is less than lower limit price, so SELL all sellables')
#            direction = 0
                choice = (self.CONTEXT.lower_limit, self.account.unsellable)
            elif self.p_asterik <= self.CONTEXT.lower_limit < self.pM:
#            direction = 0
                if self.pM < self.CONTEXT.upper_limit:
                    logging.info('p* <= p_L < pM < p_H ')
                    choice = self.__quantity_and_price(self.CONTEXT.lower_limit, self.pM)
                else:
                    logging.info('p* <= p_L < p_H <= pM')
                    choice = self.__quantity_and_price(self.CONTEXT.lower_limit, self.CONTEXT.upper_limit)
            elif self.pm <= self.CONTEXT.lower_limit and self.CONTEXT.lower_limit < self.p_asterik:
                if self.p_asterik < self.CONTEXT.upper_limit:
                    if self.pM < self.CONTEXT.upper_limit:
                        logging.info('pm <= p_L < p* < pM < p_H')
                        direction = 1 if np.random.binomial(1, (self.p_asterik - self.CONTEXT.lower_limit)/(self.pM - self.CONTEXT.lower_limit)) else 0
                        if direction == 1:
                            choice = self.__quantity_and_price(self.CONTEXT.lower_limit, self.p_asterik)
                        else:
                            choice = self.__quantity_and_price(self.p_asterik, self.pM)
                    else:
                        logging.info('pm <= p_L < p* < p_H <= pM ')
                        direction = 1 if np.random.binomial(1, (self.p_asterik - self.CONTEXT.lower_limit)/(self.CONTEXT.upper_limit - self.CONTEXT.lower_limit)) else 0
                        if direction == 1:
                            choice = self.__quantity_and_price(self.CONTEXT.lower_limit, self.p_asterik)
                        else:
                            choice = self.__quantity_and_price(self.p_asterik, self.CONTEXT.upper_limit)
                else:
                    logging.info('pm <= p_L < p_H <= p* ')
#                direction = 1
                    choice = self.__quantity_and_price(self.CONTEXT.lower_limit, self.CONTEXT.upper_limit)
            elif self.CONTEXT.lower_limit < self.pm:
                if self.pM <= self.CONTEXT.upper_limit:
                    logging.info('p_L < pm < p* < pM <= p_H ')
                    direction = 1 if np.random.binomial(1, (self.p_asterik - self.pm)/(self.pM - self.pm)) else 0
                    if direction == 1:
                        choice = self.__quantity_and_price(self.pm, self.p_asterik)
                    else:
                        choice = self.__quantity_and_price(self.p_asterik, self.pM)
                elif self.p_asterik <= self.CONTEXT.upper_limit < self.pM:
                    logging.info('p_L < pm < p* <= p_H < pM ')
                    direction = 1 if np.random.binomial(1, (self.p_asterik - self.pm)/(self.CONTEXT.upper_limit - self.pm)) else 0
                    if direction == 1:
                        choice = self.__quantity_and_price(self.pm, self.p_asterik)
                    else:
                        choice = self.__quantity_and_price(self.p_asterik, self.CONTEXT.upper_limit)
                elif self.pm <= self.CONTEXT.upper_limit < self.p_asterik:
                    logging.info('p_L < pm <= p_H < p* ')
#                direction = 1
                    choice = self.__quantity_and_price(self.pm, self.CONTEXT.upper_limit)
                else:
                    logging.info('p_L {} < p_H {} < pm {} '.format(self.CONTEXT.lower_limit, self.CONTEXT.upper_limit, self.pm))
#                direction = 1
                    holding = self.__round_to_int(self.account.cash / self.CONTEXT.upper_limit / min_unit, nosmallerthan=False) * min_unit
                    choice = (self.CONTEXT.upper_limit, holding)
                
        p, q = choice
        quantity = q - self.account.holding
        if quantity == 0:
            logging.info('agent decides to trade nothing')
            return None
        else:
            self.__book_strategy(p, quantity)
            return True

    def activate(self):
        return np.random.binomial(1, 1 / self.horizon)
        
    def respond_to_news(self, cur_fvalue, last_fvalue):
        self.prob_respond = np.tanh(abs(cur_fvalue - last_fvalue) / last_fvalue * np.log(self.horizon)) #这个线的性态还不是很好！要改的！应该加一个指数。但是会不会产生新的问题，就是，投资期很长的基本面投资者，反而一有点小波动就来围观了，不就相当于投资期很短的人了么。 
        self.respond = np.random.binomial(1, self.prob_respond)
                
    def update_precision(self):
        pass
    
    def send_order(self,price,quantity,direction,mkt=False):
        
        self.CONTEXT.receive_order(price=price,quantity=quantity,\
                         direction=direction,agentid=self.agentid,duration=self.horizon,mkt=mkt) #OrderBook关注一下。

    def scan_deal_info(self):
        if self.agentid in self.CONTEXT.deal_info.keys():
            logging.info('agent {}\'s latest deal record is {}'.format(self.agentid, self.CONTEXT.deal_info[self.agentid]))
            for deal in self.CONTEXT.deal_info[self.agentid]:
                if deal[2] == 'ask':
                    delta_holding = -deal[1]
                else:
                    delta_holding = deal[1]
                self.account.update_acc(delta_holding, deal[0], deal[3])
            del self.CONTEXT.deal_info[self.agentid]

acc_history = namedtuple('acc_history',['rd','cash','holding','wealth'])

class account():
    
    def __init__(self, cash, holding, context):
        self.cash = cash
        self.holding = holding
#        self.agent = agent
        self.wealth = self.cash + init_fv * self.holding
        self.buyrecord = []
        self.unsellable = 0
        self.CONTEXT = context
        self.history = []
        self.__acc_record()
    
    def __acc_record(self):
        self.history.append(acc_history(self.CONTEXT.TIMELINE.RD, self.cash, self.holding, self.wealth)) #rd这个变量比较尴尬
    
    def update_acc(self, delta_holding, trans_price, dealday):
        logging.info('before the deal the agent has {} cash and {} holding'.format(self.cash, self.holding))
        self.cash -= trans_price * delta_holding #这里的transaction_price要注意, 价格形成机制
        self.holding += delta_holding
        if delta_holding > 0:
            self.buyrecord.append((dealday, delta_holding))
        self.wealth = self.cash + self.holding * trans_price #这里的tp要注意
        logging.info('after the deal the agent has {} cash and {} holding'.format(self.cash, self.holding))
        self.__update_unsellable()
        self.__acc_record()

    def __update_unsellable(self): #只针对买交易
        self.unsellable = sum(x[1] for x in self.buyrecord if x[0] == self.CONTEXT.TIMELINE.DAY)
        logging.info('and {} of the holding is unsellable'.format(self.unsellable))
        if self.unsellable == 0:
            self.buyrecord = []


if __name__ == '__main__':
    import matplotlib.pyplot as plt    
#    hl = []
#    for i in range(5000):
#        a = agent()
#        a.respond_to_news(1,0.95)
#        hl.append(a.prob_respond)
#    hl = np.array(hl)
#    hl = np.array([agent().horizon for i in range(5000)])
#    hl = np.array([agent().account.wealth for i in range(5000)])
#    hl = np.array([agent().gamma for i in range(5000)])
#    print('min:', np.min(hl))
#    print('max:', np.max(hl))
#    print('mean:', np.mean(hl))
#    print('median:', np.median(hl))
#    print('horizons longer than one trading day:', np.mean(hl>240/parameters['time_unit']))
#    print('horizons longer than two trading day:', np.mean(hl>240*2/parameters['time_unit']))
#    print('horizons longer than three trading day:', np.mean(hl>240*3/parameters['time_unit']))
#    plt.hist(hl, bins=100)
    
    pass    