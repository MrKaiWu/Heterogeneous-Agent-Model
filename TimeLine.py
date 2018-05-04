# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:00:58 2018

@author: CountryOld
"""

from parameters import parameters
import logging

time_unit = parameters['time_unit']
if time_unit > 60:
    raise Exception('time_unit must be no larger than 60!')
elif time_unit <= 0:
    raise Exception('time_unit must be postive!')
elif 60 % time_unit != 0:
    raise Exception('60 % time_unit must be zero!')

class TimeLine():
    
    def __init__(self, breaks = True):
        self.DAY = 0
        self.lastDAY = 0
        self.newDay = False
        self.RD = 1  #突然醒悟，一个RD结束了没有，是由参与竞价的agent决定的，是随机的。
        self.inTrade = True #是不是处于开市状态
        self.dayEnd = False 

        if breaks == True:
            logging.info('###### simulate a trading day that is similar to the real setting ')
            self.total_units = 60 * 24 / time_unit
            logging.info('###### there are {} time units in one day '.format(self.total_units))
            self.start_time1 = int((60 * 9 + 30) / time_unit)
            self.end_time1 = int((60 * 11 + 30) / time_unit)
            self.start_time2 = int((60 * 13) / time_unit)
            self.end_time2 = int((60 * 15) / time_unit)
            self.TIME = self.start_time1
            logging.info('###### and the market starts at the {} time unit '.format(self.TIME))
            logging.info('###### now is DAY {}, ROUND {}, which corresponds to {}:{} in real time '.format(self.DAY, self.RD, self.TIME * time_unit // 60, self.TIME * time_unit % 60))
#            print(self.TIME,self.RD,self.TIME * time_unit // 60,':',self.TIME * time_unit % 60)
        
        else:
            logging.info('###### simulate a trading day that is continuous with no breaks except for the day transition ')
            self.total_units = 60 * 4 / time_unit
            logging.info('###### there are {} time units in one day '.format(self.total_units))
            self.TIME = 0
            logging.info('###### now is DAY {}, ROUND {} '.format(self.DAY, self.RD))
#            print(self.TIME,self.DAY,self.RD)
        
        self.breaks = breaks
        
    def roll(self, noprint = False):
        
        self.TIME += 1
        self.lastDAY = self.DAY
        if self.TIME == self.total_units:
            self.TIME = 0
            self.DAY += 1
            logging.info('###### switch into DAY {} '.format(self.DAY))
        
        if self.breaks:
            if self.start_time1 <= self.TIME < self.end_time1 or self.start_time2 <= self.TIME < self.end_time2:
                self.RD += 1
                self.inTrade =True
                logging.info('###### switch into ROUND {}, corresponding to real time {}:{} '.format(self.RD, self.TIME * time_unit // 60, self.TIME * time_unit % 60))
#                self.DayEnd = False
            else:
                self.inTrade = False
            if self.TIME == self.end_time2 - 1: 
                self.dayEnd = True
            else:
                self.dayEnd = False
            if not noprint:
#                print(self.TIME,self.RD,self.TIME * time_unit // 60,':',self.TIME * time_unit % 60)
                print(self.TIME, self.RD, self.TIME * time_unit // 60,':',self.TIME * time_unit % 60, self.inTrade)
        else:
            self.RD += 1
            logging.info('###### switch into ROUND {} '.format(self.RD))
            if self.TIME == self.total_units - 1:
                self.dayEnd = True
            else:
                self.dayEnd = False
            if not noprint:
                print(self.TIME, self.DAY, self.RD)
                

if __name__ == '__main__':
    tl = TimeLine(False)
    for i in range(300):
        tl.roll()
        
        
        
