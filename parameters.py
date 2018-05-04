# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 09:35:22 2018

@author: CountryOld
"""

parameters = {
        'GC': 2,
        'GF': 2,
        'NT': 1,
        'noise':1e-3,
        'tau': 60,
        'time_unit': 2, #分钟
        'init_precision': 1,
        'init_fv': 100,
        'fv_adj': 0.25,
        'fv_vol': 5,
        'init_holding_max':1000, #手数 
        'baseline_gamma':0.1,
        'price_limit': 0.1,
        'min_unit': 10
        }

parameters['init_cash_max'] = parameters['init_holding_max'] * parameters['init_fv'] * 100
          
#parameters = {
#        'GC': 1,
#        'GF': 1,
#        'NT': 1,
#        'noise':1e-3,
#        'tau': 60,
#        'time_unit': 5, #分钟
#        'init_precision': 1,
#        'init_fv': 100,
#        'fv_adj': 0.1,
#        'fv_vol': 5,
#        'init_holding_max':50, #手数 
#        'baseline_gamma':0.005,
#        }
#
#parameters['init_cash_max'] = parameters['init_holding_max'] * parameters['init_fv'] * 100