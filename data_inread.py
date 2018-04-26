# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:27:14 2018

@author: kretz01
"""

import pandas as pd
import numpy as np

#data = pd.read_table('N:\\MatMoDatPrivate\\kretz01\\linux\\SpyderProjects\\NeuralNet\\maunaloa_co2.txt', index_col=0, na_values=-99.99, skiprows=14, skipfooter=3, engine='python')
data = pd.read_table('N:\\MatMoDatPrivate\\kretz01\\linux\\SpyderProjects\\NeuralNet\\maunaloa_co2.txt', na_values=-99.99, skiprows=14, skipfooter=3, engine='python')

#np.loadtxt('N:\\MatMoDatPrivate\\kretz01\\linux\\SpyderProjects\\NeuralNet\\maunaloa_co2.txt', skiprows=16)
test = data['Jan']
#data.cumsum()
data.plot(y='Jan')

pd