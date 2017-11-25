#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 19:22:04 2017

@author: wq
"""

import os
from collections import defaultdict
import pandas as pd
import glob
from DBMgr import *
import cPickle as pickle
from sklearn.linear_model import Lasso
from scipy.stats import t
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from portfolioAssistant import *


#%% industry dict
basic = pd.read_pickle('./basic/all_industry.pickle')
industry_dict = {}
benchmark = 50
temp = basic.groupby('c_name').count().query('name>50').reset_index()
for ind in temp.c_name.unique():
    industry_dict[ind] = basic.loc[basic.c_name==ind,'code'].values
                 
#%%

# update new reports to database (check time)

# use full history to get factor

# update daily returns

# new portfolio