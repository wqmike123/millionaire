#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:12:07 2017

@author: wq
"""
import numpy as np
#from numpy.linalg import eig
import pandas as pd
#%%

class portfolioAssistant(object):
    
     
    @staticmethod
    def compute_position(x,cov,target = 0.2):
        # compute weights
        inv_cov = np.linalg.inv(cov)
        a = np.dot(np.dot(x,inv_cov),x)
        return a/target*np.dot(inv_cov,x)
    
    @staticmethod
    def shrinkage(x,cov,subspace,target = 0.2,stepSize = 0.01,maxStep = 8000, acc = 1e-5,upper=0.1):
        n = len(x)
        inv_cov = np.linalg.inv(cov)
        a = np.dot(np.dot(x,inv_cov),x)
        lbd = a/target
        w = subspace[:,0]
        for i in range(maxStep):
            w = w - stepSize*(lbd*np.dot(cov,w) - x)
            # project back
            temp = 0.
            for k in range(subspace.shape[1]):
                temp = temp + np.dot(w,subspace[:,k])/np.dot(subspace[:,k],subspace[:,k]) * subspace[:,k]
            w = temp
            w[w<0] = 0.
            w[w>0.1] = 0.1
        return w
    
    
    @staticmethod
    def aumAnalysis(aum):
        AUM_daily = pd.DataFrame(index = aum.index)
        AUM_daily['cum_profit'] = aum
        AUM_daily = AUM_daily.ffill()
        ret = AUM_daily['cum_profit'].diff().fillna(0)
        mret, vol = ret.mean(), ret.std()
        
        # MDD
        DD = AUM_daily['cum_profit'] - AUM_daily['cum_profit'].expanding().max()
        MDD = DD.min()
        MddDate = DD.idxmin().date()
        # calcuate maximum drawdown duration
        DD = DD.reset_index()
        tmp = ((DD['cum_profit']<0)*1.).diff()[1:]
        tmp1, tmp2 = tmp[tmp==1].index, tmp[tmp==-1].index
        if len(tmp1)*len(tmp2) == 0:
            Duration = np.array([len(tmp)])
        else:
            if tmp1[0] > tmp2[0]:
                tmp1 = np.array([DD.index[0]]+tmp1.tolist())
            if len(tmp1) != len(tmp2):
                tmp2 = np.array(tmp2.tolist()+[DD.index[-1]])
            Duration = tmp2 - tmp1
        NDay = len(DD)
        Nyear = float(NDay)/255
        ret_mdd = round(AUM_daily['cum_profit'][-1]/abs(MDD),2)
        sharpe = round((mret - 0.)/vol * np.sqrt(365),2)  
        res = {'CAGR':round(aum[-1]*100./Nyear,2),'Duration':Duration.max(),'Ret_MDD_Ratio':ret_mdd,'Sharpe':sharpe,'Mdd_Date':MddDate}
        return res