import pandas as pd
import numpy as np
import os
import glob
#%%
class DBMgr(object):
    seasonalRet = None
    
    def __init__(self,stock_dir = './price_2/',factor_dir = './fundamental/'):
        self.stock_dir_long = '/home/wq/Documents/files/quant/astock/data/'
        self.stock_dir = stock_dir
        self.factor_dir = factor_dir
        
    def compSeasonRet(self,stocklist):
        res = pd.DataFrame()
        for i in stocklist:
            stockfile = self.stock_dir+'price_history_'+str(i)+'.pkl'
            if not os.path.exists(stockfile):
                continue
            temp = pd.read_pickle(stockfile).reset_index().set_index('date')[['close']]
            temp.index = pd.to_datetime(temp.index)
            temp = temp.resample('M')
            temp2 = temp.last()
            temp2 = temp2.rename(columns = {'close':i})
            temp2[i] = np.log(temp2[i].values / temp.first().close.values)
            res = pd.concat([res,temp2],axis=1)
        res = res.resample('3M').sum()
        self.seasonalRet = res
    def compSeasonRetLong(self,stocklist):
        res = pd.DataFrame()
        for i in stocklist:
            stockfile = self.stock_dir_long+'CH'+str(i)+'.csv'
            if not os.path.exists(stockfile):
                continue
            temp = pd.read_csv(stockfile,index_col=0)[['P']]
            temp.index = pd.to_datetime(temp.index)
            temp = temp['2007':]
            temp = temp.resample('M')
            temp2 = temp.last()
            temp2 = temp2.rename(columns = {'P':i})
            logret = np.log(temp2[i].values / temp.first().P.values)
            #logret[logret>0.1] = 0.1
            #logret[logret<-0.1] = -0.1
            temp2[i] = logret
            res = pd.concat([res,temp2],axis=1)
        res = res.resample('3M').sum()
        self.seasonalRet = res
        
    def getSeasonRet(self,stocklist):
        if isinstance(self.seasonalRet,pd.DataFrame):
            commList = [i for i in stocklist if i in self.seasonalRet.columns]
            return self.seasonalRet.loc[:,commList]
        else:
            raise Exception
    def getDailyRet(self,stocklist):
        res = pd.DataFrame()
        for i in stocklist:
            stockfile = self.stock_dir+'price_history_'+str(i)+'.pkl'
            if not os.path.exists(stockfile):
                continue
            temp = pd.read_pickle(stockfile).reset_index().set_index('date')[['close']]        
            temp.index = pd.to_datetime(temp.index)
            temp = temp[['close']].rename(columns ={'close':i})
            temp[i] = temp[i].pct_change()
            res = pd.concat([res,temp],axis=1)
        return res
    
    def getDailyRetLong(self,stocklist):
        res = pd.DataFrame()
        for i in stocklist:
            stockfile = self.stock_dir_long+'CH'+str(i)+'.csv'
            if not os.path.exists(stockfile):
                continue
            temp = pd.read_csv(stockfile,index_col=0)[['P']]
            temp.index = pd.to_datetime(temp.index)
            temp = temp['2007':]
            temp = temp[['P']].rename(columns ={'P':i})
            temp[i] = temp[i].pct_change()
            res = pd.concat([res,temp],axis=1)
        return res
    
    def getTopPortfolio(self,df,date,ascending = False):
        date = pd.to_datetime(date)
        df.date = pd.to_datetime(df.date)
        temp = df.query('date==@date').drop(['date','name'],axis=1).set_index('code')
        res = pd.DataFrame(columns = ['best','worst'],index=temp.columns)
        for ifactor in temp.columns:
            temp2 = temp[[ifactor]]
            temp2 = temp2.dropna()
            n = len(temp2)
            if n > 10:
                temp2 = temp2.sort_values(by=ifactor,ascending=ascending)
                res.loc[ifactor,'best'] = list(temp2.iloc[:int(n*0.1)].index)
                res.loc[ifactor,'worst'] = list(temp2.iloc[int(n*0.9):].index)
            else:
                res.loc[ifactor,'best'] = []
                res.loc[ifactor,'worst'] = []            
        return res
    
    def compScores(self,stocklist,factor,group):
        score = pd.DataFrame(columns = stocklist,index = [])
        res = pd.DataFrame(columns = stocklist,index = [])
        filelist = glob.glob(self.factor_dir+'/*')
        filelist = [i for i in filelist if 'performance' not in i]
        for ifile in filelist:
            temp = pd.read_pickle(ifile)
            selectCol = [i for i in temp.columns if i in factor.keys()]
            if len(selectCol)>0:
                for ifactor in selectCol:
                    for istock in stocklist:
                        factdata = temp[['code','date',ifactor]].query('code == @istock')
                        if len(factdata)>0:
                            self._addFactor(factdata,ifactor,res)                        
                    if len(score)>0:
                        score += self._toRank(res,factor[ifactor],group)
                    else:
                        score = self._toRank(res,factor[ifactor],group)
        return score

        
    
    def _addFactor(self,factdata,factor,res):
        code = factdata['code'].values[0]
        factdata = factdata.set_index('date').rename(columns = {factor:code})
        factdata = factdata[~factdata.index.duplicated()]
        res[code] = factdata[code]
        
    def _toRank(self,factdata,ascending,group):
        factdata = factdata.apply(self._toRankRow,args=(ascending,group),axis=1)
        return factdata
    
    def _toRankRow(self,x,ascending,group = 10):
        n = len(x)
        if n < group:
            group = n
        else:
            gpSize = int(n / group)
        #temp = np.sort(ascending*x)
        #xcopy = x.copy()
        #left = temp[0]
        #for i in range(1,group+1):
        #    x[np.logical_and(xcopy>=left,xcopy<temp[i*gpSize])] = i
        #    left = i*gpSize
        #x[np.logical_not(xcopy<=max(temp))] = 0
        nan = np.logical_not(x<=max(x))
        x = group - (np.argsort(-ascending*x) / gpSize).astype('int')
        x[nan] = 0
        return x
        