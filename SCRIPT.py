#!/usr/bin/env python2
"""
Created on Fri Oct 27 13:28:49 2017

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
# 20 industry
#%%
#def toDate(x):
#    yr,sn = x.split('/')
#    if sn=='1':
#        return yr + '-' + '01'
#    elif sn=='2':
#        return yr + '-' + '04'
#    elif sn=='3':
#        return yr + '-' + '07'
#    elif sn=='4':
#        return yr + '-' + '10'        
#for ifd in fdlist:
#    fddata = pd.read_pickle(ifd)
#    fddata.date = fddata.date.apply(toDate)
#    fddata.to_pickle(ifd)


#%% factor selection:
dataMgr = DBMgr()
dataMgr.compSeasonRetLong(basic.code.values)

with open('./dataMgr.pickle','wb') as f:
    pickle.dump(dataMgr,f)
#%%
ind_ret = dataMgr.seasonalRet.mean(axis=1)
yrlist = [str(i) for i in range(2007,2018)]
seasonlist = ['01','04','07','10']
mapSeason = {'01':'04','04':'07','07':'10','10':'01'}
fdlist = glob.glob('./fundamental/*.pkl')

res = defaultdict(list)
wkDone = []
fdlist.remove('./fundamental/performance_all_quarter.pkl')
for ifd in fdlist:
    fddata = pd.read_pickle(ifd)
    #nm = os.path.basename(ifd).split('.pkl')[0]
    for yr in yrlist:
        for sn in seasonlist:
            pf = dataMgr.getTopPortfolio(fddata,yr+'-'+sn)
            try:
                for ifactor in pf.index:
                    if ifactor in wkDone:
                        continue
                    if len(pf.loc[ifactor,'best'])>0:
                        bt = dataMgr.getSeasonRet(pf.loc[ifactor,'best']).mean(axis=1)
                        bt.index = pd.to_datetime(bt.index)
                        if sn == '10':
                            iyr = str(int(yr)+1)
                        else:
                            iyr = yr
                        bt = bt[iyr+'-'+mapSeason[sn]].values
                        wt = dataMgr.getSeasonRet(pf.loc[ifactor,'worst']).mean(axis=1)
                        wt.index = pd.to_datetime(wt.index)
                        wt = wt[iyr+'-'+mapSeason[sn]].values
                        res[ifactor].extend(bt-wt)
                    else:
                        res[ifactor].append(0.)
            except KeyError:
                break
    ext = list(fddata.columns)
    ext.remove('date')
    ext.remove('code')
    ext.remove('name')
    wkDone.extend(ext)

#with open('./seasonalFactor.pickle','wb') as f:
#    pickle.dump(res,f)
#%%
with open('./dataMgr.pickle','rb') as f:
    dataMgr = pickle.load(f)
with open('./seasonalFactor.pickle','rb') as f:
    res = pickle.load(f)
#%%
output = pd.DataFrame(columns = ['cumRet','std','t-value','p-value'],index = res.keys())

for ifactor in res.keys():
    ret = np.nan_to_num(np.array(res[ifactor]))
    output.loc[ifactor,'cumRet'] = np.nansum(ret)
    output.loc[ifactor,'std'] = np.nanstd(ret)
    dif = (ret - ind_ret.values[1:])
    output.loc[ifactor,'t-value'] = np.nanmean(dif)/np.nanstd(dif)
    ttest = t(len(dif)-1)
    output.loc[ifactor,'p-value'] = 1 - ttest.cdf(output.loc[ifactor,'t-value'])
output.to_csv('./res1.csv')
# p-value < 0.5:
ls = output[output['p-value']<0.5].index.values
factor = {}
for i in ls:
    factor[i] = 1
#%% test beta
output2 = pd.DataFrame(columns = ['cumRet','std','beta','t-value','p-value'],index = res.keys())

for ifactor in res.keys():
    ret = np.nan_to_num(np.array(res[ifactor]))
    output2.loc[ifactor,'cumRet'] = np.nansum(ret)
    output2.loc[ifactor,'std'] = np.nanstd(ret)
    beta = np.dot(ret,ind_ret.values[1:])/np.dot(ret,ret)
    output2.loc[ifactor,'beta'] = beta
    sgm = np.nanstd(ind_ret.values[1:] - beta * ret)
    output2.loc[ifactor,'t-value'] = beta / sgm * np.dot(ret,ret)
    ttest = t(len(ret)-1)
    output2.loc[ifactor,'p-value'] = 1 - ttest.cdf(output2.loc[ifactor,'t-value'])
    
output2.to_csv('./res2.csv')
# p-value <0.3 or p-value >0.7
factor = {}
lsp = output2[output2['p-value']<0.3].index.values
for i in ls:
    factor[i] = 1
lsn = output2[output2['p-value']>0.7].index.values
for i in lsn:
    factor[i] = -1
                    
#%% lasso
x = np.zeros([37,len(res.keys())])
y = ind_ret.values[1:]
for i,ifactor in enumerate(res.keys()):
    ret = np.nan_to_num(np.array(res[ifactor]))
    x[:,i] = ret
model = Lasso(alpha=0.001)
model.fit(x,y)
#[res.keys()[i] for i,j in enumerate(model.coef_) if j!=0]
#factor 
#[u'currentratio',
# u'roe',
# u'net_profits',
# u'epsg',
# u'gross_profit_rate',
# u'inventory_days',
# u'icratio',
# u'nprg',
# u'adratio']
factor = {}
for i,j in enumerate(model.coef_):{u'adratio': -1,
 u'currentratio': 1,
 u'epsg': 1,
 u'gross_profit_rate': -1,
 u'icratio': 1,
 u'inventory_days': 1,
 u'net_profits': -1,
 u'nprg': 1,
 u'roe': -1}
    if j > 0:
        factor[res.keys()[i]] = 1
    elif j < 0:
        factor[res.keys()[i]] = -1
                   
    


#%% get the industry, factor average score
score_dict = {}
for ind in industry_dict.keys():
    stocklist = industry_dict[ind]
    score_dict[ind] = dataMgr.compScores(stocklist,factor,10)
#with open('./score_lasso9Factors.pickle','wb') as f:
#    pickle.dump(score_dict,f)
#with open('./score_perf3factor.pickle','wb') as f:
#    pickle.dump(score_dict,f)
#with open('./score_beta15factor.pickle','wb') as f:
#    pickle.dump(score_dict,f)
#%%
with open('./score_lasso9Factors.pickle','rb') as f:
    score_dict = pickle.load(f)
#%% select stocks!
stock_dict = defaultdict(list)
dates = [u'2007-01', u'2007-04', u'2007-07', u'2007-10', u'2008-01', u'2008-04',
       u'2008-07', u'2008-10', u'2009-01', u'2009-04', u'2009-07', u'2009-10',
       u'2010-01', u'2010-04', u'2010-07', u'2010-10', u'2011-01', u'2011-04',
       u'2011-07', u'2011-10', u'2012-01', u'2012-04', u'2012-07', u'2012-10',
       u'2013-01', u'2013-04', u'2013-07', u'2013-10', u'2014-01', u'2014-04',
       u'2014-07', u'2014-10', u'2015-01', u'2015-04', u'2015-07', u'2015-10',
       u'2016-01']

for i,idate in enumerate(dates):
    for ind in industry_dict.keys():
        if idate in score_dict[ind].index:
            n = len(score_dict[ind].loc[idate])
            stocklist = score_dict[ind].loc[idate].sort_values()[-int(n*0.05):].index.values# top 10%
            dt = pd.to_datetime(idate)
            stock_dict[ind].append(dataMgr.getSeasonRet(stocklist).mean(axis=1)[i+1])
        else:
            stock_dict[ind].append(np.nan)

factorPort = np.zeros(len(dates))
for ind in industry_dict.keys():
    factorPort = factorPort + np.nan_to_num(np.array(stock_dict[ind]))
factorPort = factorPort / len(industry_dict.keys())
plot['fact'] = np.concatenate([[0],factorPort])
plot.cumsum().plot()


#%% daily return
daily = dataMgr.getDailyRetLong(basic.code.values)
#daily[daily>=0.1] = np.nan
#daily[daily<=-0.1] = np.nan
ind_ret_daily = daily['2007-04':].mean(axis=1).cumsum()
ind_ret_daily = pd.DataFrame({'meanPortfolio':ind_ret_daily})
#%%
with open('./score_lasso9Factors.pickle','rb') as f:
    score_dict = pickle.load(f)
fact_lasso_ret_daily = pd.DataFrame(columns = ['fact_lasso'])
for st,holdst,holded in zip(dates[:-1],dates[1:],dates[2:]+['2016-04']):
    stocklist = []
    for ind in industry_dict.keys():
        if st in score_dict[ind].index:
            stocklist.extend(score_dict[ind].loc[st].sort_values()[-3:].index.values)
    fact_lasso_ret_daily = pd.concat([fact_lasso_ret_daily,
                                      pd.DataFrame(daily[holdst:holded].loc[:,stocklist].mean(axis=1)).rename(columns = {0:'fact_lasso'})],axis=0)
fact_lasso_ret_daily = fact_lasso_ret_daily[~fact_lasso_ret_daily.index.duplicated()]
fact_lasso_ret_daily = fact_lasso_ret_daily.fillna(0).cumsum()
#%%
with open('./score_perf5factor.pickle','rb') as f:
    score_dict = pickle.load(f)
fact_perf_ret_daily = pd.DataFrame(columns = ['fact_perf'])
for st,holdst,holded in zip(dates[:-1],dates[1:],dates[2:]+['2016-04']):
    stocklist = []
    for ind in industry_dict.keys():
        if st in score_dict[ind].index:
            stocklist.extend(score_dict[ind].loc[st].sort_values()[-3:].index.values)
    fact_perf_ret_daily = pd.concat([fact_perf_ret_daily,
                                      pd.DataFrame(daily[holdst:holded].loc[:,stocklist].mean(axis=1)).rename(columns = {0:'fact_perf'})],axis=0)
fact_perf_ret_daily = fact_perf_ret_daily[~fact_perf_ret_daily.index.duplicated()]
fact_perf_ret_daily = fact_perf_ret_daily.fillna(0).cumsum()
#%%
with open('./score_beta15factor.pickle','rb') as f:
    score_dict = pickle.load(f)
fact_beta_ret_daily = pd.DataFrame(columns = ['fact_beta'])
for st,holdst,holded in zip(dates[:-1],dates[1:],dates[2:]+['2016-04']):
    stocklist = []
    for ind in industry_dict.keys():
        if st in score_dict[ind].index:
            stocklist.extend(score_dict[ind].loc[st].sort_values()[-3:].index.values)
    fact_beta_ret_daily = pd.concat([fact_beta_ret_daily,
                                      pd.DataFrame(daily[holdst:holded].loc[:,stocklist].mean(axis=1)).rename(columns = {0:'fact_beta'})],axis=0)
fact_beta_ret_daily = fact_beta_ret_daily[~fact_beta_ret_daily.index.duplicated()]    
fact_beta_ret_daily = fact_beta_ret_daily.fillna(0).cumsum()
#%% hedged:
#with open('./score_perf5factor.pickle','rb') as f:
#    score_dict = pickle.load(f)
#comp500 = pd.read_csv('./index/zz500_cons.csv',header=None,names=['comp']).comp.values
#fact_perf_count_daily = pd.DataFrame(columns = ['nstock'])
#for st,holdst,holded in zip(dates[:-1],dates[1:],dates[2:]+['2016-04']):
#    stocklist = []
#    for ind in industry_dict.keys():
#        if st in score_dict[ind].index:
#            stocklist.extend(score_dict[ind].loc[st].sort_values()[-5:].index.values)
#    temp2 = pd.DataFrame(daily[holdst:holded].loc[:,stocklist].mean(axis=1)).rename(columns = {0:'nstock'})
#    cnt = 0
#    for isk in stocklist:
#        if 'CH'+isk in comp500:
#            cnt+=1
#    temp2['nstock'] = cnt
#    fact_perf_count_daily = pd.concat([fact_perf_count_daily,
#                                      temp2],axis=0)
    
#fact_perf_count_daily = fact_perf_count_daily /500.
future = pd.read_csv('./index/zz500.csv',index_col=0)
future.index = pd.to_datetime(future.index)
future = future[['PO']]
temp = pd.merge(fact_perf_ret_daily,future,left_index=True, right_index=True)
temp['PO'] = temp['PO'].pct_change()
temp.loc[temp[temp['PO']>0.1].index,'PO'] = 0.0
temp.loc[temp[temp['PO']<-0.1].index,'PO']= 0.0
temp['PO'] = temp['PO']
#temp = pd.merge(temp,fact_perf_count_daily,left_index=True, right_index=True)
temp['hedgeRatio'] = 0.4
temp['hedgeRatio'] = (temp['PO'] * temp.loc[temp[temp['fact_perf'].diff()<0].index,'hedgeRatio']).fillna(0).cumsum()
fact_perf_ret_daily_hedge = temp['fact_perf'] - temp['hedgeRatio']
fact_perf_ret_daily_hedge = pd.DataFrame({'hedgedPortfolio':fact_perf_ret_daily_hedge})

#%%
fig,ax = plt.subplots()
ind_ret_daily.plot(ax=ax)
#fact_lasso_ret_daily.plot(ax=ax)
#fact_perf_ret_daily.rename(columns = {'fact_perf':'Lasso'}).plot(ax=ax)
#fact_beta_ret_daily.plot(ax=ax)
#fact_perf_ret_daily_hedge.plot(ax=ax)
fact_perf_monthly_ret_daily.rename(columns = {'fact_perf_mk':'Restricted Markowitz'}).plot(ax=ax)
ax.grid()
ax.set_ylabel('cumProfit')
#%% monthly rolling
monthly = daily.fillna(0).resample('m').sum()
monthly = monthly.rolling(4).mean()
cov_t = daily.cov()
pca = PCA(n_components=5,copy=False)
subspace = pca.fit_transform(daily.fillna(0).values.transpose()).transpose()
df_sub = pd.DataFrame(index = monthly.columns,columns = ['comp'+str(i) for i in range(5)])
for isub,iname in zip(subspace,monthly.columns):
    df_sub.loc[iname,:] = isub.transpose()
#%%
fact_perf_monthly_ret_daily = pd.DataFrame(columns = ['fact_perf_mk'])
yrlist = [str(i) for i in range(2007,2017)]
mthlist = ['0' + str(i) if len(str(i))==1 else str(i) for i in range(1,13)]
months = []
for yr in yrlist:
    for mth in mthlist:
        months.append(yr+'-' + mth)
months = months[3:-8]
count = 0
turnover = pd.DataFrame(index=daily.columns,columns=['previous','current'])
turnover['previous'] = 0
turnover['current'] = 0
sumTurn = 0.
ret = None
for i,holdst,holded in zip(range(len(months[:-1])),months[:-1],months[1:]):
    if i%3==0:
        stocklist = []
        st = dates[count]
        for ind in industry_dict.keys():
            if st in score_dict[ind].index:
                stocklist.extend(score_dict[ind].loc[st].sort_values()[-5:].index.values)
        count += 1
    stocklist = [i for i in stocklist if i in monthly.columns]
    mthret = monthly[stocklist][holdst].values
    mthret[np.isnan(mthret)] = 0
    wt = portfolioAssistant.shrinkage(mthret.reshape(-1),cov_t.loc[stocklist,stocklist].values,df_sub.loc[stocklist].values)
    print sum(wt)
    wt = wt/sum(wt)
    turnover.loc[stocklist,'current'] = wt
    sumTurn += (turnover.current - turnover.previous).abs().sum(axis=0)
    turnover.previous = turnover.current
    turnover.current = 0.
    ret = (daily[holdst:holded].loc[:,stocklist]*wt).sum(axis=1)
    fact_perf_monthly_ret_daily = pd.concat([fact_perf_monthly_ret_daily,
                                      pd.DataFrame(ret).rename(columns = {0:'fact_perf_mk'})],axis=0)
    
fact_perf_monthly_ret_daily = fact_perf_monthly_ret_daily.fillna(0).cumsum()
