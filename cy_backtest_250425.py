# ChenYue
# 2024.9.3 at Shanghai University of Finance & Economics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlwings as xw

from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import datetime
import rqdatac
import os
cwd = os.getcwd()

class Backtest:
    def __init__(self):
        pass

    @staticmethod
    def get_annualized_return(data:list|pd.DataFrame, raw=False, leverage=1, fee=0, trading_days=252):
        '''
        ### 计算年化收益率   
        **data**: index为日期，第一列为资产价格走势的DataFrame   
        **raw**: 是否返回非年化的数据（用于计算Ytm或者Ytg），默认为False
        **leverage**: 杆杆率，默认为1
        **fee**: 交易费等费率，默认为0%，单位为%
        '''
        if isinstance(data, list):
            ls = data
            years = len(ls) / trading_days
        
        elif isinstance(data, pd.DataFrame):
            ls = data.iloc[:, 0].to_list()
            days_ls = data.index.astype(str).to_list()
            years_ls = sorted(list(set([date[:4] for date in days_ls])))
            # 第一年
            year_first =  sum([1 if date[:4]==years_ls[0] else 0 for date in days_ls])/trading_days
            # 最后一年
            year_last = sum([1 if date[:4]==years_ls[-1] else 0 for date in days_ls])/trading_days
            years = len(years_ls) - 2 + year_first + year_last
        
        raw_return = (ls[-1] - ls[0]) / ls[0]
        if raw:
            return leverage*(raw_return - fee/100)
        if raw_return < 0:
            raw_return = -raw_return
            return leverage*(-(raw_return+1) ** (1/years) + 1 - fee/100)
        return leverage*((raw_return+1) ** (1/years) - 1 - fee/100)

    @staticmethod
    def get_max_drawdown(ls:list|pd.DataFrame, leverage=1):
        '''
        ### 计算最大回撤
        **ls**: index为日期，第一列为资产价格走势的DataFrame或值为资产价格走势的list   
        **leverage**: 杆杆率，默认为1
        '''
        if isinstance(ls, pd.DataFrame):
            ls = ls.iloc[:, 0]

        max_value = ls[0]
        max_drawdown = 0
        current_drawdown = 0

        for value in ls:
            if value > max_value:
                max_value = value
                current_drawdown = 0
            else:
                current_drawdown = (max_value - value) / max_value
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown

        return leverage*max_drawdown

    @staticmethod
    def get_annualized_volatility(ls:list|pd.DataFrame, leverage=1):
        '''
        ### 计算年化波动率
        **ls**: index为日期，第一列为资产价格走势的DataFrame或值为资产价格走势的list 
        **leverage**: 杆杆率，默认为1  
        '''
        if isinstance(ls, pd.DataFrame):
            ls = ls.iloc[:, 0]

        # 计算日收益率
        daily_returns = [np.divide(ls[i] - ls[i - 1], ls[i - 1]) for i in range(1, len(ls))]
        daily_returns = [float(return_value) for return_value in daily_returns]
        # 计算日标准差
        daily_std = np.std(daily_returns, ddof=1)
        # 年化标准差
        annualized_std = daily_std * np.sqrt(252)
        
        return leverage*annualized_std

    @staticmethod
    def get_sharpe_rate(ls:list|pd.DataFrame, leverage=1, fee=0):
        '''
        ### 计算夏普率
        **ls**: index为日期，第一列为资产价格走势的DataFrame或值为资产价格走势的list  
        **fee**: 交易费等费率，默认为0%，单位为% 
        '''
        return Backtest.get_annualized_return(ls, leverage=leverage, fee=fee)/Backtest.get_annualized_volatility(ls, leverage=leverage)

    @staticmethod
    def get_weights(data:pd.DataFrame, total=False):
        '''
        ### 根据投资组合中各个资产的净值，计算各资产所占比例
        **data**: index为日期，值为各资产价格走势的DataFrame   
        **sum**: 返回的DataFrame中是否包含总净值，默认为False
        '''
        weight_df = pd.DataFrame(index=data.index, columns=data.columns)
        for index in data.index:
            total_value = data.loc[index, :].sum()
            for col in data.columns:
                weight_df.loc[index, col] = data.loc[index, col] / total_value

        if total:
            weight_df['total net value'] = data.sum(axis=1)
        
        return weight_df

    @staticmethod
    def plot_cumulative_returns(data:pd.DataFrame, 
                                title = 'Cumulative Returns of Portfolio',
                                xlabel = 'Date',
                                ylabel = 'Cumulative Returns',
                                figsize = (18, 6),
                                twindata = None,    
                                benchmark: pd.DataFrame|str='000300.SH',
                                colorlist = None):
        '''
        ### 画出策略和标的的价格走势
        **data**: index为日期，值为各类资产持仓价格走势的DataFrame   
        **title**: 图的标题，默认为 Cumulative Returns of Portfolio    
        **xlabel**: x轴标签， 默认为 Date   
        **ylabel**: y轴标签， 默认为 Cumulative Returns    
        **figsize**: 画图大小，默认为(18, 6)    
        **twindata**: 是否加入副y轴，默认为None，否则提供一个和data格式相同的数据      
        **benchmark**: 策略标的，默认为沪深300，或者给一个与data格式相同的DataFrame     
        **colorlist**: 颜色，默认为None，否则按照从data到twindata的顺序依次上色 
        '''
        _, ax = plt.subplots(figsize=figsize)

        if isinstance(benchmark, str) and benchmark!='None':
            bm = rqdatac.get_price(rqdatac.id_convert(benchmark),
                                    start_date=data.index[0],
                                    end_date=data.index[-1])['close'].to_list()
            data[benchmark] = bm
        for col in data.columns:
            if isinstance(colorlist, list):
                ax.plot(data[col], label=col, color=colorlist[0])
                colorlist = colorlist[1:]
            else:
                ax.plot(data[col], label=col)

        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel, color='blue')
        ax.legend(loc='upper left')
        ax.grid(True)

        if isinstance(twindata, pd.DataFrame):
            ax2 = ax.twinx()
            for col in twindata.columns:
                if isinstance(colorlist, list):
                    ax2.plot(twindata[col], label=col, color=colorlist[0])
                    colorlist = colorlist[1:]
                else:
                    ax2.plot(twindata[col], label=col)
            ax2.legend(loc='center left')
        elif twindata=='benchmark':
            benchmark = '000300.SH'
            bm = rqdatac.get_price(rqdatac.id_convert(benchmark),
                                    start_date=data.index[0],
                                    end_date=data.index[-1])['close'].xs(rqdatac.id_convert(benchmark)).to_frame()
            ax2 = ax.twinx()
            print(bm.columns)
            for col in bm.columns:
                if isinstance(colorlist, list):
                    ax2.plot(bm[col], label=col, color=colorlist[0])
                    colorlist = colorlist[1:]
                else:
                    ax2.plot(bm[col], label=col)
            ax2.legend(loc='center left')

    @staticmethod
    def plot_weights(data:pd.DataFrame, 
                     title='Weights for Each Assests',
                     figsize=(14, 6)):
        '''
        ### 画出各类资产的权重变化
        **data**: index为日期，值为各类资产权重的DataFrame   
        **title**: 图的标题，默认为 Weights for Each Assests      
        '''
        weights_ls = [data.iloc[:,i].to_list() for i in range(len(data.columns))]
        _, ax = plt.subplots(figsize=figsize)
        ax.stackplot(range(len(data)), *weights_ls, labels=data.columns.to_list())
        ax.legend(loc='upper left')
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Weights', color='blue')
        ax.grid(True)

    @staticmethod
    def get_corr_matrix(prices:pd.DataFrame|pd.core.series.Series):
        '''
        ### 计算资产间价格走势的相关性
        **prices**: 资产的价格走势
        '''
        returns = prices.pct_change().iloc[1:,:]
        cov_matrix = returns.cov()
        std_devs = np.sqrt(np.diag(cov_matrix))                             # 计算标准差
        corr_matrix = cov_matrix / (std_devs[:, None] * std_devs[None, :])  # 计算相关性矩阵

        return corr_matrix

    @staticmethod
    def get_weighted_assets(data:pd.DataFrame, weight:list=[1], frequency:str='1m'):
        '''
        ### 将几类资产按一定比例组合，得到加权资产
        **data**: columns为资产，值为价格走势的DataFrame      
        **weight**: 各个资产的权重      
        **frequency**： 每隔多久重新调仓至指定权重
        '''
        data = data.copy()
        weight = weight * len(data.columns) if weight==[1] else weight
        if len(weight) != len(data.columns):
            raise ValueError('num of weight does not match num of assets')
        weight = [weight[i]/sum(weight) for i in range(len(weight))]

        if frequency=='1m':
            last_adjust_month = data.index[0].month - 1
            for idx in data.index:
                if idx.month != last_adjust_month:
                    total_value = sum(data.loc[idx,:])
                    # 调仓
                    for j in range(len(data.columns)):
                        data.loc[idx:, data.columns[j]] = norm(data.loc[idx:, data.columns[j]].to_frame(), weight[j]*total_value).iloc[:,0]
                    last_adjust_month = idx.month
        return data

    @staticmethod
    def get_rolling_period_returns(data:pd.DataFrame, 
                                    period='12m'):
        '''
        ### 得到滚动时间区间的收益率列表        
        **data**: index为日期，值为资产的价格走势的DataFrame      
        **period**: 滚动时间区间的长度，一年以内设置为'{}m'，超过一年设置为'{}y'      
        '''
        month_delta = 0
        year_delta = 0
        if period[-1]=='m':
            month_delta = int(period[:-1])
        if period[-1]=='y':
            year_delta = int(period[:-1])

        date_ls = []
        rolling_period_returns_ls = []
        for date in data.index:
            year = date.year
            month = date.month
            day = date.day

            till_year = year + year_delta
            till_month = month + month_delta
            if till_month > 12:
                till_month -= 12
                till_year += 1
            till_day = day-1
            if till_day == 0:
                till_day = 31

            while True:
                try:
                    till_date = datetime.datetime(till_year, 
                                                till_month,
                                                till_day)
                    break
                except:
                    till_day -= 1
            
            if till_date >= data.index[-1]:
                break
            
            this_period_data = data.loc[date:till_date, data.columns[0]].to_frame()
            date_ls.append(date)
            rolling_period_returns_ls.append(Backtest.get_annualized_return(this_period_data, raw=True))
        return pd.DataFrame(index=date_ls, data=rolling_period_returns_ls, columns=[period+' rolling returns'])

    @staticmethod
    def get_rolling_period_corr(data:pd.DataFrame, 
                                period='12m'):
        '''
        ### 得到两类资产滚动时间区间的相关性        
        **data**: index为日期，columns有两列，值为两列资产的价格走势的DataFrame      
        **period**: 滚动时间区间的长度，一年以内设置为'{}m'，超过一年设置为'{}y'      
        '''
        month_delta = 0
        year_delta = 0
        if period[-1]=='m':
            month_delta = int(period[:-1])
        if period[-1]=='y':
            year_delta = int(period[:-1])

        date_ls = []
        rolling_period_corr_ls = []
        for date in data.index:
            year = date.year
            month = date.month
            day = date.day

            till_year = year + year_delta
            till_month = month + month_delta
            if till_month > 12:
                till_month -= 12
                till_year += 1
            till_day = day-1
            if till_day == 0:
                till_day = 31

            while True:
                try:
                    till_date = datetime.datetime(till_year, 
                                                till_month,
                                                till_day)
                    break
                except:
                    till_day -= 1
            
            if till_date >= data.index[-1]:
                break
            
            this_period_data = data.loc[date:till_date, [data.columns[0], data.columns[1]]] # 只选择了前两列
            date_ls.append(date)
            rolling_period_corr_ls.append(Backtest.get_corr_matrix(this_period_data).iloc[0, 1])
        return pd.DataFrame(index=date_ls, data=rolling_period_corr_ls, columns=[period+' rolling correlations'])

    @staticmethod
    def plot_rolling_period_returns(df:pd.DataFrame, 
                                    period='12m',
                                    bins=[-0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16]):
        '''
        ### 画出滚动时间区间的收益率频次图     
        **df**: 某个资产的价格走势      
        **period**: 滚动时间区间的长度      
        **bins**: 频次图x轴收益率的划分 
        '''
        if period[-1]=='m':
            fmt = '%Y%m'
        num = int(period[:-1])
        Ym_ls = df.index.strftime(fmt).unique().to_list()

        i = 0
        rolling_period_returns_ls = []
        while(i+num<=len(Ym_ls)):
            this_Ym_ls = Ym_ls[i:i+num]
            this_period_df = df[df.index.strftime(fmt).isin(this_Ym_ls)]
            rolling_period_returns_ls.append(Backtest.get_annualized_return(this_period_df, raw=True))
            i += 1

        data_series = pd.Series(rolling_period_returns_ls)              # 将列表转换为Pandas Series
        cut_data = pd.cut(data_series, bins=bins, include_lowest=True)  # 使用cut函数分割数据
        frequency = cut_data.value_counts().sort_index()                # 计算每个区间的频率
        plt.figure(figsize=(18,6))
        bar_width = 0.7                                                 # 调整柱子宽度
        frequency.plot(kind='bar', width=bar_width)

        plt.title('12 month rolling period returns distribution')
        plt.xlabel('Value Range')
        plt.ylabel('Frequency')
        plt.xticks(rotation=0)                                          # 0度表示标签水平

        plt.show()
        # return rolling_period_returns_ls

    @staticmethod
    def get_annualized_datas(data:pd.DataFrame, nv_col:str|int=0, benchmark:pd.DataFrame|str='000300.SH', 
                             leverage=1, 
                             fee=0,
                             trading_days=252):
        '''
        ### 得到年度的，包含收益率、超额收益率、年化波动率、最大回撤、夏普率、卡玛率的表格
        **data**: index为日期，且有某一列为价格走势的DataFrame      
        **nv_col**: 价格走势的列数，或者列名，默认为0      
        **benchmark**: 策略标的，默认为沪深300，或者给一个与data格式相同的DataFrame      
        **leverage**: 杆杆率，默认为1
        **fee**: 交易费等费率，默认为0%，单位为%
        '''
        if isinstance(nv_col, str):
            print(nv_col)
        elif isinstance(nv_col, int):
            print(data.columns[nv_col])
        else:
            raise TypeError('para nv_col should be str or int')

        if not isinstance(data.index[0], datetime.date):
            raise TypeError('the index of the pd.DataFrame should be datetime.date')
        if isinstance(nv_col, int):
            data = data.iloc[:, nv_col].to_frame()
        elif isinstance(nv_col, str):
            data = data[nv_col].to_frame()
        
        start_date = data.index[0]
        end_date = data.index[-1]
        if isinstance(benchmark, str) and benchmark!='None':
            bm = rqdatac.get_price(rqdatac.id_convert(benchmark),
                                start_date=start_date,
                                end_date=end_date,
                                frequency='1d',
                                fields=['close']).xs(rqdatac.id_convert(benchmark))
        elif isinstance(benchmark, pd.DataFrame):
            bm = benchmark
            benchmark = bm.columns[0]
        
        dct = {}

        for year in data.index.year.unique():
            this_year_df = data[data.index.year==year]
            if benchmark != 'None':
                this_year_bm = bm[bm.index.year==year]
            this_year_trading_dates = rqdatac.get_trading_dates(start_date='{}0101'.format(year),
                                                                end_date='{}1231'.format(year))
            if ((this_year_df.index[-1].strftime('%Y%m%d')!=this_year_trading_dates[-1].strftime('%Y%m%d'))
                and year==data.index.year.unique()[-1]):
                col_name = '{}ytd'.format(year)
                ratio = len(this_year_df)/len(this_year_trading_dates)
            elif ((this_year_df.index[0].strftime('%Y%m%d')!=this_year_trading_dates[0].strftime('%Y%m%d')) 
                and year==data.index.year.unique()[0]):
                col_name = '{}ytg'.format(year)
                ratio = len(this_year_df)/len(this_year_trading_dates)
            else:
                col_name = '{}'.format(year)
                ratio = 1
            this_year_datas_ls = []

            this_year_datas_ls.append(format(Backtest.get_annualized_return(this_year_df, raw=True, leverage=leverage, fee=fee*ratio, trading_days=trading_days) * 100, '.2f')+'%')       # 收益率
            if benchmark != 'None':
                this_year_datas_ls.append(format(Backtest.get_annualized_return(this_year_df, raw=True, leverage=leverage, fee=fee*ratio, trading_days=trading_days) * 100 
                                                - Backtest.get_annualized_return(this_year_bm, raw=True, leverage=leverage, fee=fee*ratio, trading_days=trading_days) * 100, '.2f')+'%')     # vs benchmark
            this_year_datas_ls.append(format(Backtest.get_annualized_volatility(this_year_df, leverage=leverage) * 100, '.2f')+'%')             # 年化波动率
            this_year_datas_ls.append(format(Backtest.get_max_drawdown(this_year_df, leverage=leverage) * 100, '.2f')+'%')                      # 最大回撤 

            try:
                this_year_datas_ls.append(format(Backtest.get_annualized_return(this_year_df, raw=True, leverage=leverage, fee=fee*ratio, trading_days=trading_days)
                                                / Backtest.get_annualized_volatility(this_year_df, leverage=leverage), '.2f'))                     # 夏普比率
            except ZeroDivisionError:
                this_year_datas_ls.append(np.nan)
                
            try:
                this_year_datas_ls.append(format(Backtest.get_annualized_return(this_year_df, raw=True, leverage=leverage, fee=fee*ratio, trading_days=trading_days)                          # 卡玛比率
                                                / Backtest.get_max_drawdown(this_year_df, leverage=leverage), '.2f'))       
            except ZeroDivisionError:
                this_year_datas_ls.append(np.nan)

            dct.update({col_name:this_year_datas_ls})

        aggregate_datas_ls = []
        aggregate_datas_ls.append(format(Backtest.get_annualized_return(data, raw=False, leverage=leverage, fee=fee, trading_days=trading_days) * 100, '.2f')+'%')       # 收益率
        if benchmark!='None':
            aggregate_datas_ls.append(format(Backtest.get_annualized_return(data, raw=False, leverage=leverage, fee=fee, trading_days=trading_days) * 100 
                                            - Backtest.get_annualized_return(bm, raw=False, leverage=leverage, fee=fee, trading_days=trading_days) * 100, '.2f')+'%')     # vs benchmark
        aggregate_datas_ls.append(format(Backtest.get_annualized_volatility(data, leverage=leverage) * 100, '.2f')+'%')             # 年化波动率
        aggregate_datas_ls.append(format(Backtest.get_max_drawdown(data, leverage=leverage) * 100, '.2f')+'%')                      # 最大回撤 

        aggregate_datas_ls.append(format(Backtest.get_annualized_return(data, raw=False, leverage=leverage, fee=fee, trading_days=trading_days)
                                        / Backtest.get_annualized_volatility(data, leverage=leverage), '.2f'))                     # 夏普比率
        aggregate_datas_ls.append(format(Backtest.get_annualized_return(data, raw=False, leverage=leverage, fee=fee, trading_days=trading_days)                          # 卡玛比率
                                        / Backtest.get_max_drawdown(data, leverage=leverage), '.2f'))       
        dct.update({'合计':aggregate_datas_ls})

        index = ['年化收益率',
                'vs {}'.format(benchmark),
                '年化波动率',
                '最大回撤',
                '夏普比率',
                '卡玛比率']
        
        if benchmark == 'None':
            index = ['年化收益率',
                     '年化波动率',
                     '最大回撤',
                     '夏普比率',
                     '卡玛比率']

        annualized_datas = pd.DataFrame(dct, index=index)
        return annualized_datas

class Beta:
    def __init__(self):
        pass

    @staticmethod    
    def ma(days, data_origin):
        '''
        ### 移动平均   
        **days**: 时间窗口的长度   
        **data_origin**: 时间序列数据
        '''
        if days != 0:
            ma_adjusted = data_origin.rolling(days).mean() 
        return ma_adjusted

    @staticmethod
    def zscore(days, data_origin):
        if days != 0:
            zscore_adjusted = (data_origin - data_origin.rolling(days).mean()) \
                                / data_origin.rolling(days).std()
        return zscore_adjusted

    @staticmethod
    def month_plus1(origin_date):
        
        if origin_date.month != 12:                
            month = origin_date.month + 1
            return pd.Timestamp(year = origin_date.year, month = month , day = 1)
        else:
            year = origin_date.year + 1
            return pd.Timestamp(year = year, month = 1 , day = 1)

    # 获取vix_date当日全部期权信息
    def get_options_data(vix_date, maturity, option_info, open_interest, volume, close_price):

        options_index = maturity.loc[vix_date, :].dropna().index
        options = pd.DataFrame(index=options_index)

        options['exe_mode'] = option_info.loc[options_index, 'exe_mode']
        options['exe_price'] = option_info.loc[options_index, 'exe_price']
        options['ttm'] = maturity.loc[vix_date, options_index]
        options['volume'] = volume.loc[vix_date, options_index]
        options['close'] = close_price.loc[vix_date, options_index]
        options['open_interest'] = open_interest.loc[vix_date, options_index]

        return options

    # 找到目标期限horizon两边到期日最近的日历日
    @staticmethod
    def get_near_next(options, horizon):
        error_ = False
        ttm = options.ttm.drop_duplicates()
        near = next_ = np.nan
        if (options.empty) | (np.min(ttm) >= 1.2*horizon):
            error_ = True
        elif (np.min(ttm) >= horizon):
            near = np.min(ttm)
            next_ = np.min(ttm.drop(ttm.idxmin()))
        else:
            near = ttm[ttm[ttm < horizon].idxmax()]
            next_ = ttm[ttm[ttm > horizon].idxmin()]
        
        return error_, near, next_

    # 获取vix_date当日无风险利率（用7日逆回购利率代理）

    def get_rf(vix_date, rf):
        return float(rf.loc[vix_date])/100

    # 找到收盘价差最小的期权 并计算forward level index

    def cal_fwd_lv(options, rf, ttm):
        # options 为给定vix_data的期权信息 含exe_mode exe_price mmt volume close
        # 限制options到期日
        options = options[options['ttm'] == ttm]
        call = options[options['exe_mode'] == 'C'].sort_values(
            by=['exe_price']).reset_index(drop=True)
        put = options[options['exe_mode'] == 'P'].sort_values(
            by=['exe_price']).reset_index(drop=True)
        price_diff = np.abs(call['close']-put['close']).min()
        strike = call.loc[np.abs(call['close']-put['close']).astype(np.float64).idxmin(), 'exe_price']
        fwd_lv = strike + np.exp(rf * ttm/365) * price_diff

        return fwd_lv

    # sigma中的求和部分，apply用

    def foo_component(x):
        return x['delta_k']*x['close']/(x['exe_price']**2)

    # 求单个期限（near或next）的sigma_squre

    def cal_sigma_square(options, fwd_lv, rf, ttm):
        # 限制options到期日
        options = options[options['ttm'] == ttm]
        # choose K0
        k0 = options[options['exe_price'] < fwd_lv].max()['exe_price']

        # get atm & oom with volume>0
        call = options[(options['exe_mode'] == 'C') & (options['exe_price'] > k0) & (
            options['volume'] > 0)].sort_values(by=['exe_price']).reset_index(drop=True)
        if call.empty:
            return np.nan
        put = options[(options['exe_mode'] == 'P') & (options['exe_price'] <= k0) & (
            options['volume'] > 0)].sort_values(by=['exe_price']).reset_index(drop=True)
        # put & call atm 取一个
        put.loc[len(put)-1,
                'close'] = np.mean(options[options['exe_price'] == k0]['close'])

        selected_options = pd.concat([put, call]).reset_index(drop=True)

        # calculate delta_k
        selected_options['delta_k'] = 0
        selected_options.loc[0, 'delta_k'] = selected_options.loc[1,
                                                                'exe_price']-selected_options.loc[0, 'exe_price']
        selected_options.loc[len(selected_options)-1, 'delta_k'] = selected_options.loc[len(
            selected_options)-1, 'exe_price']-selected_options.loc[len(selected_options)-2, 'exe_price']
        for i in range(1, len(selected_options)-1):
            selected_options.loc[i, 'delta_k'] = (
                selected_options.loc[i+1, 'exe_price']-selected_options.loc[i-1, 'exe_price'])/2
        sigma_squre = selected_options.apply(lambda x: Beta.foo_component(x), axis=1).sum(
        )*(np.exp(rf*ttm/365))*(2*365/ttm)-(((fwd_lv/k0)-1)**2)*365/ttm

        return sigma_squre

    # 插值计算目标期限的csi300_vix
    @staticmethod
    def cal_csi300_vix(horizon, options, rf):
        error_, near, next_ = Beta.get_near_next(options, horizon)
        if error_:
            csi300_vix = np.nan
        else:
            fwd_lv_near = Beta.cal_fwd_lv(options, rf, near)
            fwd_lv_next = Beta.cal_fwd_lv(options, rf, next_)
            sigma_squre_near = Beta.cal_sigma_square(options, fwd_lv_near, rf, near)
            sigma_squre_next = Beta.cal_sigma_square(options, fwd_lv_next, rf, next_)

            csi300_vix = np.sqrt((near*sigma_squre_near*(next_-horizon)/(next_-near) +
                                next_*sigma_squre_next*(horizon-near)/(next_-near))/horizon)

        return csi300_vix

    @staticmethod
    def atm_nearest_k(ori_list, price):
        k_list = ori_list.copy()
        k_list.append(price)
        k_list.sort()
        
        rank = k_list.index(price)
        if rank == 0:
            choosed_k = k_list[rank+1]
        elif rank == len(k_list)-1:
            choosed_k = k_list[rank-1]
        else:
            a_k = k_list[rank-1]
            b_k = k_list[rank+1]
            if abs(price-a_k) < abs(price-b_k):
                choosed_k = a_k
            else:
                choosed_k = b_k
        
        return choosed_k

class Covered_Call:
    pass

class Futures:
    pass

class Facotr:
    def __init__(sefl):
        pass

    # 01
    @staticmethod
    def ROC(daily_close, N=5):
        '''
        ### ROC 动量因子，输出相差N日的收盘价变化率
        daily_close: 日频收盘价
        N: 时间窗口长度
        '''      
        signal_source = (daily_close / daily_close.shift(N) - 1)
        return signal_source

    # 02
    @staticmethod
    def RET(daily_close, N=5):
        '''
        ### RET 因子，输出相差 N 日的收盘价差值
        daily_close: 日频收盘价
        N: 时间窗口长度
        '''    
        signal_source = (daily_close - daily_close.shift(N))
        return signal_source

    # 03
    @staticmethod
    def SROC(daily_close, N=5):
        '''
        ### SROC 因子计算函数，计算平滑收盘价变化率
        daily_close: 日频收盘价
        N: 时间窗口长度
        '''
        # 计算日频收盘价的指数加权移动平均值（EMA）
        ema = daily_close.ewm(span=N, adjust=False).mean()

        # 计算信号源，即 EMA 与日频收盘价的比率减 1
        signal_source = (ema / daily_close - 1)

        return signal_source

    # 04
    @staticmethod
    def COPP(daily_close, N1=2, N2=4, M=3):
        '''
        ### COPP 因子计算函数，计算 COPPOCK 曲线指标
        daily_close: 日频收盘价
        N1: 第一个时间窗口长度
        N2: 第二个时间窗口长度
        M: 加权计算的周期数
        '''
        # 计算 N1 周期的价格变化率
        rc_n1 = daily_close / daily_close.shift(N1) - 1
        # 计算 N2 周期的价格变化率
        rc_n2 = daily_close / daily_close.shift(N2) - 1
        # 将两个周期的价格变化率相加并乘以 100
        rc = (rc_n1 + rc_n2) * 100

        # 对 rc 进行加权求和，较近的数据权重更大
        weighted_rc = sum([rc.shift(i) * (M - i + 1) for i in range(1, M + 1)])
        # 计算 COPPOCK 曲线指标
        copp = weighted_rc / (M * (M + 1) / 2)

        return copp

    # 05
    @staticmethod
    def POS(daily_close, N=5):
        '''
        ### POS 因子计算函数，计算价格振荡信号
        daily_close: 日频收盘价，为 pandas.DataFrame 类型，每列代表一只股票
        N: 时间窗口长度
        '''
        # 计算 N 期的收益率
        r = daily_close / daily_close.shift(N) - 1
        # 创建一个与 daily_close 相同索引和列名的空 DataFrame 用于存储 POS 值
        pos = pd.DataFrame(index=daily_close.index, columns=daily_close.columns)

        # 遍历每只股票
        for col in daily_close.columns:
            # 获取该股票的收益率序列
            stock_returns = r[col]
            # 初始化该股票的 POS 值列表
            pos_values = [None] * len(stock_returns)

            # 从第 N 个元素开始计算 POS 值
            for t in range(N - 1, len(stock_returns)):
                # 获取最近 N 期的收益率
                recent_returns = stock_returns.iloc[t - N + 1:t + 1]
                # 计算最近 N 期收益率的最小值
                min_return = recent_returns.min()
                # 计算最近 N 期收益率的最大值
                max_return = recent_returns.max()
                # 当最大值和最小值不相等时，计算 POS 值
                if max_return != min_return:
                    pos_value = (stock_returns.iloc[t] - min_return) / (max_return - min_return) * 100
                # 当最大值和最小值相等时，POS 值设为 0
                else:
                    pos_value = 0
                # 将计算得到的 POS 值存入列表
                pos_values[t] = pos_value
            # 将该股票的 POS 值列表填入 pos DataFrame 对应的列
            pos[col] = pos_values

        return pos

    # 06
    @staticmethod
    def DELTA(daily_close, paras=[40, 60, 120, 250]):
        '''
        ### DELTA 因子计算函数，结合不同周期的统计特征生成信号源
        daily_close: 日频收盘价
        paras: 包含多个时间周期的列表，用于计算不同周期的统计量
        '''
        # 初始化信号源，形状与 daily_close 相同，所有元素初始化为 0
        signal_source = daily_close.copy() * 0
        # 计算日收益率
        ret = daily_close.pct_change()

        # 遍历每个时间周期
        for period in paras:
            # 计算指定周期内的 T 统计量
            tstat = np.sqrt(period - 1) * ret.rolling(period).mean() / ret.rolling(period).std()
            # 计算 T 统计量对应的标准正态分布的累积分布函数值
            # 当累积分布函数值为 1 时，将其替换为 np.nan
            cdf = pd.DataFrame(np.where(norm.cdf(tstat.fillna(np.inf)) == 1,
                                        np.nan, norm.cdf(tstat.fillna(np.inf))),
                            index=tstat.index, columns=tstat.columns)
            # 将 (2 * cdf - 1) 的结果累加到信号源上
            signal_source = signal_source + (2 * cdf - 1)

        return signal_source
    # 07
    @staticmethod
    def REG(daily_close, N=5):
        '''
        ### REG 因子计算函数，通过线性回归预测收盘价偏离度
        daily_close: 日频收盘价
        N: 回归窗口长度
        '''
        reg = pd.DataFrame(index=daily_close.index, columns=daily_close.columns)
        for col in daily_close.columns:
            stock_close = daily_close[col]
            reg_values = np.full(len(stock_close), np.nan)
            for t in range(N, len(stock_close)):
                X = np.arange(1, N+1).reshape(-1, 1)
                Y = stock_close.iloc[t-N:t].values
                if np.any(np.isnan(Y)): continue
                model = LinearRegression().fit(X, Y)
                Y_hat = model.coef_[0] * N + model.intercept_
                reg_values[t] = (stock_close.iloc[t] / Y_hat) - 1
            reg[col] = reg_values    
        return reg

    # 08
    @staticmethod
    def MA(daily_close, N=5):
        '''
        ### MA 因子计算函数，计算简单移动平均线
        daily_close: 日频收盘价
        N: 移动平均窗口长度
        '''
        return daily_close.rolling(window=N).mean()

    # 09
    @staticmethod
    def WMA(daily_close, N=10):
        '''
        ### WMA 因子计算函数，计算加权移动平均线
        daily_close: 日频收盘价
        N: 加权窗口长度
        '''
        weight = sum([daily_close.shift(i-1) * (N-i+1) for i in range(1, N+1)])
        return weight / (N * (N+1) / 2)

    # 10
    @staticmethod
    def EMA(daily_close, short_window=40):    
        '''
        ### EMA 因子计算函数，计算指数移动平均线与价格差值
        daily_close: 日频收盘价
        short_window: 短期指数平均窗口长度
        '''
        ema_close = daily_close.ewm(span=short_window, adjust=False).mean()
        return daily_close - ema_close

    # 11
    @staticmethod
    def BIAS(daily_close):
        '''
        ### BIAS 因子计算函数，返回不同周期的乖离率
        daily_close: 日频收盘价
        '''
        def bias(daily_close, N):
            moving_avg = daily_close.rolling(window=N).mean()
            return (daily_close / moving_avg - 1) * 100
        return bias(daily_close, 6), bias(daily_close, 12), bias(daily_close, 24)

    # 12
    @staticmethod
    def BBI(daily_close):
        '''
        ### BBI 因子计算函数，计算多空指数
        daily_close: 日频收盘价
        '''
        N = [3, 6, 12, 24]
        ma = {f'MA{n}': daily_close.rolling(window=n).mean() for n in N}
        return (ma['MA3'] + ma['MA6'] + ma['MA12'] + ma['MA24']) / 4

    # 13
    @staticmethod
    def DPO(daily_close, N=4):
        '''
        ### DPO 因子计算函数，计算区间震荡指标
        daily_close: 日频收盘价
        N: 时间窗口长度
        '''
        dpo = pd.DataFrame(index=daily_close.index, columns=daily_close.columns)
        lag = N // 2 + 1
        for col in daily_close.columns:
            stock_close = daily_close[col]
            dpo_values = [None] * len(stock_close)
            for t in range(N+lag, len(stock_close)):
                sma = stock_close.iloc[t-lag-N:t-lag].mean()
                dpo_values[t] = stock_close.iloc[t] - sma
            dpo[col] = dpo_values
        return dpo

    # 14
    @staticmethod
    def VIDYA(daily_close, N=5):
        '''
        ### VIDYA 因子计算函数，计算变异性自适应移动平均线
        daily_close: 日频收盘价
        N: 窗口长度
        '''
        weight = sum([abs(daily_close.shift(i) - daily_close.shift(i+1)) for i in range(N)])
        vi = abs(daily_close - daily_close.shift(N-1)) / weight
        return vi * daily_close + (1 - vi) * daily_close.shift(1)

    # 15
    @staticmethod
    def KST(daily_close, M=5):
        '''
        ### KST 因子计算函数，计算 Know Sure Thing 指标
        daily_close: 日频收盘价
        M: 移动平均周期
        '''
        def calculate_diff(daily_close, N): return daily_close.diff(N)
        def calculate_maroc(daily_close, N, M): 
            return calculate_diff(daily_close, N).rolling(window=M, min_periods=1).mean()
        
        maroc_10 = calculate_maroc(daily_close, 10, M)
        maroc_15 = calculate_maroc(daily_close, 15, M)
        maroc_20 = calculate_maroc(daily_close, 20, M)
        maroc_30 = calculate_maroc(daily_close, 30, M)
        
        index_KST = (maroc_10 + 2*maroc_15 + 3*maroc_20 + 4*maroc_30)
        return index_KST.rolling(window=9, min_periods=1).mean()

    # 16
    @staticmethod
    def MOM_MA(daily_close, N=5):
        '''
        ### MOM_MA 因子计算函数，计算累积收益率
        daily_close: 日频收盘价
        N: 累积周期
        '''
        return daily_close.pct_change().rolling(window=N, min_periods=1).sum()

    # 17
    @staticmethod
    def DBCD(daily_close, N1=2, N2=3, N=4):
        '''
        ### DBCD 因子计算函数，计算异同离差乖离率
        daily_close: 日频收盘价
        N1: 短期差分周期
        N2: 移动平均周期
        N: 基准周期
        '''
        ma_N = daily_close.rolling(window=N, min_periods=1).mean()
        bias = (daily_close - ma_N) / ma_N * 100
        bias_diff = bias - bias.shift(N1-1)
        return bias_diff.rolling(window=N2, min_periods=1).mean()

    # 18
    @staticmethod
    def TMA(daily_close, N=5):
        '''
        ### TMA 因子计算函数，计算三重移动平均线
        daily_close: 日频收盘价
        N: 移动平均窗口长度
        '''
        close = daily_close.rolling(window=N, min_periods=1).mean()
        return close.rolling(window=N, min_periods=1).mean()

    # 19
    @staticmethod
    def TSI(daily_close, N1=3, N2=4):
        '''
        ### TSI 因子计算函数，计算趋势强度指标
        daily_close: 日频收盘价
        N1: 短期指数平均周期
        N2: 长期指数平均周期
        '''
        diff = daily_close.diff()
        abs_diff = diff.abs()
        EMA_Diff = diff.ewm(span=N1, adjust=False).mean()
        EMA_Abs_Diff = abs_diff.ewm(span=N1, adjust=False).mean()
        TSI_EMA_Diff = EMA_Diff.ewm(span=N2, adjust=False).mean()
        TSI_EMA_Abs_Diff = EMA_Abs_Diff.ewm(span=N2, adjust=False).mean()
        return TSI_EMA_Diff / TSI_EMA_Abs_Diff * 100

    # 20
    @staticmethod
    def DEMA(daily_close, N=5):
        '''
        ### DEMA 因子计算函数，计算双指数移动平均线
        daily_close: 日频收盘价
        N: 指数平均窗口长度
        '''
        ema1 = daily_close.ewm(span=N, adjust=False).mean()
        ema2 = ema1.ewm(span=N, adjust=False).mean()
        return 2*ema1 - ema2

    # 21
    @staticmethod
    def ROC_CHANGE(daily_close, short_window=40, long_window=60):
        '''
        ### ROC_CHANGE 因子计算函数，计算不同周期ROC差值
        daily_close: 日频收盘价
        short_window: 短期ROC周期
        long_window: 长期ROC周期
        '''
        ROC1 = daily_close / daily_close.shift(short_window) - 1
        ROC2 = daily_close / daily_close.shift(long_window) - 1        
        return ROC1 - ROC2

    # 22
    @staticmethod
    def EFFICIENCY(daily_close, N1=2, N2=4):
        '''
        ### EFFICIENCY 因子计算函数，计算效率比率差值
        daily_close: 日频收盘价
        N1: 短期效率周期
        N2: 长期效率周期
        '''
        def effi(daily_close, N):
            r_N = daily_close / daily_close.shift(N) - 1
            effi = pd.DataFrame(index=daily_close.index, columns=daily_close.columns)
            for col in daily_close.columns:
                effi_values = [None] * len(daily_close)
                for t in range(N-1, len(daily_close)):
                    efmom = sum(daily_close.iloc[t-i]/daily_close.iloc[t-i-1]-1 for i in range(N))/N
                    effi_values[t] = r_N[col].iloc[t] / efmom if efmom != 0 else None
                effi[col] = effi_values
            return effi
        return effi(daily_close, N1) - effi(daily_close, N2)

    # 23
    @staticmethod
    def INVVOL(daily_close, N1=2, N2=4):
        '''
        ### INVVOL 因子计算函数，计算波动率倒数差值
        daily_close: 日频收盘价
        N1: 短期波动率周期
        N2: 长期波动率周期
        '''
        returns = daily_close / daily_close.shift(1) - 1
        def calculate_invv(N, returns):
            std_dev = returns.rolling(window=N).std()
            return 1 / std_dev
        invvol_N1 = calculate_invv(N1, returns)
        invvol_N2 = calculate_invv(N2, returns)
        return invvol_N1 - invvol_N2

    # 24
    @staticmethod
    def AVG_LINE(daily_close, short_window=2, long_window=4):
        '''
        ### AVG_LINE 因子计算函数，计算移动平均线差值
        daily_close: 日频收盘价
        short_window: 短期窗口长度
        long_window: 长期窗口长度
        '''
        return daily_close.rolling(window=short_window).mean() - daily_close.rolling(window=long_window).mean()

    # 25
    @staticmethod
    def EXPMA(daily_close, short_window=2, long_window=4):
        '''
        ### EXPMA 因子计算函数，计算指数移动平均线差值
        daily_close: 日频收盘价
        short_window: 短期指数平均周期
        long_window: 长期指数平均周期
        '''
        expma_N1 = daily_close.ewm(span=short_window, adjust=False).mean()
        expma_N2 = daily_close.ewm(span=long_window, adjust=False).mean()
        return expma_N1 - expma_N2

    # 26
    @staticmethod
    def MOM(daily_close, short_window=2, long_window=4):
        '''
        ### MOM 因子计算函数，计算动量差值
        daily_close: 日频收盘价
        short_window: 短期动量周期
        long_window: 长期动量周期
        '''
        returns = daily_close / daily_close.shift(1) - 1    
        return returns.rolling(window=short_window).mean() - returns.rolling(window=long_window).mean()

    # 27
    @staticmethod
    def SHARP_MOM(daily_close, short_window=2, long_window=4):
        '''
        ### SHARP_MOM 因子计算函数，计算夏普动量差值
        daily_close: 日频收盘价
        short_window: 短期夏普周期
        long_window: 长期夏普周期
        '''
        returns = daily_close / daily_close.shift(1) - 1
        def calculate_sharp_mom(N, returns):
            mean = returns.rolling(window=N).mean()
            std = returns.rolling(window=N).std()
            return mean / std.replace(0, np.nan)
        return calculate_sharp_mom(short_window, returns) - calculate_sharp_mom(long_window, returns)

    # 28
    @staticmethod
    def HULLMA(daily_close, short_window=20, long_window=60):    
        '''
        ### HULLMA 因子计算函数，计算赫尔移动平均线差值
        daily_close: 日频收盘价
        short_window: 短期赫尔周期
        long_window: 长期赫尔周期
        '''
        nema_short = 2*daily_close.ewm(span=int(short_window/2), adjust=False).mean() - daily_close.ewm(span=short_window, adjust=False).mean()
        nema_long = 2*daily_close.ewm(span=int(long_window/2), adjust=False).mean() - daily_close.ewm(span=long_window, adjust=False).mean()
        hullma_short = nema_short.ewm(span=int(short_window**0.5), adjust=False).mean()
        hullma_long = nema_long.ewm(span=int(long_window**0.5), adjust=False).mean()                
        return hullma_short - hullma_long

    # 29
    @staticmethod
    def MACD(daily_close, N1=3, N2=4, N3=5):
        '''
        ### MACD 因子计算函数，计算移动平均收敛散度
        daily_close: 日频收盘价
        N1: 短期指数平均周期
        N2: 长期指数平均周期
        N3: 信号线周期
        '''
        ema_N1 = daily_close.ewm(span=N1, adjust=False).mean()
        ema_N2 = daily_close.ewm(span=N2, adjust=False).mean()
        DIF = ema_N1 - ema_N2
        DEA = DIF.ewm(span=N3, adjust=False).mean()
        return DIF - DEA

    # 30
    @staticmethod
    def ZLMACD(daily_close, N1=2, N2=4):
        '''
        ### ZLMACD 因子计算函数，计算增强型MACD
        daily_close: 日频收盘价
        N1: 短期指数平均周期
        N2: 长期指数平均周期
        '''
        ema_N1 = daily_close.ewm(span=N1, adjust=False).mean()
        ema_N2 = daily_close.ewm(span=N2, adjust=False).mean()
        ema_N1_ = ema_N1.ewm(span=N1, adjust=False).mean()
        ema_N2_ = ema_N2.ewm(span=N2, adjust=False).mean()
        return (2*ema_N1-ema_N1_) - (2*ema_N2-ema_N2_)

    # 31
    @staticmethod
    def PPO(daily_close, N1=3, N2=4, N3=5):
        '''
        ### PPO 因子计算函数，计算百分比价格振荡指标
        daily_close: 日频收盘价
        N1: 短期指数平均周期
        N2: 长期指数平均周期
        N3: 信号线周期
        '''
        ema_N1 = daily_close.ewm(span=N1, adjust=False).mean()
        ema_N2 = daily_close.ewm(span=N2, adjust=False).mean()
        ppo = ema_N1 / ema_N2 - 1
        emappo = ppo.ewm(span=N3, adjust=False).mean()
        return ppo - emappo

    # 32
    @staticmethod
    def OSC(daily_close, N=3, M=3):
        '''
        ### OSC 因子计算函数，计算震荡指标
        daily_close: 日频收盘价
        N: 基准周期
        M: 移动平均周期
        '''
        osc = daily_close - daily_close.rolling(window=N).mean()
        oscma = osc.rolling(window=M).mean()
        return osc - oscma

    # 33
    @staticmethod
    def PMO(daily_close, N1=3, N2=4, N3=5):
        '''
        ### PMO 因子计算函数，计算价格振荡指标
        daily_close: 日频收盘价
        N1: 短期平滑系数周期
        N2: 中期平滑系数周期
        N3: 长期平滑系数周期
        '''
        roc = daily_close.pct_change() * 100
        def DMA(daily_close, X): return X*daily_close + (1-X)*daily_close.shift(1)
        dmaroc = 10 * DMA(roc, 2/N1)
        pmo = DMA(dmaroc, 2/N2)
        dmapmo = DMA(pmo, 2/(N3+1))
        return pmo - dmapmo

    # 34
    @staticmethod
    def BIAS36(daily_close):
        '''
        ### BIAS36 因子计算函数，计算3日与6日乖离率的移动平均
        daily_close: 日频收盘价
        '''
        N1, N2, M = 3, 6, 6
        bias36 = daily_close.rolling(window=N1).mean() - daily_close.rolling(window=N2).mean()
        return bias36.rolling(window=M).mean()

    # 35
    @staticmethod
    def CMO(daily_close, N=5):
        '''
        ### CMO 因子计算函数，计算蔡金资金流量指标
        daily_close: 日频收盘价
        N: 窗口长度
        '''
        diff = daily_close.diff()
        up_diff = diff.clip(lower=0)
        down_diff = -diff.clip(upper=0)
        sum_up = up_diff.rolling(window=N).sum()
        sum_down = down_diff.rolling(window=N).sum()
        return (sum_up + sum_down) / (sum_up - sum_down) * 100 if (sum_up - sum_down).ne(0) else 0

    # 36
    @staticmethod
    def PSY(daily_close, N=5):
        '''
        ### PSY 因子计算函数，计算心理线指标
        daily_close: 日频收盘价
        N: 心理周期
        '''
        diff = daily_close.diff()
        indexdiff = (diff.clip(lower=0) / diff.abs()).fillna(0)
        return indexdiff.rolling(window=N).mean() * 100

    # 37
    @staticmethod
    def UP2DOWN(daily_close, N=5, M=0):
        '''
        ### UP2DOWN 因子计算函数，计算多空收益比
        daily_close: 日频收盘价
        N: 收益计算周期
        M: 基准阈值
        '''
        ret = daily_close.pct_change()
        up_ret = ret.clip(lower=0)
        down_ret = ret.clip(upper=0)
        up_r_N = up_ret.rolling(window=N).mean()
        down_r_N = down_ret.rolling(window=N).mean()
        up_r_N_ = (up_ret / ret.abs()).fillna(0).rolling(window=N).mean()
        down_r_N_ = (down_ret / ret.abs()).fillna(0).rolling(window=N).mean()
        up2down = (up_r_N / up_r_N_) / (down_r_N / down_r_N_).fillna(1)
        return up2down - M

    # 38
    @staticmethod
    def RSIH(daily_close, N1=3, N2=4):
        '''
        ### RSIH 因子计算函数，计算相对强弱指数差
        daily_close: 日频收盘价
        N1: 短期RSI周期
        N2: 长期EMA周期
        '''
        diff = daily_close.diff()
        up_diff = diff.clip(lower=0)
        down_diff = diff.clip(upper=0)
        sma_up = up_diff.rolling(window=N1).mean()
        sma_updown = (up_diff - down_diff).rolling(window=N1).mean()
        rsi = sma_up / sma_updown * 100 if sma_updown.ne(0) else 0
        return rsi - rsi.ewm(span=N2, adjust=False).mean()

    # 39
    @staticmethod
    def TII(daily_close, N=5, M=0):
        '''
        ### TII 因子计算函数，计算趋势强度指标
        daily_close: 日频收盘价
        N: 基准周期
        M: 基准阈值
        '''
        dev = daily_close - daily_close.rolling(window=N).mean()
        up_dev = dev.clip(lower=0)
        down_dev = dev.clip(upper=0)
        P = N // 2 + 1
        sum_up = up_dev.rolling(window=int(P)).sum()
        sum_down = down_dev.rolling(window=int(P)).sum()
        tii = sum_up / (sum_up - sum_down) if (sum_up - sum_down).ne(0) else 0
        return tii - M

    # 40
    @staticmethod
    def THRES_AVG(daily_close, N=5, M=0):
        '''
        ### THRES_AVG 因子计算函数，计算偏离度均值
        daily_close: 日频收盘价
        N: 平均周期
        M: 基准阈值
        '''
        dev = daily_close - daily_close.rolling(window=N).mean()
        index_dev = (dev / dev.abs()).clip(lower=0)
        return index_dev.rolling(window=N).mean() - M

    # 41
    @staticmethod
    def MASS(daily_close, N=5, M=0):
        '''
        ### MASS 因子计算函数，计算质量指标
        daily_close: 日频收盘价
        N: 窗口长度
        M: 基准阈值
        '''
        moving_avg = daily_close.rolling(window=N).mean()
        index_mass = (daily_close - moving_avg).div((daily_close - moving_avg).abs()).clip(lower=0)
        return index_mass.rolling(window=N).mean() - M
 
class Risk_Parity:
    def __init__(self, df):
        '''
        df: pd.DataFrame, time as index and asset as columns, the values are prices, typically close price      
        '''
        self.num_assets = len(df.columns)
    
    @staticmethod
    def get_returns(prices:pd.DataFrame|pd.Series):
        '''
        ### 计算资产的回报率，由相邻时间的价格变化得到
        prices: 资产的价格走势
        '''
        return prices.pct_change().iloc[1:,:]

    @staticmethod
    def get_cov_matrix(returns):
        '''
        ### 计算协方差矩阵
        **returns**: 各类资产的回报率
        '''
        return returns.cov()

    @staticmethod
    def get_weights(cov_matrix, lb=0, ub=1, risk_contributions=None):
        '''
        ### 根据风险平价计算各个资产的权重       
        **cov_matrix**: 各类资产的协方差矩阵        
        **lb**: 任一资产的权重下限，默认为0        
        **ub**: 任一资产的权重上限，默认为1    
        **risk_contributions**: 各个资产的风险预算，默认每个资产预算相同
        '''
        num_assets = len(cov_matrix)

        lb = [lb] * num_assets
        ub = [ub] * num_assets

        # Handle default and single values for risk contributions
        if risk_contributions is None:
            risk_contributions = np.ones(num_assets) / num_assets
        elif isinstance(risk_contributions, float) or round(sum(risk_contributions), 2) != 1:
            raise ValueError('the sum of risk_contributions need to be 1')

        # Prepare initial weights, bounds, and constraints
        initial_weights = np.ones(num_assets) / num_assets
        bounds = list(zip(lb, ub))
        constraints = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})

        # Define the objective function
        def objective(weights, cov_mat, rc):
            portfolio_variance = weights.T @ cov_mat @ weights
            asset_contributions = weights * (cov_mat @ weights) / portfolio_variance  # Normalize the asset contributions
            return np.sum((asset_contributions - rc) ** 2)

        # Run the optimization
        result = minimize(objective, initial_weights, args=(cov_matrix, risk_contributions),
                        bounds=bounds, constraints=constraints, method='SLSQP')

        # Create a Series for the results
        if result.success:
            weights = pd.Series(result.x, index=cov_matrix.columns)
        else:
            print("Optimization failed")
            weights = pd.Series([np.nan] * num_assets, index=cov_matrix.columns)

        return weights

class CPPI:
    '''
    ### 初始化一个CPPI实例
    **data**: index为日期，且包含风险资产和无风险资产的价格走势的DataFrame   
    **TIPP**: 是否切换为TIPP（可变价值底线），默认为False   
    **rate**: 保本率，默认为0.9   
    **multiplier**: 风险乘数，默认为3    
    **bound**: 风险资产比例上限，默认为1   
    **asset**: 期初总资产，默认为1
    '''
    def __init__(self, data:pd.DataFrame, TIPP=False, rate=0.9, multiplier=3, bound=1, asset=1):
        self.data = data    # 几类资产的价格走势
        self.TIPP = TIPP    # 是否切换为TIPP
        self.rate = rate                # 保本率
        self.multiplier = multiplier    # 风险乘数
        self.bound = bound              # 风险资产比例上限
        self.asset = asset              # 期初资产
        self.floor = asset * rate               # 价值底线
        self.cushion = asset - self.floor       # 安全垫
        self.risk = self.cushion * multiplier   # 风险投资
        self.nrisk = asset - self.risk          # 无风险投资
        self.weight_df = None                       
    
    def monthly_adjust(self, risk_col:str, nrisk_col:str):
        '''
        ### 使用 CPPI（TIPP）进行月频调仓   
        **risk_col**: 风险资产在data中的所在列   
        **nrisk_col**: 无风险资产在data中的所在列
        '''
        data = self.data                # 几类资产的价格走势
        TIPP = self.TIPP                # 是否切换为TIPP
        rate = self.rate                # 保本率
        multiplier = self.multiplier    # 风险乘数
        bound = self.bound              # 风险资产比例上限
        asset = self.asset              # 期初资产
        floor = self.floor              # 价值底线
        cushion = self.cushion          # 安全垫
        risk = self.risk                # 风险投资
        nrisk = self.nrisk              # 无风险投资

        cppi = []
        risk_Q = []
        nrisk_Q = []
        date_ls = []
   
        risk_Q.append(risk / data.iloc[0][risk_col])
        nrisk_Q.append(nrisk / data.iloc[0][nrisk_col])

        weight_df = pd.DataFrame(index=data.index, columns=['risk weight', 'nrisk weight', 'asset'])
        # 月频调仓
        month = data.index[0].month
        for date in data.index:
            last_month = month
            month = date.month
            if month != last_month:   
                asset = (risk_Q[-1] * data.loc[date.strftime('%Y-%m-%d')][risk_col]
                                + nrisk_Q[-1] * data.loc[date.strftime('%Y-%m-%d')][nrisk_col])
                weight_df.loc[date, 'asset'] = asset
                weight_df.loc[date, 'risk weight'] = risk_Q[-1] * data.loc[date.strftime('%Y-%m-%d')][risk_col] / asset
                weight_df.loc[date, 'nrisk weight'] = nrisk_Q[-1] * data.loc[date.strftime('%Y-%m-%d')][nrisk_col] / asset
                if TIPP:
                    floor = max(floor, asset*rate)
                cushion = max(asset - floor, 0)

                risk = min(cushion*multiplier, asset*bound)     # 风险投资
                nrisk = asset - risk                            # 无风险投资

                cppi.append(asset)
                risk_Q.append(risk / data.loc[date.strftime('%Y-%m-%d')][risk_col])
                nrisk_Q.append(nrisk / data.loc[date.strftime('%Y-%m-%d')][nrisk_col])
                date_ls.append(date.strftime('%Y-%m-%d'))

            else:
                asset = (risk_Q[-1] * data.loc[date.strftime('%Y-%m-%d')][risk_col]
                                + nrisk_Q[-1] * data.loc[date.strftime('%Y-%m-%d')][nrisk_col])
                weight_df.loc[date, 'asset'] = asset
                weight_df.loc[date, 'risk weight'] = risk_Q[-1] * data.loc[date.strftime('%Y-%m-%d')][risk_col] / asset
                weight_df.loc[date, 'nrisk weight'] = nrisk_Q[-1] * data.loc[date.strftime('%Y-%m-%d')][nrisk_col] / asset
                cppi.append(asset)
                date_ls.append(date.strftime('%Y-%m-%d'))
        self.weight_df = weight_df       # 包含风险资产、无风险资产比例的DataFrame         
        col_name = 'CPPI'
        if TIPP:
            col_name = 'TIPP'
        return pd.DataFrame(index=date_ls, data={col_name: cppi})

class Econometrics:
    def __init__(self):
        pass
    
    class LinearReg:
        def __init__(self, X, y):
            self.X = X
            self.y = y
            self.model = LinearRegression()
            self.model.fit(X, y)
            self.y_pred = self.model.predict(X)

            self.mse = mean_squared_error(self.y, self.y_pred)
            self.r2 = r2_score(self.y, self.y_pred)

        def plot_linear_regression(self, ):
            '''
            ### 画出散点图和对应的回归线   
            **data**: index为日期，第一列为资产价格走势的DataFrame   
            **raw**: 是否返回非年化的数据（用于计算Ytm或者Ytg），默认为False
            '''
            pass

def norm(data:list|pd.DataFrame, to_value:int|float=1):
    '''
    ### 将价格数据标准化，使得期初价格为某个值
    **data**: 资产的价格走势   
    **to_value**: 将期初价格放缩到该值，默认为1
    '''
    data = data.copy()
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            for i in range(len(data)):
                if not pd.isna(data.iloc[i][col]):
                    if data.iloc[i][col] == 0:
                        raise ValueError('the first value cannot be zero')
                    else:
                        data.loc[:, col] = data.loc[:, col] / data.iloc[i][col] * to_value
                        break
        return data
    elif isinstance(data, list):
        if data[0] == 0:
            raise ValueError('the first value cannot be zero')
        return [data[i]/data[0]*to_value for i in range(len(data))]

def fill_nan_linear(data):
    '''
    ### 线性填充缺失值
    #### 如果缺失值在列的中间，使用直线（线性插值）填充
    #### 如果缺失值在列的开头，与第一个非缺失值保持一致
    #### 如果缺失值在列的结尾，与最后一个非缺失值保持一致
    **data**: 资产的价格走势   
    '''
    # 复制数据，避免修改原始数据
    data = data.copy()
    
    # 遍历每一列
    for col in data.columns:
        # 确保列数据是数值类型
        if isinstance(data[col], pd.Series):  # 确保是 Series
            data[col] = pd.to_numeric(data[col], errors='coerce')
        else:
            # 如果列不是 Series，尝试将其转换为 Series
            try:
                data[col] = pd.Series(data[col].values.flatten())
                data[col] = pd.to_numeric(data[col], errors='coerce')
            except Exception as e:
                raise TypeError(f"Column '{col}' cannot be converted to a Series. Error: {e}")
        
        # 如果某列全部为 NaN，跳过该列
        if data[col].isna().all():
            continue
        
        # 处理开头的缺失值
        first_valid_index = data[col].first_valid_index()
        if first_valid_index is not None:
            data.loc[data[col].isna() & (data.index < first_valid_index), col] = data.loc[first_valid_index, col]
        
        # 处理结尾的缺失值
        last_valid_index = data[col].last_valid_index()
        if last_valid_index is not None:
            data.loc[data[col].isna() & (data.index > last_valid_index), col] = data.loc[last_valid_index, col]
    
    # 对中间的缺失值进行线性插值
    data = data.interpolate(method='linear')
    
    return data

# 2024.11.17 - new class Econometrics
# 2024.12.01 - new class Beta
# 2025.03.08 - new class Factor
# 2025.04.25 - last edit