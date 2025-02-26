import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import cy_backtest as bt
import rqdatac
rqdatac.init('license','MVLMpkusnVpHpE1JKm2gdXbkg6NBNQwH83Kb78-vBtdPd6BL03ke1XOCkUjzoP9FQxZ4PSWJEmrCp6iUXrnvdxE4OPh8HKrSUxpRwVVQtVq7Vfh1-TuJ-4FNgX51wtrnird4Mtoj5XUhdU_lpuFPyr9jBBA0ftj3kjMmjf2Io0Y=PQoU58SJxz8CHOeGMUKVQJ-oFSzKL971HEd4UOf10KjVqya6bMTgBW9wa_QBIFN3JrE0CPx_ADUk6sJs1ngEXCiRvIRysu_3xX4M57AQJbazDxAyhjWEM7qTyAVnh308SsAOOH3xSGyKFT3zq1ESg1JCFuX19gCXxXUqlqTKask=')

import os
cwd = os.getcwd()

daily_close = pd.read_excel('close.xlsx', index_col=0).loc['2016-01-01':,:]

def converter(factor, num):
    signal = factor.copy().resample('ME').last()

    signal_output = pd.DataFrame(np.where(signal.rank(axis=1, ascending=False, method='first').le(num), 1, 0),
                                 index=signal.index, 
                                 columns=signal.columns)
    
    return signal_output

def adjuster_for_equal_weights(raw:pd.DataFrame, signal:pd.DataFrame):

    signal = signal.copy().loc['2020-01-01':,:]
    
    converted_signal = pd.DataFrame(0, index=range(10), columns=signal.index)
    for column in converted_signal.columns:
        converted_signal[column] = signal.loc[column][signal.loc[column]==1].index

    position_id = converted_signal.reset_index().drop(['index'],axis=1) 

    data = raw.copy().loc['2020-01-01':,:]
    last_month = data.index[0].month
    signal_col = 0
    for date in data.index:
        month = date.month
        if month != last_month:   
            # 等权重调仓
            this_month_signal = position_id.iloc[:, signal_col].to_list()
            weights = pd.Series(dict(zip(this_month_signal, [1/10]*10)))

            # 根据权重调仓
            total_assets = sum(data.loc[date.strftime('%Y-%m-%d')])
            data.loc[date.strftime('%Y-%m-%d'):, :] = 0
            for col in weights.index:
                data.loc[date.strftime('%Y-%m-%d'):, col] = bt.norm(raw.loc[date.strftime('%Y-%m-%d'):, col].to_frame(), 
                                                                    total_assets*weights[col]).iloc[:,0]
            
            # 记录上次调仓时间
            signal_col += 1
        else:
            pass
        last_month = month

    data['Equal Weights'] = data.mean(axis=1)
    # bt.Backtest.get_annualized_datas(data, nv_col='Equal Weights', benchmark='None')
    return bt.Backtest.get_annualized_return(data['Equal Weights'].to_list())

# backtest begin
return_list = []

def ROC(daily_close, N):
    
    signal_source = (daily_close / daily_close.shift(N) - 1)
    
    return signal_source

factor_1_ROC = ROC(daily_close, 5)
signal = converter(factor_1_ROC, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def RET(daily_close, N):
    
    signal_source = (daily_close - daily_close.shift(N))
    
    return signal_source

factor_2_RET = RET(daily_close, 5)
signal = converter(factor_2_RET, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def SROC(daily_close, N):
    
    ema = daily_close.ewm(span=N, adjust=False).mean()
    
    signal_source = (ema / daily_close - 1)
    
    return signal_source

factor_3_SROC = SROC(daily_close, 5)
signal = converter(factor_3_SROC, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def COPP(daily_close, N1, N2, M):
    
    rc_n1 = daily_close / daily_close.shift(N1) - 1
    rc_n2 = daily_close / daily_close.shift(N2) - 1
    rc = (rc_n1 + rc_n2) * 100
    
    weighted_rc = sum([rc.shift(i) * (M - i + 1) for i in range(1, M + 1)])
    copp = weighted_rc / (M * (M + 1) / 2)
    
    return copp

factor_4_COPP = COPP(daily_close, 2, 4, 3)
signal = converter(factor_4_COPP, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def POS(daily_close, N):
    
    r = daily_close / daily_close.shift(N) - 1
    pos = pd.DataFrame(index=daily_close.index, columns=daily_close.columns)
    
    for col in daily_close.columns:
        stock_returns = r[col]
        pos_values = [None] * len(stock_returns)
        
        for t in range(N-1, len(stock_returns)):
            recent_returns = stock_returns.iloc[t-N+1:t+1]
            min_return = recent_returns.min()
            max_return = recent_returns.max()
            if max_return != min_return:
                pos_value = (stock_returns.iloc[t] - min_return) / (max_return - min_return) * 100
            else:
                pos_value = 0  # 当最大值和最小值相等时，POS 值设为 0
            pos_values[t] = pos_value
        # 将 POS 值填入 DataFrame 中
        pos[col] = pos_values

    return pos

factor_5_POS = POS(daily_close, 5)
signal = converter(factor_5_POS, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def DELTA(daily_close, paras=[40,60,120,250]):

    signal_source = daily_close.copy() * 0
    # 日收益率
    ret = daily_close.pct_change()
        
    for period in paras:
        # 计算T统计量
        tstat = np.sqrt(period - 1) * ret.rolling(period).mean() / ret.rolling(period).std()
        # 计算概率密度分布
        cdf = pd.DataFrame(np.where(norm.cdf(tstat.fillna(np.inf)) == 1,
                           np.nan, norm.cdf(tstat.fillna(np.inf))),
                          index=tstat.index, columns=tstat.columns)
        signal_source = signal_source + (2 * cdf - 1)

    return signal_source

factor_6_DELTA = DELTA(daily_close, paras=[40,60,120,250])
signal = converter(factor_6_DELTA, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def REG(daily_close, N):
    # 初始化 REG 值的 DataFrame，与 daily_close 结构相同
    reg = pd.DataFrame(index=daily_close.index, columns=daily_close.columns)

    for col in daily_close.columns:
        stock_close = daily_close[col]
        reg_values = np.full(len(stock_close), np.nan)
        # 对每个时间点进行回归分析
        for t in range(N, len(stock_close)):
            # 自变量 X 生成一个1到N+1-1=N的一维数组再转换为二维数组
            X = np.arange(1, N + 1).reshape(-1, 1)
            # 因变量 Y
            Y = stock_close.iloc[t-N:t].values
            if np.any(np.isnan(Y)):  # 检查是否有 NaN 值
                continue  # 跳过包含 NaN 的情况
            # 线性回归模型
            model = LinearRegression().fit(X, Y)
            a = model.coef_[0]
            b = model.intercept_
            # 计算拟合值
            Y_hat = a * N + b
            # 计算 REG 值
            reg_value = (stock_close.iloc[t] / Y_hat) - 1
            reg_values[t] = reg_value
        # 将 REG 值填入 DataFrame 中
        reg[col] = reg_values    
    
    return reg
    
factor_7_REG = REG(daily_close, 5)
signal = converter(factor_7_REG, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def MA(daily_close, N):
    
    ma = daily_close.rolling(window=N).mean()
    
    return ma

factor_8_MA = MA(daily_close, 5)
signal = converter(factor_8_MA, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def WMA(daily_close, N):
    
    weight = sum([daily_close.shift(i - 1) * (N - i + 1) for i in range(1, N + 1)])
    wma = weight / (N * (N + 1) / 2)

    return wma

factor_9_WMA = WMA(daily_close, 10)
signal = converter(factor_9_WMA, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def EMA(daily_close, short_window):    
           
    ema_close = daily_close.ewm(span=short_window, adjust=False).mean()
    # 指标信号
    signal_source = daily_close - ema_close

    return signal_source

factor_10_EMA = EMA(daily_close, 40)
signal = converter(factor_10_EMA, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def BIAS(daily_close):
    
    def bias(daily_close, N):
        # 计算移动平均值
        moving_avg = daily_close.rolling(window=N).mean()
        # 计算 BIAS 值
        bias = (daily_close / moving_avg - 1) * 100

        return bias
    
    return bias(daily_close, 6), bias(daily_close, 12), bias(daily_close, 24)
    
factor_11_BIAS6, factor_11_BIAS12, factor_11_BIAS24 = BIAS(daily_close)
signal = converter(factor_11_BIAS6, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def BBI(daily_close):
    
    N = [3, 6, 12, 24]
   
    def cal_ma(daily_close, N):
        ma = {}
        for n in N:
            ma[f'MA{n}'] = daily_close.rolling(window=n).mean()
        return ma
    # 计算不同窗口长度的移动平均值
    ma = cal_ma(daily_close, N)
    # 计算 BBI 值
    bbi = (ma['MA3'] + ma['MA6'] + ma['MA12'] + ma['MA24']) / 4

    return bbi

factor_12_BBI = BBI(daily_close)
signal = converter(factor_12_BBI, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def DPO(daily_close, N): # 时间窗口加滞后
    # 初始化 DPO 值的 DataFrame，与 daily_close 结构相同
    dpo = pd.DataFrame(index=daily_close.index, columns=daily_close.columns)
    # 计算滞后期
    lag = N // 2 + 1
    
    for col in daily_close.columns:
        stock_close = daily_close[col]
        dpo_values = [None] * len(stock_close)
        
        for t in range(N+lag, len(stock_close)):
            # 计算滞后 N 天的移动平均值
            sma = stock_close.iloc[t-lag-N:t-lag].mean()
            # 计算 DPO 值
            dpo_value = stock_close.iloc[t] - sma
            dpo_values[t] = dpo_value
        # 将 DPO 值填入 DataFrame 中
        dpo[col] = dpo_values
    
    return dpo

factor_13_DPO = DPO(daily_close, 4)
signal = converter(factor_13_DPO, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))
 
def VIDYA(daily_close, N):
    
    weight = sum([abs(daily_close.shift(i) - daily_close.shift(i + 1)) for i in range(N)])
    vi = abs(daily_close - daily_close.shift(N - 1)) / weight
    
    vidya = vi * daily_close + (1 - vi) * daily_close.shift(1)
    
    return vidya

factor_14_VIDYA = VIDYA(daily_close, 5)
signal = converter(factor_14_VIDYA, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def KST(daily_close, M):
    
    def calculate_diff(daily_close, N):
        return daily_close.diff(N)

    def calculate_maroc(daily_close, N, M):
        diffs = calculate_diff(daily_close, N)
        return diffs.rolling(window=M, min_periods=1).mean()

    maroc_10 = calculate_maroc(daily_close, 10, M)
    maroc_15 = calculate_maroc(daily_close, 15, M)
    maroc_20 = calculate_maroc(daily_close, 20, M)
    maroc_30 = calculate_maroc(daily_close, 30, M)
    
    index_KST = (maroc_10 + 2 * maroc_15 + 3 * maroc_20 + 4 * maroc_30)
        
    return index_KST.rolling(window=9, min_periods=1).mean()

factor_15_KST = KST(daily_close, 5)
signal = converter(factor_15_KST, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def MOM_MA(daily_close, N):
    
    ret = daily_close.pct_change()
    
    mom = ret.rolling(window=N, min_periods=1).sum()
    
    return mom

factor_16_MOM_MA = MOM_MA(daily_close, 5)
signal = converter(factor_16_MOM_MA, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def DBCD(data, N1, N2, N):
    # 计算过去 N 天的移动平均值
    ma_N = daily_close.rolling(window=N, min_periods=1).mean()
    # 计算 BIAS
    bias = (daily_close - ma_N) / ma_N * 100
    bias_diff = bias - bias.shift(N1 - 1)
    # 计算 BIAS 的 SMA
    dbcd = bias_diff.rolling(window=N2, min_periods=1).mean()
    
    return dbcd

factor_17_DBCD = DBCD(daily_close, 2, 3, 4)
signal = converter(factor_17_DBCD, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def TMA(daily_close, N):
    
    close = daily_close.rolling(window=N, min_periods=1).mean()
    
    tma = close.rolling(window=N, min_periods=1).mean()
    
    return tma

factor_18_TMA = TMA(daily_close, 5)
signal = converter(factor_18_TMA, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def TSI(daily_close, N1, N2):
    
    diff = daily_close.diff()
    abs_diff = diff.abs()
    
    EMA_Diff = diff.ewm(span=N1, adjust=False).mean()
    EMA_Abs_Diff = abs_diff.ewm(span=N1, adjust=False).mean()
    # 对 EMA(价格差值) 和 EMA(绝对价格差值) 计算双重 EMA
    TSI_EMA_Diff = EMA_Diff.ewm(span=N2, adjust=False).mean()
    TSI_EMA_Abs_Diff = EMA_Abs_Diff.ewm(span=N2, adjust=False).mean()
    # 计算 TSI 值
    TSI = TSI_EMA_Diff / TSI_EMA_Abs_Diff * 100
   
    return TSI

factor_19_TSI = TSI(daily_close, 3, 4)
signal = converter(factor_19_TSI, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def DEMA(daily_close, N):
    
    ema1 = daily_close.ewm(span=N, adjust=False).mean()
    ema2 = ema1.ewm(span=N, adjust=False).mean()
    
    return 2*ema1 - ema2

factor_20_DEMA = DEMA(daily_close, 5)
signal = converter(factor_20_DEMA, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def ROC_CHANGE(daily_close, short_window, long_window):
    # ROC_CHANGE    
    ROC1 = daily_close / daily_close.shift(short_window) - 1
    ROC2 = daily_close / daily_close.shift(long_window) - 1        
    signal_source= ROC1 - ROC2
    
    return signal_source

factor_21_ROC_CHANGE = ROC_CHANGE(daily_close, 40, 60)
signal = converter(factor_21_ROC_CHANGE, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def EFFICIENCY(daily_close, N1, N2):
    
    def effi(daily_close, N):
        r_N = daily_close / daily_close.shift(N) - 1
        effi = pd.DataFrame(index=daily_close.index, columns=daily_close.columns)
        
        for col in daily_close.columns:
            stock_close = daily_close[col]
            effi_values = [None] * len(stock_close)
            # 从 N 开始计算 EMA
            for t in range(N-1, len(stock_close)):
                efmom = sum(stock_close.iloc[t-i] / stock_close.iloc[t-i-1] - 1 for i in range(N)) / N
                effi_values[t] = r_N[col].iloc[t] / efmom if efmom != 0 else None            
            effi[col] = effi_values
        return effi
    
    effi_N1 = effi(daily_close, N1)
    effi_N2 = effi(daily_close, N2)
    
    return effi_N1 - effi_N2

factor_22_EFFICIENCY = EFFICIENCY(daily_close, 2, 4)
signal = converter(factor_22_EFFICIENCY, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def INVVOL(daily_close, N1, N2):
    # 计算收益率
    returns = daily_close / daily_close.shift(1) - 1
    
    def calculate_invv(N, returns):
        # mean_returns = returns.rolling(window=N).mean() # 计算 N 天均值
        std_dev = returns.rolling(window=N).std() # 计算 N 天标准差
        # 计算 INVVOL
        invvol = 1 / std_dev
        
        return invvol
    # 计算短期和长期 INVVOL
    invvol_N1 = calculate_invv(N1, returns)
    invvol_N2 = calculate_invv(N2, returns)
    # 计算 INVVOL 的差值
    invvol_diff = invvol_N1 - invvol_N2
    
    return invvol_diff
    
factor_23_INVVOL = INVVOL(daily_close, 2, 4)
signal = converter(factor_23_INVVOL, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def AVG_LINE(daily_close, short_window, long_window):
    # 计算短期和长期移动平均
    avg_line_N1 = daily_close.rolling(window=short_window).mean()
    avg_line_N2 = daily_close.rolling(window=long_window).mean()
    
    return avg_line_N1 - avg_line_N2

factor_24_AVG_LINE = AVG_LINE(daily_close, 2, 4)
signal = converter(factor_24_AVG_LINE, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def EXPMA(daily_close, short_window, long_window):
    # Calculate short-term and long-term EMA
    expma_N1 = daily_close.ewm(span=short_window, adjust=False).mean()
    expma_N2 = daily_close.ewm(span=long_window, adjust=False).mean()
   
    return expma_N1 - expma_N2

factor_25_EXPMA = EXPMA(daily_close, 2, 4)
signal = converter(factor_25_EXPMA, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def MOM(daily_close, short_window, long_window):
    # 计算收益率
    returns = daily_close / daily_close.shift(1) - 1    
    # 计算短期和长期动量
    mom_N1 = returns.rolling(window=short_window).mean()
    mom_N2 = returns.rolling(window=long_window).mean()
    # 返回短期和长期动量之差
    return mom_N1 - mom_N2

factor_26_MOM = MOM(daily_close, 2, 4)
signal = converter(factor_26_MOM, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def SHARP_MOM(daily_close, short_window, long_window):
    # 计算收益率
    returns = daily_close / daily_close.shift(1) - 1
    
    def calculate_mean_std(N, returns):
        # 计算 N 天收益率均值
        mean_returns = returns.rolling(window=N).mean()
        # 计算 N 天收益率标准差
        std_dev = returns.rolling(window=N).std()
        
        return mean_returns, std_dev
    
    def calculate_sharp_mom(N, returns):
        mean_returns, std_dev = calculate_mean_std(N, returns)
        # 计算夏普动量# 注意：需要避免标准差为0的情况，防止除以0的错误
        sharp_mom = mean_returns / std_dev.replace(0, np.nan)  # 使用 NaN 替代除以零的情况
        
        return sharp_mom
    # 计算短期和长期夏普动量
    sharp_mom_N1 = calculate_sharp_mom(short_window, returns)
    sharp_mom_N2 = calculate_sharp_mom(long_window, returns)
    # 返回短期和长期夏普动量
    return sharp_mom_N1 - sharp_mom_N2

factor_27_SHARP_MOM = SHARP_MOM(daily_close, 2, 4)
signal = converter(factor_27_SHARP_MOM, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def HULLMA(daily_close, short_window, long_window):    
    
    nema_short = 2*daily_close.ewm(span=int(short_window/2), adjust=False).mean()  - \
                 daily_close.ewm(span=short_window, adjust=False).mean()
    nema_long = 2*daily_close.ewm(span=int(long_window/2), adjust=False).mean()  - \
                daily_close.ewm(span=long_window, adjust=False).mean()
    hullma_short = nema_short.ewm(span=int(short_window ** 0.5), adjust=False).mean()
    hullma_long = nema_long.ewm(span=int(long_window ** 0.5), adjust=False).mean()                
    signal_source = hullma_short - hullma_long
    
    return signal_source    

factor_28_HULLMA = HULLMA(daily_close, 20, 60)
signal = converter(factor_28_HULLMA, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def MACD(daily_close, N1, N2, N3):
    
    ema_N1 = daily_close.ewm(span=N1, adjust=False).mean()
    ema_N2 = daily_close.ewm(span=N2, adjust=False).mean()
    DIF = ema_N1 - ema_N2
    DEA = DIF.ewm(span=N3, adjust=False).mean()
    MACD = DIF - DEA
    
    return MACD

factor_29_MACD = MACD(daily_close, 3, 4, 5)
signal = converter(factor_29_MACD, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def ZLMACD(daily_close, N1, N2):
    
    ema_N1 = daily_close.ewm(span=N1, adjust=False).mean()
    ema_N2 = daily_close.ewm(span=N2, adjust=False).mean()
    ema_N1_ = ema_N1.ewm(span=N1, adjust=False).mean()
    ema_N2_ = ema_N2.ewm(span=N2, adjust=False).mean()
    zlmacd = (2*ema_N1-ema_N1_)-(2*ema_N2-ema_N2_)
    
    return zlmacd

factor_30_ZLMACD = ZLMACD(daily_close, 2, 4)
signal = converter(factor_30_ZLMACD, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def PPO(daily_close, N1, N2, N3):
    
    ema_N1 = daily_close.ewm(span=N1, adjust=False).mean()
    ema_N2 = daily_close.ewm(span=N2, adjust=False).mean()
    ppo = ema_N1 / ema_N2 - 1
    emappo = ppo.ewm(span=N3, adjust=False).mean()
    
    return ppo - emappo

factor_31_PPO = PPO(daily_close, 3, 4, 5)
signal = converter(factor_31_PPO, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def OSC(daily_close, N, M):
    
    osc = daily_close - daily_close.rolling(window=N).mean()
    oscma = osc.rolling(window=M).mean()
    
    return osc - oscma

factor_32_OSC = OSC(daily_close, 3, 3)
signal = converter(factor_32_OSC, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def PMO(daily_close, N1, N2, N3):
    
    roc = daily_close.pct_change() * 100

    def DMA(daily_close, X):
        dma = X*daily_close + (1-X)*daily_close.shift(1)
        return dma
    
    dmaroc = 10 * DMA(roc, 2/N1)
    pmo = DMA(dmaroc, 2/N2)
    dmapmo = DMA(pmo, 2/(N3+1))
    
    return pmo - dmapmo

factor_33_PMO = PMO(daily_close, 3, 4, 5)
signal = converter(factor_33_PMO, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def BIAS36(daily_close): # 公式有笔误∑上N-1应为M-1
    N1 = 3
    N2 = 6
    M = 6
    bias36 = daily_close.rolling(window=N1).mean() - daily_close.rolling(window=N2).mean()
    mabias36 = bias36.rolling(window=M).mean()
    
    return mabias36

factor_34_BIAS36 = BIAS36(daily_close)
signal = converter(factor_34_BIAS36, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def CMO(daily_close, N):
    
    diff = daily_close.diff()
    
    up_diff = diff.clip(lower=0)
    down_diff = -diff.clip(upper=0)
    
    sum_up_diff = up_diff.rolling(window=N).sum()
    sum_down_diff = down_diff.rolling(window=N).sum()
    
    cmo = (sum_up_diff + sum_down_diff) / (sum_up_diff - sum_down_diff) * 100
    
    return cmo

factor_35_CMO = CMO(daily_close, 5)
signal = converter(factor_35_CMO, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))    

def PSY(daily_close, N):
    
    diff = daily_close.diff()
    indexdiff = (diff.clip(lower=0) / diff.abs()).fillna(0)
    
    psy = indexdiff.rolling(window=N).mean() * 100
    
    return psy

factor_36_PSY = PSY(daily_close, 5)
signal = converter(factor_36_PSY, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def UP2DOWN(daily_close, N, M):
    
    ret = daily_close.pct_change()
    
    up_ret = ret.clip(lower=0)
    up_ret_ = (up_ret / ret.abs()).fillna(0)
    down_ret = ret.clip(upper=0)
    down_ret_ = (down_ret / ret.abs()).fillna(0)
    
    up_r_N = up_ret.rolling(window=N).mean()
    down_r_N = down_ret.rolling(window=N).mean()
    up_r_N_ = up_ret_.rolling(window=N).mean()
    down_r_N_ = down_ret_.rolling(window=N).mean()
    # 计算正收益均值
    up_r_N_mean = up_r_N / up_r_N_
    # 计算负收益均值
    down_r_N_mean = down_r_N / down_r_N_
    # 计算 UP2DOWN 指标
    up2down = (up_r_N_mean / down_r_N_mean).fillna(0)

    return up2down - M

factor_37_UP2DOWN = UP2DOWN(daily_close, 5, 0)
signal = converter(factor_37_UP2DOWN, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def RSIH(daily_close, N1, N2):
    
    diff = daily_close.diff()
    up_diff = diff.clip(lower=0)
    down_diff = diff.clip(upper=0)
    
    sma_up_diff = up_diff.rolling(window=N1, min_periods=1).mean()
    sma_updown_diff = (up_diff-down_diff).rolling(window=N1, min_periods=1).mean()
    
    rsi = sma_up_diff / sma_updown_diff * 100
    
    rsih = rsi - rsi.ewm(span=N2, adjust=False).mean()
    
    return rsih

factor_38_RSIH = RSIH(daily_close, 3, 4)
signal = converter(factor_38_RSIH, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def TII(daily_close, N, M):
    
    dev = daily_close - daily_close.rolling(window=N).mean()
    
    up_dev = dev.clip(lower=0)
    down_dev = dev.clip(upper=0)
    
    P = N / 2 + 1
    sum_up_dev = up_dev.rolling(window=int(P)).sum()
    sum_down_dev = down_dev.rolling(window=int(P)).sum()
 
    tii = sum_up_dev / (sum_up_dev - sum_down_dev) 
    
    return tii - M

factor_39_TII = TII(daily_close, 5, 0)
signal = converter(factor_39_TII, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def THRES_AVG(daily_close, N, M):
    
    dev = daily_close - daily_close.rolling(window=N).mean()
    
    index_dev = (dev / dev.abs()).clip(lower=0)
    
    percent = index_dev.rolling(window=N).mean()
    
    return percent - M

factor_40_THRES_AVG = THRES_AVG(daily_close, 5, 0)
signal = converter(factor_40_THRES_AVG, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

def MASS(daily_close, N, M):
    
    moving_avg = daily_close.rolling(window=N).mean()

    index_mass = (daily_close - moving_avg).div((daily_close - moving_avg).abs()).clip(lower=0)
    
    mass = index_mass.rolling(window=N).mean()
    
    return mass - M

factor_41_MASS = MASS(daily_close, 5, 0)
signal = converter(factor_41_MASS, 10)
return_list.append(adjuster_for_equal_weights(daily_close, signal))

temp = pd.DataFrame(index=range(1, 42), data=return_list, columns=['return'])
print(temp.sort_values(by='return', ascending=False))