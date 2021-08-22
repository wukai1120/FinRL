import sys
sys.path.append(".")

import pandas as pd
import numpy as np
import datetime

#from finrl.neo_finrl.data_processors.processor_ccxt import CCXTEngineer
from finrl.neo_finrl.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from btc_env import BtcEnv

start_date = '2021-07-01'
split_date = '2021-08-01'
end_date = '2021-08-20'

information_cols = ['x_2h', 'min_p_2h', 'max_p_2h', 'x_8h', 'min_p_8h', 'max_p_8h', 'x_24h', 'min_p_24h', 'max_p_24h']

#CE = CCXTEngineer()
#df = CE.data_fetch(start = '20210201 00:00:00', end = '20210820 00:00:00',pair_list = ['BTC/USDT','ETH/USDT'], period = '5m')
#df.to_csv('btc.csv', index=False)

df = pd.read_csv('btc.csv')
df = df[(df['date'] >= start_date) & (df['date'] < end_date)]
df = df.reset_index(drop=True)
df = df[df['tic'] == 'BTC/USDT']
print(df.shape)

df['open'] = df['open'].astype(int)
df['close'] = df['close'].astype(int)
df['high'] = df['high'].astype(int)
df['low'] = df['low'].astype(int)
df['volume'] = df['volume'].astype(int)
df['change'] = df['close'] - df['open']
df['x_2h'] = 0
df['min_p_2h'] = 0
df['max_p_2h'] = 0
df['x_8h'] = 0
df['min_p_8h'] = 0
df['max_p_8h'] = 0
df['x_24h'] = 0
df['min_p_24h'] = 0
df['max_p_24h'] = 0
print(df.head())
print(len(df))
for i in range(len(df)):
    if i < 24 * 12:
        continue
    # 计算2小时内最高点、最低点、当前价格在区间的高度
    max = df[i-23:i+1]['high'].max()
    idxmax = df[i-23:i+1]['high'].idxmax()
    min = df[i-23:i+1]['low'].min()
    idxmin = df[i-23:i+1]['low'].idxmin()
    df.loc[i,'x_2h'] = round((df.loc[i,'close'] - min) / (max - min),2)
    df.loc[i,'max_p_2h'] = (i - idxmax)
    df.loc[i,'min_p_2h'] = (i - idxmin)
    # 计算8小时内最高点、最低点、当前价格在区间的高度
    max = df[i-95:i+1]['high'].max()
    idxmax = df[i-95:i+1]['high'].idxmax()
    min = df[i-95:i+1]['low'].min()
    idxmin = df[i-95:i+1]['low'].idxmin()
    df.loc[i,'x_8h'] = round((df.loc[i,'close'] - min) / (max - min),2)
    df.loc[i,'max_p_8h'] = (i - idxmax)
    df.loc[i,'min_p_8h'] = (i - idxmin)
    # 计算24小时内最高点、最低点、当前价格在区间的高度
    max = df[i-287:i+1]['high'].max()
    idxmax = df[i-287:i+1]['high'].idxmax()
    min = df[i-287:i+1]['low'].min()
    idxmin = df[i-287:i+1]['low'].idxmin()
    df.loc[i,'x_24h'] = round((df.loc[i,'close'] - min) / (max - min),2)
    df.loc[i,'max_p_24h'] = (i - idxmax)
    df.loc[i,'min_p_24h'] = (i - idxmin)
    
df = df[288:]
df.to_csv('btc_proc.csv', index=False)