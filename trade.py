import sys
sys.path.append(".")

import pandas as pd
import numpy as np

from finrl.neo_finrl.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from btc_env import BtcEnv

start_date = '2021-08-01'
end_date = '2021-08-20'

information_cols = ['x_2h', 'min_p_2h', 'max_p_2h', 'x_8h', 'min_p_8h', 'max_p_8h', 'x_24h', 'min_p_24h', 'max_p_24h']

df = pd.read_csv('btc_proc.csv')

trade = df[(df['date'] >= start_date) & (df['date'] < end_date)]
trade = trade.reset_index(drop=True)
print(trade.head())
print(len(trade))

from stable_baselines3 import PPO
model = PPO.load('quicksave_btc_1.model')
e_trade_gym = BtcEnv(df = trade, 
    initial_amount=1e4,
    transaction_cost_pct=2e-4,
    daily_information_cols = information_cols,
    print_verbosity = 100)

state = e_trade_gym.reset()
done = False
total_reword = 0
while not done:
    action = model.predict(state)[0]
    state, reward, done, _ = e_trade_gym.step(action)
    total_reword+= reward
print((total_reword))