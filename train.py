import sys
sys.path.append(".")

import pandas as pd
import numpy as np

from finrl.neo_finrl.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from btc_env import BtcEnv

start_date = '2021-07-01'
end_date = '2021-08-01'

information_cols = ['x_2h', 'min_p_2h', 'max_p_2h', 'x_8h', 'min_p_8h', 'max_p_8h', 'x_24h', 'min_p_24h', 'max_p_24h']

df = pd.read_csv('btc_proc.csv')
#数据拆分
train = df[(df['date'] >= start_date) & (df['date'] < end_date)]
train = train.reset_index(drop=True)
print(train.head())
print(len(train))

#开始训练
e_train_gym = BtcEnv(df = train, 
    initial_amount=1e4,
    transaction_cost_pct=2e-4,
    daily_information_cols = information_cols,
    print_verbosity = 500)

env_train, _ = e_train_gym.get_sb_env()
agent = DRLAgent(env = env_train)

#from torch.nn import Softsign, ReLU
ppo_params = { 'n_steps': 2048, 'ent_coef': 0.01,  'learning_rate': 0.00025, 'batch_size': 512, 'gamma': 0.95 }

policy_kwargs = {
#     "activation_fn": ReLU,
    "net_arch": [1024, 1024, 1024], 
#     "squash_output": True
}

model = agent.get_model("ppo", model_kwargs = ppo_params, policy_kwargs = policy_kwargs, verbose = 0)

model.learn(total_timesteps = 200000, 
  log_interval = 1, tb_log_name = 'btc_1024_5_more_ooc_penalty',
  reset_num_timesteps = True)

model.save("quicksave_btc_1.model")