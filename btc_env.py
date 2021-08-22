import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

class BtcEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        transaction_cost_pct=3e-4,
        print_verbosity=10,
        reward_scaling=1e-4,
        initial_amount=1e4,
        leverage=20,#20倍杠杆
        daily_information_cols=["open", "close", "high", "low", "volume"]
    ):
        self.df = df
        self.initial_amount = initial_amount
        self.print_verbosity = print_verbosity
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.leverage = leverage
        self.daily_information_cols = daily_information_cols
        self.state_space = len(self.daily_information_cols)
        self.action_space = spaces.Box(low=-0.9, high=0.9, shape=(1,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))
        self.episode = -1  # initialize so we can call reset
        self.episode_history = []
        self.printed_header = False

    def reset(self):
        self.date_index = 0
        self.episode += 1
        self.state_memory = []
        self.account_information = {
            "cash": [],
            "holding": [],
            "margin": [],
            "total_assets": [],
            'reward': []
        }
        self.state_memory.append(np.array(self.get_date_vector(self.date_index)))
        return [0 for _ in range(self.state_space)]

    def get_date_vector(self, index, cols=None):
        if cols is None:
            cols = self.daily_information_cols
        return self.df.loc[index, cols].tolist()
    
    def log_step(self, reason, terminal_reward=None):
        if terminal_reward is None:
            terminal_reward = self.account_information['reward'][-1]
        cash_pct = self.account_information['cash'][-1] / self.account_information['total_assets'][-1]
        rec = [self.episode, self.date_index, reason, round(self.account_information['total_assets'][-1],2), round(terminal_reward,2)]

        self.episode_history.append(rec)
        print(self.template.format(*rec))

    def step(self, actions):
        if self.printed_header is False:
            self.template = "{0:8}|{1:10}|{2:15}|{3:7}|{4:10}" # column widths: 8, 10, 15, 7
            print(self.template.format("EPISODE", "STEPS", "TERMINAL_REASON", "TOT_ASSETS", "REWARD"))
            self.printed_header = True

        def return_terminal(reason='Last Date', extra_reward=0):
            state = self.state_memory[-1]
            reward = 0
            reward += extra_reward
            self.log_step(reason = reason, terminal_reward= reward)
            reward = reward * self.reward_scaling
            return state, reward, True, {}

        if (self.date_index + 1) % self.print_verbosity == 0:
            self.log_step(reason = 'update')

        if self.date_index == len(self.df) - 1:
            terminal_reward = self.account_information['total_assets'][-1] - self.initial_amount
            return return_terminal(extra_reward = terminal_reward)
        else:
            price = self.get_date_vector(self.date_index, cols='close')
            change = self.get_date_vector(self.date_index, cols='change')
            cash = self.initial_amount
            holding = 0
            reward = 0
            margin = 0
            if self.date_index > 0:
                cash = self.account_information["cash"][-1]
                margin = self.account_information["margin"][-1]
                holding = self.account_information["holding"][-1]
                reward = holding * change

            cash += reward
            total_assets = cash + margin

            actions = actions[0] * total_assets * self.leverage / price #actions[-1,1]放大到买入数量

            num = actions - holding #实际需要买入或卖出数量
            margin = abs(actions) * price / self.leverage #保证金
            costs = abs(num) * price * self.transaction_cost_pct #手续费

            total_assets -= costs
            cash = total_assets - margin

            self.account_information["cash"].append(cash) #可用保证金
            self.account_information["holding"].append(actions) #持仓数量
            self.account_information["margin"].append(margin) #占用保证金
            self.account_information["total_assets"].append(total_assets) #总资产
            self.account_information['reward'].append(reward) #本次奖励

            if cash < 0 or total_assets < self.initial_amount * 0.5:
                return return_terminal(reason='broke', extra_reward=-self.initial_amount)

            self.date_index += 1
            state = self.get_date_vector(self.date_index)
            self.state_memory.append(state)
            reward = reward - costs
            reward = reward * self.reward_scaling
            return state, reward, False, {}

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
    
    def get_multiproc_env(self, n = 10):
        def get_self():
            return deepcopy(self)
        e = SubprocVecEnv([get_self for _ in range(n)], start_method = 'fork')
        obs = e.reset()
        return e, obs
