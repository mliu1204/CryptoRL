"""
This is LSTM future prediction + MLP PPO environment V1

"""

import gymnasium
from gymnasium import spaces
import numpy as np
import random
from collections import deque
import json
from enum import IntEnum

class ActIdx(IntEnum):
    BUY = 0
    SELL = 1
    HOLD = 2
class ObsIdx(IntEnum):
    TIME = 0
    PRICE = 1
    BUY_STREAK_AVG = 2
    SELL_STREAK_AVG = 3
    LIFE_TRANSACTIONS = 4
    LIFE_TRIV_TRANSACTIONS = 5
    LIFE_MAX_PRICE = 6
    RECENT_TRANSACTIONS = 7
    RECENT_TRIV_TRANSACTIONS = 8
    RECENT_MAX_PRICE = 9
    LSTM1m = 10
    LSTM5m = 11
    LSTM10m = 12
    HOLDING_COINS = 13
class ObsScaling(IntEnum):
    TIME = 1.0e-03
    PRICE = 1.0e08
    BUY_STREAK_AVG = 1
    SELL_STREAK_AVG = 1
    LIFE_TRANSACTIONS = 1
    LIFE_TRIV_TRANSACTIONS = 1
    LIFE_MAX_PRICE = 1.0e08
    RECENT_TRANSACTIONS = 1
    RECENT_TRIV_TRANSACTIONS = 1
    RECENT_MAX_PRICE = 1.0e08
    LSTM1m = 1.0e08
    LSTM5m = 1.0e08
    LSTM10m = 1.0e08
    HOLDING_COINS = 1.0e-07

class PPOEnv(gymnasium.Env):
    # action and observation index values

    def __init__(self, data_file_name, quantity_to_buy = 1):
        super(PPOEnv, self).__init__()
        
        self.action_space = spaces.Discrete(len(ActIdx))

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(len(ObsIdx),), dtype=np.float32)

        # load all coins
        with open(data_file_name, 'r') as f:
            self.all_coins = json.load(f)

        # Current observation (trade data + holding)
        self.current_obs = None
        
        # Index of next observation
        self.next_obs_index = None

        # Last reward (for rendering)
        self.last_reward = None

        # Last action taken (for rendering)
        self.last_action = None

        self.coins_held = None
        self.net_solana = None
        self.last_portfolio_value = None
        self.quantity_to_buy = quantity_to_buy
        

    def seed(self, seed: int = None):
        """
        Seed both numpy and random for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return [seed]

    def reset(self, *, seed: int = 1):
        """
        Reset environment state and history.

        Args:
            seed (int, optional): random seed for reproducibility.

        Returns:
            observation (np.ndarray): array([0 or 1], dtype=int8)
            info (dict): empty dict
        """
        self.seed(seed)

        self.current_obs = np.zeros(len(ObsIdx), dtype=np.float32)
        
        self.coin = random.choice(self.all_coins['coins'])
        self.transactions = self.coin['pastTrades'] + self.coin['futureTrades']
        
        latest_past_transaction = self.coin['pastTrades'][-1]
        self.update_obs(latest_past_transaction, hold=0)
        

        self.last_reward = None
        self.last_action = None
        self.last_portfolio_value = 0
        self.next_obs_index = len(self.coin['pastTrades'])
        self.coins_held = 0
        self.net_solana = 0

        return np.array([self.current_obs], dtype=np.float32), {}

    def step(self, action: int):
        """
        Current order of events: 
            1. calculate old portfolio value
            2. take action (buy sell hold)
            3. update observation (aka change which transaction we are on)
            4. calculate new portfolio value
            5. calculate reward and set old = new port value
            
        Take an action (0 or 1) and return next transition.
        """
        try:
            act_enum = ActIdx(action)
        except ValueError:
            raise ValueError(f"Invalid action: {action}.  Must be 0 (BUY) or 1 (SELL).")

        # Store action for render()
        self.last_action = action

        # take action and update observations
        new_hold = self.current_obs[ObsIdx.HOLDING_COINS]
        success_action = True
        if action == ActIdx.BUY:
            new_hold, success_action = self.buy()
        elif action == ActIdx.SELL:  
            new_hold, success_action = self.sell()
        self.update_obs(transaction=self.transactions[self.next_obs_index], hold=new_hold)
        self.next_obs_index += 1
        
        # calculate reward and update last_portfolio_value
        new_portfolio_value = self.coins_held *  self.current_obs[ObsIdx.PRICE] + self.net_solana
        reward = new_portfolio_value-self.last_portfolio_value
        self.last_portfolio_value = new_portfolio_value

        # Store reward for render()
        self.last_reward = reward

        truncated = self.next_obs_index == len(self.transactions)
        if truncated:
            if self.coins_held != 0:
                new_hold, success_action = self.sell()
                self.net_solana -= self.fees(type = "hoarder")
                new_portfolio_value = self.coins_held *  self.current_obs[ObsIdx.PRICE] + self.net_solana
                reward = new_portfolio_value-self.last_portfolio_value
                self.last_portfolio_value = new_portfolio_value
            elif self.coins_held == 0:
                self.net_solana -= self.fees(type = "stupid")
                new_portfolio_value = self.coins_held *  self.current_obs[ObsIdx.PRICE] + self.net_solana
                reward = new_portfolio_value-self.last_portfolio_value
                self.last_portfolio_value = new_portfolio_value
        terminated = success_action and action == ActIdx.SELL 
            
        return np.array([self.current_obs], dtype=np.float32), reward, terminated, truncated, {}

    def render(self):
        """
        Print out a simple summary of the last transition:
          - step index
          - action name
          - current price
          - coins held
          - cash (net_solana)
          - portfolio value
          - last reward
        """
        # If no step has been taken yet, just show initial state:
        if self.next_obs_index is None or self.next_obs_index == len(self.coin['pastTrades']):
            price = self.current_obs[ObsIdx.PRICE]
            print(f"Init ▶ Price={price:.3f} │ Holdings={self.coins_held} │ Cash={self.net_solana:.3f}")
            return

        # Compute the index of the step that just happened:
        step_idx = self.next_obs_index - 1
        # Map action integer to its name (BUY, SELL, HOLD)
        action_name = ActIdx(self.last_action).name if self.last_action is not None else "None"
        price = self.current_obs[ObsIdx.PRICE]
        coins = int(self.coins_held)
        cash = self.net_solana
        portfolio = coins * price + cash
        reward = self.last_reward if self.last_reward is not None else 0.0

        print(
            f"Step {step_idx} ▶ Action={action_name} │ "
            f"Price={price} │ Coins_held={coins} │ Net_solana={cash} │ "
            f"Value={portfolio} │ Reward={reward}"
        )


# CUSTOM HELPER FUNCTIONS
    def fees(self, type, volume_sol = 0.0):
        fee = 0
        if type == "transaction":
            fee = volume_sol / 100
        elif type == "hoarder": # only buy and does not sell
            fee = 0.01
        elif type == "stupid": # does not buy or sell the entire time
            fee = 0.01
        elif type == "bad_transaction":
            fee = volume_sol / 100 * 5
        return fee
    
    def buy(self):
        if self.coins_held != 0:
            self.net_solana -= self.fees(type="bad_transaction")
            return self.coins_held, False
        else:
            self.coins_held += self.quantity_to_buy 
            spent_solana = self.quantity_to_buy * self.current_obs[ObsIdx.PRICE]
            fee = self.fees(type="transaction", volume_sol = spent_solana)
            self.net_solana -= (fee + spent_solana)
            return self.coins_held, True
        
    def sell(self):
        if self.coins_held == 0:
            self.net_solana -= self.fees(type="bad_transaction")
            return self.coins_held, False
        else:
            earned_solana = self.coins_held * self.current_obs[ObsIdx.PRICE]
            fee = self.fees(type="transaction", volume_sol = earned_solana)
            self.net_solana += earned_solana - fee
            self.coins_held = 0 
            return self.coins_held, True

    def update_obs(self, transaction, hold):
        self.current_obs[ObsIdx.TIME]                    = float(transaction['time'] * ObsScaling.TIME)
        self.current_obs[ObsIdx.PRICE]                   = float(transaction['price']* ObsScaling.PRICE)
        self.current_obs[ObsIdx.BUY_STREAK_AVG]          = float(transaction['avg_buy_streak']* ObsScaling.BUY_STREAK_AVG)
        self.current_obs[ObsIdx.SELL_STREAK_AVG]         = float(transaction['avg_sell_streak']* ObsScaling.SELL_STREAK_AVG)
        self.current_obs[ObsIdx.LIFE_TRANSACTIONS]       = float(transaction['life_transactions']* ObsScaling.LIFE_TRANSACTIONS)
        self.current_obs[ObsIdx.LIFE_TRIV_TRANSACTIONS]  = float(transaction['life_trivial_transactions']* ObsScaling.LIFE_TRIV_TRANSACTIONS)
        self.current_obs[ObsIdx.LIFE_MAX_PRICE]          = float(transaction['life_max_price']* ObsScaling.LIFE_MAX_PRICE)
        self.current_obs[ObsIdx.RECENT_TRANSACTIONS]     = float(transaction['recent_transactions']* ObsScaling.RECENT_TRANSACTIONS)
        self.current_obs[ObsIdx.RECENT_TRIV_TRANSACTIONS]= float(transaction['recent_trivial_transactions']* ObsScaling.RECENT_TRIV_TRANSACTIONS)
        self.current_obs[ObsIdx.RECENT_MAX_PRICE]        = float(transaction['recent_max_price']* ObsScaling.RECENT_MAX_PRICE)
        self.current_obs[ObsIdx.LSTM1m]                  = float(transaction['LSTM1m']* ObsScaling.LSTM1m)
        self.current_obs[ObsIdx.LSTM5m]                  = float(transaction['LSTM5m']* ObsScaling.LSTM5m)
        self.current_obs[ObsIdx.LSTM10m]                 = float(transaction['LSTM10m']* ObsScaling.LSTM10m)
        self.current_obs[ObsIdx.HOLDING_COINS]           = float(hold* ObsScaling.HOLDING_COINS)     

    def close(self):
        pass
