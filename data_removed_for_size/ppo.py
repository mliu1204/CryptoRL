
"""
This is the MLP-LSTM environment
"""


import gymnasium
from gymnasium import spaces
import json
import numpy as np
import time
import random
from gymnasium.utils import seeding


from stable_baselines3.common.evaluation import evaluate_policy


class CustomEnv(gymnasium.Env):
    """
    actions: 0 is buy, 1 is sell, 2 is hold
    """
    BUY = 0
    SELL = 1
    HOLD = 2
    metadata = {'render.modes': ['console']}

    def __init__(self, data_file_name, quantity_to_buy = 1, steps_back = 100, incorrect_action_fee = 0.0):
        super(CustomEnv, self).__init__()
        self.info = {}
        self.steps_back = steps_back # past transaction context
        self.quantity_to_buy = quantity_to_buy # can only buy once, this is how much it gets to buy
        self.incorrect_action_fee = incorrect_action_fee # when it buys again or sells with nothing
        with open(data_file_name, 'r') as f:
            self.all_coins = json.load(f)

        self.all_coins = self.all_coins['coins']
        self.action_space = spaces.Discrete(3)
    
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(6,), dtype=np.float32)
        
        self.np_random = None
        self.seed()

    def seed(self, seed=None):
        """
        Set the seed for the environment's random number generator.
        
        Args:
            seed (int, optional): Random seed. If None, a random seed is chosen.
            
        Returns:
            list: The seed that was actually used
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        # how much were we worth
        old_portfolio_value = self.coins_held *  self.last_price + self.net_solana

        # Agent acts on environment 
        if action == self.BUY:
            self.buy(self.last_transaction)
            self.info["log"] = f"Tried to buy the stuff at {self.transaction_index}"
                
        elif action == self.SELL:
            self.sell(self.last_transaction)
            self.info["log"] = f"tried to sell the stuff at {self.transaction_index}"

        self.info["action"] = f"did this action {action}"
        
        cur_transaction = self.all_transactions[self.transaction_index]

        # if we are at the end of the future transactions, truncate and sell all
        if self.transaction_index + 1 == len(self.all_transactions):
            self.truncated = True
            self.info["log"] = f"truncated the stuff at {self.transaction_index}"
            if self.coins_held > 0:
                self.sell(cur_transaction)
                self.net_solana -= 4.5

        # environment updates 
        self.observation = self.clean_transaction(cur_transaction)
        cur_price = cur_transaction['price']
        self.last_price = cur_price
        self.last_transaction = cur_transaction
        self.transaction_index += 1

        # how much do we worth now
        new_portfolio_value = self.coins_held * cur_price + self.net_solana
        reward = new_portfolio_value - old_portfolio_value

        return self.observation, reward, self.terminated, self.truncated, self.info
  
    def reset(self, seed = None, options = None):
        if seed is not None:
            self.seed(seed)

        self.terminated = False
        self.truncated = False
        self.coin = self.np_random.choice(self.all_coins)
        self.coins_held = 0 # 
        self.net_solana = 0 # after fees and everything, think of it as wallet
        self.transaction_index = 1 # next transaction I haven't seen

        self.info = {"log": "have not bought yet", 'action': "no prev action"}
        self.all_transactions = self.coin['pastTrades'] + self.coin['futureTrades']
        self.past_trades_count = len(self.coin['pastTrades'])

        self.observation = self.clean_transaction(self.all_transactions[0])
        
        self.last_price = self.all_transactions[0]['price']
        self.last_transaction = self.all_transactions[0]

        return (self.observation, self.info)
    
    def render(self, mode='human'):
        print("---------------")
        print(f"Transaction {self.transaction_index}")
        print(f"Our current holdings: {self.coins_held}")
        print(f"The most recent price: {self.last_price}")
        print(f"The amount we spent, including fees: {self.net_solana}")
        print(f"This is the info log: {self.info['log']}")
        print(f"This is the info action: {self.info['action']}")
        print(f"Can trade {self.can_trade()}")
        return 0
    def close (self):
        return 0
    
    def fees(self, type, sol_quantity_bought):
        return 0
    #-sol_quantity_bought

    def buy(self, last_transaction):
        fees = 0
        if self.coins_held > 0 or not self.can_trade():
            fees = self.incorrect_action_fee
        else:        
            fees = self.fees("buy", self.quantity_to_buy)
            cur_price = last_transaction['price']
            coins_recieved = self.quantity_to_buy/cur_price
            self.coins_held += coins_recieved
            self.net_solana -= self.quantity_to_buy
        self.net_solana -= fees
    
    def sell(self, last_transaction):
        fees = 0
        sol_received = 0
        if self.coins_held == 0:
            fees = self.incorrect_action_fee
        else:
            self.terminated = True
            fees = self.fees("sell", self.coins_held)
            cur_price = last_transaction['price']
            sol_received = self.coins_held * cur_price
            self.coins_held = 0
        self.net_solana += sol_received - fees


    def can_trade(self):
        return self.transaction_index >= self.past_trades_count
        

    def clean_transaction(self, transaction):
        obs = np.zeros(6, dtype=np.float32)
        # how much the transaction paid in sol
        obs[0] = float(transaction['volumeSol']) 

        # price at the time
        obs[1] = float(transaction['price'])

        # time of transaction
        obs[2] = float(transaction['time'])

        # how many prev trivial 
        obs[3] = 0 if "prevTrivial" not in transaction.keys() else transaction['prevTrivial']

        # bought or not
        bought = 1 if self.coins_held > 0 else 0
        obs[4] = bought

        # can or cannot trade (not past data)
        obs[5] = self.can_trade()
        return obs

