"""
added handcrafted features in addition to LSTM predictions for final features list
"""

import matplotlib
matplotlib.use('Agg') #headless so it doesn't show
import matplotlib.pyplot as plt
import datetime as dt
import time
import pytz 
from util import *
import os
import json
import math
import numpy as np
my_tz = pytz.timezone("America/Los_Angeles")


def make_json(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    return data

import os

def make_tiny(file_path, save_path, num_coins=5):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path!r} does not exist.")

    if os.path.getsize(file_path) == 0:
        raise ValueError(f"{file_path!r} is empty – cannot read JSON.")

    with open(file_path, 'r') as f:
        text = f.read().strip()
        if not text:
            raise ValueError(f"{file_path!r} contains only whitespace – invalid JSON.")
        data = json.loads(text)

    coins = data['coins']
    small = coins[:num_coins]
    data['coins'] = small
    make_json(save_path, data)

def update_streaks(vol,
                   current_buy_streak,
                   current_sell_streak,
                   buy_streak_lengths,
                   sell_streak_lengths):
    if vol > 0:
        if current_sell_streak > 0:
            sell_streak_lengths.append(current_sell_streak)
            current_sell_streak = 0
        current_buy_streak += 1
    else:
        if current_buy_streak > 0:
            buy_streak_lengths.append(current_buy_streak)
            current_buy_streak = 0
        current_sell_streak += 1

    if vol > 0:
        total_buy_streaks = buy_streak_lengths + [current_buy_streak]
        avg_buy = sum(total_buy_streaks) / len(total_buy_streaks)
    else:
        avg_buy = (
            sum(buy_streak_lengths) / len(buy_streak_lengths)
            if buy_streak_lengths else 0.0
        )

    if vol < 0:
        total_sell_streaks = sell_streak_lengths + [current_sell_streak]
        avg_sell = sum(total_sell_streaks) / len(total_sell_streaks)
    else:
        avg_sell = (
            sum(sell_streak_lengths) / len(sell_streak_lengths)
            if sell_streak_lengths else 0.0
        )

    return current_buy_streak, current_sell_streak, avg_buy, avg_sell
    
def add_LSTM_cheating_data(coin, future_times = [1,5,10]):
    trades = coin['pastTrades'] + coin['futureTrades']
    past_len = len(coin['pastTrades'])
    
    for i in range(past_len - 1, len(trades)):
        trade = trades[i]
        start_time = trade['time']
        for future_time in future_times:
            cur_trade_index = i
            while cur_trade_index < len(trades) and (trades[cur_trade_index]['time'] - start_time) < future_time*1000*60: 
                future_price = trades[cur_trade_index]['price']
                cur_trade_index += 1
            key = f"LSTM{future_time}m"
            trade[key] = future_price


def add_features(data_path, seconds_back = 10 * 60, LSTM_cheating = False, LSTM_future_times = [1,5,10]):
    root, tail = os.path.split(data_path)
    name, ext = os.path.splitext(tail)  
    save_data_file_path = os.path.join(root, f"{name}_added_features.json")
    with open (data_path) as f:
        data = json.load(f)
        coins = data['coins']
    for i, coin in enumerate(coins):
        # directly adds LSTM cheating data to future trades 
        if LSTM_cheating:
            add_LSTM_cheating_data(coin, LSTM_future_times)
            
        # initializing tracked features and all trades
        trades = coin['pastTrades'] + coin['futureTrades']

        # lifetime features
        life_transactions = 0
        life_trivial_transactions = 0
        life_max_price = 0.0

        # recent features
        recent_transactions = 0
        recent_trivial_transactions = 0
        recent_prices = []
        oldest_transaction = 0 # oldest transactions held in recent

        # streak features
        current_buy_streak = 0     # counts consecutive trades where volumeSol > 0
        current_sell_streak = 0    # counts consecutive trades where volumeSol < 0
        buy_streak_lengths = []    # list of completed buy streak lengths
        sell_streak_lengths = []   # list of completed sell streak lengths

        for trade in trades:
            vol = trade.get("volumeSol")
            if vol == 0:
                raise ValueError("volume of a trade cannot be zero")
            current_buy_streak, current_sell_streak, avg_buy, avg_sell = update_streaks(
                vol,
                current_buy_streak,
                current_sell_streak,
                buy_streak_lengths,
                sell_streak_lengths
            )
            
            trade["avg_buy_streak"] = avg_buy
            trade["avg_sell_streak"] = avg_sell

            # Update lifetime features
            life_transactions += 1
            life_trivial_transactions += 0 if 'prevTrivial' not in trade.keys() else trade['prevTrivial']
            life_max_price = max(life_max_price, trade['price'])

            # update recent features
            while (trade['time'] - trades[oldest_transaction]['time']) > seconds_back*1000:
                old_trade = trades[oldest_transaction]
                recent_transactions -= 1
                recent_trivial_transactions -= 0 if 'prevTrivial' not in old_trade.keys() else old_trade['prevTrivial']
                recent_prices.remove(old_trade["price"])
                oldest_transaction += 1
            recent_transactions += 1
            recent_trivial_transactions += 0 if 'prevTrivial' not in trade.keys() else trade['prevTrivial']
            recent_prices.append(trade['price'])
            recent_max_price = max(recent_prices)

            # update features in trade
            trade['life_transactions'] = life_transactions
            trade['life_trivial_transactions'] = life_trivial_transactions
            trade['life_max_price'] = life_max_price
            trade['recent_transactions'] = recent_transactions
            trade['recent_trivial_transactions'] = recent_trivial_transactions
            trade['recent_max_price'] = recent_max_price      
    make_json(save_data_file_path,data)      
    return

def get_y_change(coin, idx_include_up_to, time_in_future, index_of_last_y = None):
    if idx_include_up_to == -1 and len(coin['futureTrades']) == 0:
        return 1, None

    current_transaction = coin['futureTrades'][idx_include_up_to]
    time_to_measure_at = current_transaction['time'] + time_in_future
    cur_price = coin['futureTrades'][idx_include_up_to]['price']
    if idx_include_up_to == -1:
        cur_price = coin['pastTrades'][-1]['price']

    if index_of_last_y == "past_final_transaction":
        return coin['futureTrades'][-1]['price']/cur_price, "past_final_transaction"
    start_idx = idx_include_up_to
    
    if index_of_last_y is not None:
        start_idx = index_of_last_y
    
    for index, transaction in enumerate(coin['futureTrades'][start_idx:]):
        if transaction['time'] > time_to_measure_at:
            last_trans_before_time_idx = index - 1
            #print(transaction)
            #print(coin['futureTrades'][last_trans_before_time_idx])
            y_val = coin['futureTrades'][last_trans_before_time_idx + start_idx]['price']

            return y_val/cur_price, last_trans_before_time_idx
    return coin['futureTrades'][-1]['price']/cur_price, "past_final_transaction"




def get_np_data(file_name, trans_back_window = 100, time_forward = 10 * get_length_of_time('min'), 
                step = 10, scaling = None, features = ['volumeSol', "coin_count", "time", "price"]):
    with open(file_name, 'r') as f:
        data = json.load(f)
    y_points = []
    X = []
    num_features = len(features)
    for coin in data['coins']:
        all_transactions = coin['pastTrades'] + coin['futureTrades']
        random_start = np.random.randint(0, step)

        index = len(coin['pastTrades']) + random_start #first transaction not included
        index_of_last_y = None
        while index <= len(all_transactions): #<= since not included
            single_x_datapoint = np.zeros((trans_back_window, num_features))
            for i, transaction in enumerate(all_transactions[index - trans_back_window: index]):
                for j, feature in enumerate(features):
                    single_x_datapoint[i][j] = transaction[feature]
                    if feature == "time":
                        single_x_datapoint[i][j] = (transaction['time'] - coin['pastTrades'][-1]['time'])
                    if scaling is not None:
                        single_x_datapoint[i][j] = single_x_datapoint[i][j] * scaling[feature]

            y, index_of_last_y = get_y_change(coin, index - 1 - len(coin['pastTrades']), 
                                        time_forward, index_of_last_y = index_of_last_y)
            y = min(2, y)
            X.append(single_x_datapoint)
            y_points.append(y)
            index += step


    y = np.array(y_points)
    X = np.stack(X, axis = 0)
    return X, y




def main():
    file_path = "data/small_removed.json"
    save_path = "data/small_all_incorrect_points_removed.json"
    # make_tiny(file_path, save_path, num_coins = 5)

    add_features(save_path, seconds_back=60* 3, LSTM_cheating= True, LSTM_future_times = [1,5,10])



if __name__ == "__main__":
    main()

    exit()
    # scaling = {'volumeSol': 1, "coin_count": 1e-6, "time": 1e-4, "price": 1e7}
    # X, y = get_np_data("data/transaction_data.json", trans_back_window = 50, time_forward = get_length_of_time('min')/2, 
    #             step = 10, scaling = scaling, features = ['volumeSol', "coin_count", "time", "price"])
    # print(X.shape)
