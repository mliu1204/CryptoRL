"""
used to debug data and create plots
"""

import time
from util import *
import json
import random
from collections import Counter
import datetime
import matplotlib.pyplot as plt
import math
import pandas as pd

def graph_coin_creation_date():
    with open("data/tracked_coins.json", 'r') as f:
        data = json.load(f)
    unix_timestamps = []
    for coin in data['coins']:
            unix_timestamps.append(coin['createdAt']/1000)


    dates = [datetime.datetime.fromtimestamp(ts).date() for ts in unix_timestamps]

    counts = Counter(dates)
    del counts[datetime.date(2025, 5, 15)]
    del counts[datetime.date(2025, 5, 23)]
    sorted_dates = sorted(counts)
    frequencies = [counts[d] for d in sorted_dates]

    plt.figure()
    plt.bar(sorted_dates, frequencies)
    plt.ylabel("Count")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/coins_over_time.png")

#how many transactions there had been when we grabbed the coin
def graph_transaction_dis_before_following():
    with open("data/tracked_coins.json", 'r') as f:
        data = json.load(f)
    total_transactions = []
    no_transaction_coins = []
    test = []
    for coin in data['coins']:
        total = coin['totalTransactions']
        if total == 0:
            no_transaction_coins.append(coin['mint'])
        else:
            test.append(coin['totalTransactions'])
            total_transactions.append(math.log(coin['totalTransactions'], 10))
    print(min(test))
    print(no_transaction_coins)
    plt.hist(total_transactions, bins = 30)
    plt.ylabel("Count")
    plt.xlabel("Log Transactions")
    plt.savefig("plots/total_transactions.png")

def graph_transaction_dis_after_following():
    with open(TRANSACTION_DATA, 'r') as f:
        data = json.load(f)
    past_counts = []
    future_counts = []
    for coin in data['coins']:
        past_counts.append(len(coin['pastTrades']))
        future_counts.append(len(coin['futureTrades']))
        if len(coin['futureTrades']) > 5000:
            print(coin['mint'])
    plt.hist(past_counts)
    plt.savefig("plots/real_past_trades.png")
    plt.hist(future_counts)
    plt.savefig("plots/real_future_trades.png")

def graph_rl_learning(file_name):
    df = pd.read_csv(file_name)


    plt.figure(figsize=(10, 6))
    plt.plot(df['Wall time'], df['Value'], marker='o', linewidth=2)
    plt.xlabel('Wall time', fontsize=14, fontweight='bold')
    plt.ylabel('Value', fontsize=14, fontweight='bold')
    plt.title('Value over Time', fontsize=16, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()

    #display the plot
    plt.savefig("plots/rl_learning.png")

def plot_price_over_time1(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    prices = []
    coin = data['coins'][0]
    for trans_t in ['pastTrades', 'futureTrades']:
        for t in coin[trans_t]:
            prices.append(t['price'])

    time = list(range(len(prices)))

    plt.figure(figsize=(10, 6), dpi=100)

    plt.plot(
        time, prices,
        linestyle='-',           # solid line connecting points
        linewidth=1.5,            # a slightly thicker line for visibility
        marker='.',               # pixel‐sized marker (one screen pixel)
        markersize=4,             # 4 is small enough that adjacent points don't really overlap
        alpha=1.0,                # opaque so dots won’t “blend” into a blur
        color='C0'
    )

    plt.xlabel('Time (step index)', fontsize=18, fontweight='bold')
    plt.ylabel('Price ', fontsize=18, fontweight='bold')
    plt.title('Simulated Predictable Data', fontsize=22, fontweight='semibold')

    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    plt.margins(x=0.02, y=0.05)
    plt.tight_layout()
    plt.savefig('plots/price_changes.png', dpi=200, bbox_inches='tight')
    plt.close()

def plot_price_over_time(file_path, coin_indices=[0, 1, 2, 3]):
    with open(file_path, 'r') as f:
        data = json.load(f)

    #helper to extract the price‐list for a given coin index
    def extract_prices(coin_idx):
        prices = []
        coin = data['coins'][coin_idx]
        for trans_t in ['pastTrades', 'futureTrades']:
            for t in coin[trans_t]:
                prices.append(t['price'])
        return prices

    plt.figure(figsize=(10, 6), dpi=100)

    for idx, coin_idx in enumerate(coin_indices):
        prices = extract_prices(coin_idx)
        time = np.arange(len(prices))

        plt.plot(
            time,
            prices,
            label=f'Coin #{coin_idx}',
            color=f'C{idx}',
            linewidth=1.5,
            marker='.',        # pixel‐sized dot
            markersize=5,
            alpha=1.0
        )

    plt.xlabel('Time (step index)', fontsize=14, fontweight='bold')
    plt.ylabel('Price (USD)', fontsize=14, fontweight='bold')
    plt.title('Price Changes Over Time for Four Coins', fontsize=16, fontweight='semibold')
    plt.legend(fontsize=12, loc='upper left')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    plt.margins(x=0.02, y=0.05)
    plt.tight_layout()
    plt.savefig('plots/four_coins_price_changes.png', dpi=200, bbox_inches='tight')
    plt.close()

def info_on_our_transaction_data():
    with open("data/test.json", 'r') as f:
        data = json.load(f)
    coin_count = 0
    transaction_count = 0
    transaction_volume = 0
    trivial_count = 0
    trade_index_names = ['pastTrades', 'futureTrades']
    too_big_count = 0
    max_time_dif = 0
    too_short_history = 0

    none_in_future = 0
    for coin in data['coins']:
        if len(coin['pastTrades']) < 0:
            too_short_history += 1
        coin_count += 1
        try:
            time_dif = coin['futureTrades'][-1]['time'] - coin['pastTrades'][0]['time']
            if time_dif > max_time_dif:
                max_time_dif = time_dif
            if time_dif > get_length_of_time('hour') * 1.5:
                print(coin['mint'])
                print(coin['futureTrades'][-1]['time'], coin['pastTrades'][0]['time'])
        except:
            none_in_future += 1
            

        for index_name in trade_index_names:
            transactions = coin[index_name]
            for transaction in transactions:
                transaction_count += 1
                transaction_volume +=  abs(transaction['volumeSol'])
                if 'prevTrivial' in transaction.keys():
                    trivial_count += transaction['prevTrivial']
    print(f"We have a total of {coin_count} coins and {transaction_count} transactions with {transaction_volume} total volume in solana, with an additional {trivial_count} trivial transaction")
    print(f"The maximum time difference is {max_time_dif}")
    print(f"The count that have none in the future is {none_in_future}")
    print(f"We have {too_short_history} with too short of a history")
def smaller_dataframe(output_file):
    with open(TRANSACTION_DATA, 'r') as f:
        data = json.load(f)
    
    all_coins = data['coins']
    subsampled_coins = all_coins[::10]
    smaller_data = {'coins': subsampled_coins}

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(smaller_data, f, indent=2)


if __name__ == "__main__":
    graph_rl_learning("data/PPO_0.csv")