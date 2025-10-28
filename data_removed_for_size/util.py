"""
Alot of the data cleaning here
"""


import matplotlib
matplotlib.use('Agg') #headless so it doesn't show
import matplotlib.pyplot as plt
import datetime as dt
import time
import pytz 
import os
import json
import math
import numpy as np

my_tz = pytz.timezone("America/Los_Angeles")


MIN_PCT_CHANGE_FROM_BIRTH = 10
TRACKED_COINS = "data/tracked_coins.json"
OLD_TRACKED_COINS = "data/tracked_coins_saved.json"
TRANSACTION_DATA = "data/transaction_data.json"
OLD_TRANSACTION_DATA = "data/transaction_data_saved.json"
TOO_BIG_COINS = "data/too_big_coins.json"

HEADERS = {
  "x-api-key": "323ad003-9417-4ed1-b2e8-80846d30add9"
}
API_CALL_COUNT = 0


def unix_to_datetime_str(timestamp, tz=my_tz):
    timestamp = timestamp/1000
    if tz is None:
        dt_obj = dt.datetime.fromtimestamp(timestamp)
    else:
        dt_obj = dt.datetime.fromtimestamp(timestamp, tz)
    return dt_obj.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt_obj.microsecond // 1000:03d}"


def plot_items_over_time(items, x_label, y_label, title):
    plt.figure()
    plt.plot(range(len(items)), items)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('my_plot.png')

def graph_coin_price_over_time(mint_address):
    url = 'https://data.solanatracker.io/chart/' + mint_address
    try:
        pass
    except Exception as e:
        print(f"Failed: {mint_address}")

def get_cur_time():
    return time.time() * 1000

def get_length_of_time(unit_str):
    ms_per_unit = {
        'sec': 1_000,
        'min': 60    * 1_000,
        'hour': 60   * 60   * 1_000,
        'day': 24    * 60   * 60   * 1_000,
        'week': 7    * 24   * 60   * 60   * 1_000,
        'year': 365  * 24   * 60   * 60   * 1_000,
    }
    try:
        return ms_per_unit[unit_str.lower()]
    except:
        raise ValueError("Queried get length of time on invalid time unit")


def time_datetime_str(timestamp, tz):
    if tz is None:
        dt_obj = dt.datetime.fromtimestamp(timestamp)
    else:
        dt_obj = dt.datetime.fromtimestamp(timestamp, tz)
    return dt_obj.strftime("%Y-%m-%d %H:%M")


def make_json(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    #print(f"Initialized JSON at {file_path}: {data}")
    return data

def remove_prev_trivial(file_path_in, file_path_out):
    with open(file_path_in, 'r') as f:
        data = json.load(f)
    
    min_coins_bought_needed = 2000
    trans_types = ['pastTrades', 'futureTrades']
    
    for coin in data['coins']:
        trivial_in_a_row = 0
        for trans_type in trans_types:
            all_transactions = []
            for transaction in coin[trans_type]:
                if transaction['coin_count'] < min_coins_bought_needed:
                    trivial_in_a_row += 1
                else:
                    prev_trivial_count = 0
                    if trivial_in_a_row > 0:
                        trivial_count_before = 0
                        if "prevTrivial" in transaction.keys():
                            trivial_count_before = transaction["prevTrivial"]
                        prev_trivial_count = trivial_count_before + trivial_in_a_row
                        transaction['prevTrivial'] = prev_trivial_count

                    all_transactions.append(transaction)
                    trivial_in_a_row = 0
            coin[trans_type] = all_transactions
    with open(file_path_out, 'w') as f: 
        json.dump(data, f, indent = 2)



def edit_timestamps(file_path_in, file_path_out):
    with open(file_path_in, 'r') as f:
        data = json.load(f)

    no_future = 0
    for coin in data['coins']:
        if len(coin['futureTrades']) > 0:
            init = coin['futureTrades'][0]['time']
        else:

            fifteen_min = 15 * get_length_of_time('min')
            init = np.random.uniform(coin['pastTrades'][-1]['time'], coin['pastTrades'][0]['time'] + fifteen_min)
            no_future += 1

        for transaction in coin['pastTrades']:
            transaction['time'] = transaction['time'] - init
        for transaction in coin['futureTrades']:
            transaction['time'] = transaction['time'] - init
    print(f"There were {no_future} with no future")
    with open(file_path_out, 'w') as f:
        json.dump(data, f, indent = 2) #remove indent = 2 for storage

def get_datapoints_mean(data):
    transaction_count = 0
    total_vol_sol = 0
    total_coins_given = 0
    total_time = 0
    transaction_categories = ['pastTrades', 'futureTrades']

    for coin in data['coins']:
        for category in transaction_categories:
            for trans in coin[category]:
                transaction_count += 1
                total_vol_sol += trans["volumeSol"]
                total_coins_given += trans["coin_count"]
                total_time += trans["time"]

    vol_sol_mean = total_vol_sol/transaction_count
    coins_given_mean = total_coins_given/transaction_count
    total_time_mean = total_time/transaction_count

    summary_stats = {'vol_sol_mean': vol_sol_mean, 'coins_given_mean': coins_given_mean, 'time_mean': total_time_mean}
    return summary_stats, transaction_count

def get_datapoints_std(data, summary_stats, transaction_count):
    total_var_vol_sol = 0
    total_var_coins_given = 0
    total_var_time = 0
    transaction_categories = ['pastTrades', 'futureTrades']

    for coin in data['coins']:
        for category in transaction_categories:
            for trans in coin[category]:
                total_var_vol_sol += (trans["volumeSol"] - summary_stats['vol_sol_mean']) ** 2
                total_var_coins_given += (trans["coin_count"] - summary_stats['coins_given_mean']) ** 2
                total_var_time += (trans["time"] - summary_stats['time_mean']) ** 2

    vol_sol_std = math.sqrt(total_var_vol_sol/transaction_count)
    coins_given_std = math.sqrt(total_var_coins_given/transaction_count)
    time_std = math.sqrt(total_var_time/transaction_count)
    summary_stats['vol_sol_std'] = vol_sol_std
    summary_stats['coins_given_std'] = coins_given_std
    summary_stats['time_std'] = time_std
    return summary_stats

#DOES NOT SUBTRACT THE MEAN
def update_data(data, summary_stats, final_file_path):
    transaction_categories = ['pastTrades', 'futureTrades']

    for coin in data['coins']:
        for category in transaction_categories:
            for trans in coin[category]:
                trans['volumeSol'] = (trans['volumeSol'])/(summary_stats['vol_sol_std'])
                trans['time'] = (trans['time'])/(summary_stats['time_std'])
                trans['coin_count'] = (trans['coin_count'])/(summary_stats['coins_given_std'])
    
    with open(final_file_path, "w") as f:
        json.dump(data, f, indent = 2)
    

#DOES NOT SUBTRACT THE MEAN
def normalize_data(init_file_path, final_file_path):
    with open(init_file_path, 'r') as f:
        data = json.load(f)

    summary_stats, transaction_count = get_datapoints_mean(data)
    

    summary_stats = get_datapoints_std(data, summary_stats, transaction_count)
    print(summary_stats)
    update_data(data, summary_stats, final_file_path)

def remove_too_short_history(init_file_path, final_file_path):
    with open(init_file_path, 'r') as f:
        data = json.load(f)
    old_coins = data['coins']
    new_coins = []
    for coin in old_coins:
        try:
            if len(coin['pastTrades']) >= 100:
                new_coins.append(coin)
        except:
            pass
    print(f'We had before {len(old_coins)} and now we have {len(new_coins)}')
    new_data = {'coins': new_coins}
    with open(final_file_path, 'w') as f:
        json.dump(new_data, f, indent = 2)


def remove_fradulent_time_data_coins(init_file_path, final_file_path):
    with open(init_file_path, 'r') as f:
        data = json.load(f)
    old_coins = data['coins']
    new_coins = []
    for coin in old_coins:
        try:
            time_dif = coin['futureTrades'][-1]['time'] - coin['pastTrades'][0]['time']
            if time_dif < 1.5 * get_length_of_time('hour'):
                new_coins.append(coin)
        except:
            pass
    print(f'We had before {len(old_coins)} and now we have {len(new_coins)}')
    new_data = {'coins': new_coins}
    with open(final_file_path, 'w') as f:
        json.dump(new_data, f, indent = 2)

def add_price(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    transaction_types = ['futureTrades', 'pastTrades']
    for coin in data['coins']:
        for trans_type in transaction_types:
            for trans in coin[trans_type]:
                trans['price'] = abs(trans['volumeSol'])/trans['coin_count']
    with open(file_path, 'w') as f:
        json.dump(data, f, indent = 2)

def move_price_fake(price, direction, index):
    index += 1
    movement = np.random.uniform(0, 0.01)
    price += movement * direction

    if (np.random.randint(0, 10) == 0):
        direction = np.random.randint(-1, 2)
    transaction = {'volumeSol': np.random.normal(),
                    "coin_count": np.random.normal(),
                    "price": price,
                    "time": index}
    return price, direction, transaction, index

def generate_fake_coin_new():
    prior_transaction_count = 200
    future_transaction_count = 1000
    coin = {}
    pastTrades = []
    price = 1
    direction = np.random.randint(-1, 2)
    index = 0
    while index < prior_transaction_count:
        price, direction, transaction, index = move_price_fake(price, direction, index)
        pastTrades.append(transaction)
    
    futureTrades = []
    while index < prior_transaction_count + future_transaction_count:
        price, direction, transaction, index = move_price_fake(price, direction , index)
        futureTrades.append(transaction)
    
    coin= {"pastTrades": pastTrades, "futureTrades": futureTrades}
    return coin

def generate_fake_coin_old(): #from 0.3 -> 1 or 2, so should be able to get 6
    prior_transaction_count = np.random.randint(100, 200)
    future_transaction_count = np.random.randint(600, 800)
    coin = {}
    pastTrades = []
    price = 0.0001
    for i in range(prior_transaction_count):
        transaction = {'volumeSol': np.random.normal(),
                        "coin_count": np.random.normal(),
                        "price": price,
                        "time": -((prior_transaction_count - i)/prior_transaction_count)}
        pastTrades.append(transaction)
        price += 0.002
    coin['pastTrades'] = pastTrades
    
    futureTrades = []
    decreasing = False
    for i in range(future_transaction_count):
        transaction = {'volumeSol': np.random.normal(),
                        "coin_count": np.random.normal(),
                        "price": price,
                        "time": ((i)/800)}
        futureTrades.append(transaction)
        if price > 1:
            decreasing = True
        if decreasing:
            price -= 0.002
        else:
            price += 0.002
    coin['futureTrades'] = futureTrades
    return coin



def make_fake_data(num_coins, file_path):
    data = {'coins': []}
    for i in range(num_coins):
        coin = generate_fake_coin_new()
        data['coins'].append(coin)
    print(f"made {len(data['coins'])} fake coins")
    make_json(file_path, data)

    return data

def remove_overlap_with_test_file(train, test):
    with open(train, 'r') as f:
        train_data = json.load(f)
    with open(test, 'r') as f:
        test_data = json.load(f)

    train_mints = set([])
    for coin in train_data['coins']:
        train_mints.add(coin['mint'])

    final_test_coins = []
    for coin in test_data['coins']:
        if coin['mint'] not in train_mints:
            final_test_coins.append(coin)
    final_test_data = {"coins": final_test_coins}
    print(f'We have {len(final_test_coins)} legitimate and {len(test_data["coins"]) - len(final_test_coins)} illegitimate')
    with open(test, 'w') as f:
        json.dump(final_test_data, f, indent = 2)


if __name__ == "__main__":
    #add_price("data/transaction_data.json")
    #remove_fradulent_time_data_coins("data/small_transaction_data.json", "data/small_all_incorrect_points_removed.json")
    #remove_too_short_history("data/small_all_incorrect_points_removed.json", "data/small_all_incorrect_points_removed.json")
    #edit_timestamps("data/small_all_incorrect_points_removed.json", "data/small_all_incorrect_points_removed.json")
    #normalize_data("data/small_updated.json", "data/final_small.json")
    #make_fake_data(500, "data/fake_data.json")
   
    make_fake_data(5, "data/fake_data_new.json")