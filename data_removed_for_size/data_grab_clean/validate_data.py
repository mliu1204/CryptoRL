import pandas as pd
import copy
import time
import requests
from util import *
import json
import random
from backgrab import get_trades
from operator import itemgetter



def confirm_no_overlapping_mints():
    problems = False
    df = pd.read_csv(TRACKED_COINS)
    mints_list = list(df['mint'])
    mints_set = set(mints_list)
    if len(mints_list) != len(mints_set):
        print("Problem: We Have Collected one Mint Address Multiple Times")
    else:
        print("Passed check; no overlapping mints")
    
def graph_coin_count_over_time(count_to_track, days_gap):
    cur_time = time.time() * 1000
    one_hour = 1000 * 60 * 60
    one_day = one_hour * 24
    coins = []
    time_frame = one_day * days_gap
    for days in range(count_to_track):
        end = cur_time - time_frame * days
        start = cur_time - time_frame * (days + 1)
        params = {
            "limit": 100,
            "minCreatedAt": start,
            "maxCreatedAt": end
        }
        url = 'https://data.solanatracker.io/search'

        resp = requests.get(url, headers=HEADERS, params=params)
        data = resp.json()

        coins.append(data['total'])
    plot_items_over_time(coins, x_label = 'days/weeks/months ago', y_label = 'all_coin_count', title = 'coin count over time')
    
    

def marginal_coin(lower_bound_transactions, upper_bound_transactions):
    cur_time = time.time()
    past_min = 15
    one_hour = 60 * 60
    one_day = one_hour * 24
    cur_time = cur_time - one_day
    params_1 = {
      "limit": 100,
      "minCreatedAt": 1000 * cur_time - 1000 * 60 * past_min,
      "maxCreatedAt": 1000 * cur_time,
      "minTotalTransactions": lower_bound_transactions,
      }

    params_2 = copy.deepcopy(params_1)
    params_2['minTotalTransactions'] = upper_bound_transactions

    url = 'https://data.solanatracker.io/search'

    resp_1 = requests.get(url, headers=HEADERS, params=params_1)
    data_1 = resp_1.json()
    total_mints = set([])
    for coin in data_1['data']:
        total_mints.add(coin['mint'])
    
    time.sleep(2)

    resp_2 = requests.get(url, headers=HEADERS, params=params_2)
    data_2 = resp_2.json()
    subset_mints = set([])
    print(data_2)
    for coin in data_2['data']:
        subset_mints.add(coin['mint'])
    print("Printing all the coins that fall between the two bounds")
    for coin in total_mints:
        if coin not in subset_mints:
            print(coin)
    print(f'Total: {len(total_mints)} | Subset: {len(subset_mints)}')
    


def test_grabbing_data(n_coins = 1):

    with open("data/tracked_coins.json", 'r') as f:
        data = json.load(f)
    coins = data['coins']
    coins = sorted(coins, key = itemgetter('createdAt'))
    print(coins[0])
    collectedData = []
    for i in range(10):
        coin = coins[i]
        coin_mint = coin['mint']
        tracking_start = coin['lastUpdated']
        example = get_trades(coin_mint, tracking_start, additional_time = get_length_of_time('hour')*2)
        print(example['mint'])
        print(example['totalTrades'])
        print(example['pastTrades'][0]['time'])
        print(example['pastTrades'][-1]['time'])
        collectedData.append(example)
    for i in range(10):
        coin = coins[-(i+1)]
        coin_mint = coin['mint']
        tracking_start = coin['lastUpdated']
        example = get_trades(coin_mint, tracking_start, additional_time = get_length_of_time('hour')*2)
        example["expectedTotalPastTrade"] = coin['totalTransactions']
        print(example['mint'])
        print(example['totalTrades'])
        print(example['pastTrades'][0]['time'])
        print(example['pastTrades'][-1]['time'])
        collectedData.append(example)
    save_path = "data/example_trades.json"
    with open(save_path, 'w') as json_file:
        json.dump(collectedData, json_file, indent=4)






if __name__ == "__main__":
    test_grabbing_data()
        



    # marginal_coin(100, 100000)