import requests
import os
import json
from util import *
import time
from backgrab import get_trades

MAX_API_CALLS = 100
TOTAL_API_CALLS = 0

def load_in_df():
    if os.path.exists(TRACKED_COINS):
        with open(TRACKED_COINS, 'r', encoding = 'utf-8') as f:
            data = json.load(f)
        coins = data['coins']

        included = set([])
        for coin in coins:
            included.add(coin['mint'])

    else:
        data = {'coins': []}
        included = set([])
    print(f"Tracked mints loaded: {len(data['coins'])}")
    return data, included

def live_grab_data(min_transactions):
    data, included_mints = load_in_df()
    fifteen_min = 15 * get_length_of_time('min')

    while True:
        start_time = time.time()
        try:
            current_time = get_cur_time()
            coins_data = api_call_candidate_mints(start = current_time - fifteen_min, end = current_time, min_transactions = min_transactions)
            if coins_data['total'] > 100:
                print(f"Error: found more than 100 datapoints in period ending at {current_time}")

            for coin in coins_data['data']:
                if coin['mint'] not in included_mints and coin['lpBurn'] != 0:
                    data['coins'].append(coin)
                    included_mints.add(coin['mint'])
        except Exception as e:
            print(f'Start time of {current_time} failed')

        with open(TRACKED_COINS, 'w', encoding = 'utf-8') as f:
            json.dump(data, f, indent = 2)

        print(f'Total coins: {len(data["coins"])}')
        time_to_sleep = 60 - (time.time() - start_time)

        if time_to_sleep < 0:
            print("Problem: code took too long")
        else:
            time.sleep(time_to_sleep)


      
def get_mints_to_track_historical(end_time, hours_back, min_transactions):
    data, included_mints = load_in_df()
    fifteen_min = 15 * get_length_of_time('min')
    start_time = end_time - hours_back * get_length_of_time('hour')

    more_than_100_dp = [] #list of start times with > 100 datapoints to load
    failed_api_calls = [] #list of start times where the api call failed

    begin = start_time
    while begin + fifteen_min <= end_time:
        try:
            coins_data = api_call_candidate_mints(start = begin, end = begin + fifteen_min, min_transactions = min_transactions)
            print(coins_data['total'])
            if coins_data['total'] > 100:
                print(f"Error: found more than 100 datapoints in period beginning at {begin}")
                more_than_100_dp.append(begin)
            for coin in coins_data['data']:
                if coin['mint'] not in included_mints and coin['lpBurn'] != 0:
                    data['coins'].append(coin)
                    included_mints.add(coin['mint'])
        except Exception as e:
            print(f'Start time of {begin} failed with reason {e}')
            failed_api_calls.append(begin)

        begin += fifteen_min #should be one minute
        break

    print(f'Total mints included: {len(included_mints)}')

    if len(more_than_100_dp) > 0:
        print("The following start times had more than 100 datapoints in their range")
        print(more_than_100_dp)

    if len(failed_api_calls) > 0:
        print("The following start times ran a bug in their api call")
        print(failed_api_calls)

    with open(TRACKED_COINS, 'w', encoding = 'utf-8') as f:
        json.dump(data, f, indent = 2)

    
def api_call_candidate_mints(start, end, min_transactions):
    global TOTAL_API_CALLS
    params = {
        "limit": 100,
        "minCreatedAt": start,
        "maxCreatedAt": end,
        "minTotalTransactions": min_transactions,
        "showPriceChanges": True
        }
    url = 'https://data.solanatracker.io/search'  
    resp = requests.get(url, headers=HEADERS, params=params)
    data = resp.json()

    TOTAL_API_CALLS += 1
    if TOTAL_API_CALLS >= MAX_API_CALLS:
        raise ValueError(f"Reached max number of api calls of {TOTAL_API_CALLS}")

    return data

def remove_bonk():
    with open(TRACKED_COINS, 'r') as f:
        data = json.load(f)
    new_data = {"coins": []}
    count_total = 0
    count_included = 0
    for coin in data['coins']:
        count_total += 1
        mint = coin['mint']
        if mint[-4:] == "pump":
            count_included += 1
            new_data['coins'].append(coin)
    with open(TRACKED_COINS, 'w') as f:
        json.dump(new_data, f, indent = 2)
    print(f'We initially have {count_total} and after removing bonk coins we have {count_included}')

def open_and_save_files():
    with open(TRANSACTION_DATA, 'r') as t_f:
        transaction_data = json.load(t_f)

    with open(OLD_TRANSACTION_DATA, 'w') as old_t_f:
        json.dump(transaction_data, old_t_f, indent = 2)
    
    with open(TOO_BIG_COINS, 'r') as too_big_f:
        too_big_data = json.load(too_big_f)

    mints_alr_in_df = set([])
    for coin in transaction_data['coins']:
        mints_alr_in_df.add(coin['mint'])

    with open (TRACKED_COINS, 'r') as m_f:
        tracking_data = json.load(m_f)
    
    return transaction_data, too_big_data, tracking_data, mints_alr_in_df

def update_files(tracking_data, transaction_data, too_big_data, mints_to_remove):
    mints_to_remove = set(mints_to_remove)
    new_tracking_data = {'coins': []}
    for coin in tracking_data['coins']:
        if coin['mint'] not in mints_to_remove:
            new_tracking_data['coins'].append(coin)
    
    with open(TRACKED_COINS, 'w') as f:
        json.dump(new_tracking_data, f, indent = 2)

    with open(TRANSACTION_DATA, 'w') as f:
        json.dump(transaction_data, f, indent = 2)

    with open(TOO_BIG_COINS, 'w') as f:
        json.dump(too_big_data, f, indent= 2)

def grab_all_transaction_data(additional_time):
    global TOTAL_API_CALLS

    transaction_data, too_big_data, tracking_data, mints_alr_in_df = open_and_save_files()
    mints_to_remove = []

    for coin in tracking_data['coins']:
        print("one coin")
        tracking_start = coin['lastUpdated']
        mint = coin['mint']
        mints_to_remove.append(mint)
        if mint in mints_alr_in_df:
            print("Tried to add existing mint")
            continue

        trades, api_call_used = get_trades (mint, tracking_start, additional_time)
        if trades is not None:
            coin["trackingStart"] = trades["trackingStart"]
            coin["trackingEnd"] = trades["trackingEnd"]
            coin["actualTotalTrades"] = trades["totalTrades"]
            coin["pastTrades"] = trades["pastTrades"]
            coin["futureTrades"] = trades["futureTrades"]
            coin['futureTooBigDaddy'] = trades['futureTooBigDaddy']
            if trades['pastTooBigDaddy']:
                too_big_data['coins'].append(coin)
            else:
                transaction_data['coins'].append(coin)

        TOTAL_API_CALLS += api_call_used
        if TOTAL_API_CALLS > MAX_API_CALLS:
            break
        
    update_files(tracking_data, transaction_data, too_big_data, mints_to_remove)



if __name__ == "__main__":
    grab_all_transaction_data(get_length_of_time('hour'))