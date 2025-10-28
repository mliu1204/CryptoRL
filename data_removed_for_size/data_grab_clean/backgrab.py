import requests
import os
import json
import time
from util import unix_to_datetime_str, get_length_of_time

HEADERS = {
  "x-api-key": "INSERT HERE"
}

def safe_request(url, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=10)
            response.raise_for_status()
            return response.json(), attempt + 1
        except (requests.exceptions.RequestException, ValueError) as e:
            error_msg = f"[Attempt {attempt + 1}] Time {time.time()} Error for URL {url} with params {params}:\n{e}"
            print(error_msg)
            log_error(error_msg)
            time.sleep(1)  
    return None, max_retries  # all attempts failed

def log_error(message: str):
    os.makedirs("data", exist_ok=True)
    with open("data/backgrab_error_log.txt", "a") as f:
        f.write(message + "\n")



def get_trades(mint, tracking_start, additional_time, max_api_calls = 100, percent_api_calls_future = 0.25) -> dict:
    """
    Purpose: Get all trades information for one coin and return a dictionary of trades.
    
    Parameters:
        mint (str): Mint address of the coin.
        tracking_start (float): Start time for tracking trades.
        additional_time (float): Time after tracking_start to include in future trades.

    Returns:
        dict: Structure containing trade metadata and lists of past/future trades.
            {
                "mint": str,
                "trackingStart": float,
                "trackingEnd": float, ()
                "totalTrades": int,
                "pastTrades": list,
                "futureTrades": list,
                "pastTooBigDaddy": boolean, (is true when expended all api calls and was still unable to capture good amounts of future trades)
                "futureTooBigDaddy": boolean (is true when expended all api calls and was still unable to capture good amounts of future trades)
            }

    Notes:
        All trades are in ascending order, i.e., pastTrades[i+1] happens after pastTrades[i].
    """
    tracking_end = tracking_start + additional_time
    total_api_used = 0
    data = {
        "mint": mint,
        "trackingStart": tracking_start,
        "trackingEnd": tracking_end,
        "pastTrades": [],
        "futureTrades": [],
        "pastTooBigDaddy": False,
        "futureTooBigDaddy": False,
    }

    params = {
        "cursor": tracking_start,
        "showMeta": True,
        "sortDirection": 'DESC',
        # dont know what this does 
        # "parseJupiter": True, 
    }

    base_url = 'https://data.solanatracker.io/trades'
    url = os.path.join(base_url, mint)
    

    # past trades
    next_page = True
    while next_page:
        response, api_used = safe_request(url, params)
        total_api_used += api_used
        if response is None:
            print("Failed to fetch past trades: see error log for more info")
            return None, total_api_used
        print("max past calls", max_api_calls * (1-percent_api_calls_future))
        print(total_api_used)
        if total_api_used > (max_api_calls * (1-percent_api_calls_future)): # too many api calls used for past
            print("Too many API calls made for past trades")
            data["pastTrades"].extend(response.get("trades", []))
            data["totalTrades"] = len(data["pastTrades"])
            data["pastTrades"].reverse()
            data["pastTooBigDaddy"] = True
            data["futureTooBigDaddy"] = True
            data["trackingEnd"] = data["pastTrades"][-1]["time"]
            return data, total_api_used
        data["pastTrades"].extend(response.get("trades", []))
        next_page = response.get("hasNextPage")
        params["cursor"] = response.get("nextCursor")
    data["pastTrades"].reverse()

    # future trades

    params["cursor"]= tracking_start
    params["sortDirection"]= "ASC"

    final_time = 0
    next_page = True

    while final_time < tracking_end and next_page:
        if total_api_used > max_api_calls:
            print("Too many API calls made for future trades")
            data["pastTooBigDaddy"] = False
            data["futureTooBigDaddy"] = True
            data["trackingEnd"] = data["futureTrades"][-1]["time"]
            break
        response, api_used= safe_request(url, params)
        total_api_used += api_used
        if response is None:
            print("Failed to fetch future trades.")
            return None, total_api_used
        trades = response.get("trades", [])
        if not trades:
            print("Warning: last page of future trades contain no data.")
            break
        for trade in trades:
            trade_time = trade.get("time") or 0
            final_time = trade_time
            if trade_time <= tracking_end:
                data["futureTrades"].append(trade)
            else:
                break
        next_page = response.get("hasNextPage")
        params["cursor"] = response.get("nextCursor")
    data["totalTrades"] = len(data["pastTrades"]) + len(data["futureTrades"])
    return data, total_api_used


# example
coin = "67Sbg1wQCKdAQr93Rds6dGy7J8yfJtiJCPcFy5tpump"
grabbed_time = 1747182983874.2368

