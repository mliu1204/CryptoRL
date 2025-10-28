import pandas as pd
import copy
import time
import requests
from util import *
import json
import random
from backgrab import get_trades
from operator import itemgetter

def print_basic_data (data):

    for cur in data:
        print(cur.keys())
        totalTrades = cur['totalTrades']
        pastTrades = len(cur['pastTrades'])
        print (cur['mint'])
        print ("total Trades count: " + str(cur['totalTrades']))
        print("past trades count: " + str(len(cur['pastTrades'])))
        print("expected past trades:" + str(cur['expectedTotalPastTrade']))
        # print (cur['pastTrades'][0]['tx'])
        # print (cur['pastTrades'][-1]['tx'])
        # if totalTrades != pastTrades:
        #     print (cur['futureTrades'][0]['tx'])
        #     print (cur['futureTrades'][-1]['tx'])
        # else:
        #     print("only past trades")
        print ("----------------------------------------")

def print_all_trades(trades):
    all_trades = trades['pastTrades'] + trades['futureTrades']
    for trade in all_trades:
        print(trade['tx'][:6])


with open("data/example_trades.json", 'r') as f:
    data = json.load(f)

print_basic_data(data)
# print_all_trades(data[-2])
