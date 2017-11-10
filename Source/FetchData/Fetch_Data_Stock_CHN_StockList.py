import sys, os, time, datetime, requests, warnings, configparser
import pandas as pd
import numpy as np
import tushare as ts
import concurrent.futures
from tqdm import tqdm

cur_path = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    root_path = cur_path[0:cur_path.rfind('/', 0, len(cur_path))]
    cur_path = root_path
sys.path.append(root_path + "/" + 'Source/DataBase/')
from DB_API import queryStockList, storeStockList

def getStocksList_CHN(root_path):
    try:
        df = queryStockList(root_path, "DB_STOCK", "SHEET_CHN")
        df.index = df.index.astype(str).str.zfill(6)
    except Exception as e:
        df = pd.DataFrame()

    if df.empty == False: return df
    
    stock_info = ts.get_stock_basics()
    listData = pd.DataFrame(stock_info)
    listData['daily_update'] = '1970-07-01'
    listData['weekly_update'] = '1970-07-01'
    listData['monthly_update'] = '1970-07-01'
    listData['news_update'] = '1970-07-01'
    listData.index.name = 'symbol'
    listData = listData.reset_index()

    #listData.index.name = 'symbol'
    #listData.index = listData.index.astype(str).str.zfill(6) #[str(symbol).zfill(6) for symbol in listData.index] #listData.index.astype(str).str.zfill(6)
    #print(listData.index)
    #listData['symbol'] = listData['symbol'].str.strip()

    storeStockList(root_path, "DB_STOCK", "SHEET_CHN", listData)
    df = queryStockList(root_path, "DB_STOCK", "SHEET_CHN")

    if df.empty == False: df.index = df.index.astype(str).str.zfill(6)
    return df


if __name__ == "__main__":
    pd.set_option('precision', 3)
    pd.set_option('display.width',1000)
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    config = configparser.ConfigParser()
    config.read(root_path + "/" + "config.ini")
    storeType = int(config.get('Setting', 'StoreType'))

    getStocksList_CHN(root_path)

