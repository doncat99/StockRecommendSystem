import sys, os, io, time, datetime, requests, warnings, configparser
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
import concurrent.futures
from tqdm import tqdm

cur_path = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    root_path = cur_path[0:cur_path.rfind('/', 0, len(cur_path))]
    cur_path = root_path
sys.path.append(root_path + "/" + 'Source/DataBase/')
sys.path.append(root_path + "/" + 'Source/Utility/')

from DB_API import queryStockList, storeStockList
import fix_yahoo_finance as yf

def getStocksList_US(root_path):
    try:
        df = queryStockList(root_path, "DB_STOCK", "SHEET_US")
    except Exception as e:
        df = pd.DataFrame()

    if df.empty == False:
        return df

    for exchange in ["NASDAQ", "NYSE"]:
        print("fetching " + exchange + " stocklist...")
        url = "http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=%s&render=download" % exchange
        repeat_times = 1 # repeat downloading in case of http error
        for _ in range(repeat_times): 
            try:
                urlData = requests.get(url, timeout=15).content
                if df.empty:
                    df = pd.read_csv(io.StringIO(urlData.decode('utf-8')))
                else:
                    df = pd.concat([df, pd.read_csv(io.StringIO(urlData.decode('utf-8')))])                    
                break
            except Exception as e:
                print ("exception in getStocks:" + exchange, str(e))
                continue

    df = df[(df['MarketCap'] > 100000000)]
    df = df.drop_duplicates(subset=['Symbol'], keep='first')
    df.sort_index(ascending=True, inplace=True)
    listData = df[['Symbol', 'Name', 'MarketCap', 'Sector', 'Industry']].copy()

    listData=listData.rename(columns = {'Symbol':'symbol', 'Name':'name', 'MarketCap':'market_cap', 'Sector':'sector', 'Industry':'industry'})
    listData.loc[len(listData)] = ['SPY', 'SPDR S&P 500 ETF Trust', 0.0, '', '']
    listData.loc[len(listData)] = ['^VIX', 'VOLATILITY S&P 500', 0.0, '', '']
    listData['symbol'] = listData['symbol'].str.strip()

    listData['daily_update'] = '1970-07-01'
    listData['weekly_update'] = '1970-07-01'
    listData['monthly_update'] = '1970-07-01'
    listData['news_update'] = '1970-07-01'

    listData = listData.reset_index(drop=True)
    storeStockList(root_path, "DB_STOCK", "SHEET_US", listData)
    return queryStockList(root_path, "DB_STOCK", "SHEET_US")




if __name__ == "__main__":
    pd.set_option('precision', 3)
    pd.set_option('display.width',1000)
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    getStocksList_US(root_path)


    