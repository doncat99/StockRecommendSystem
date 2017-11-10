import sys, os, time, datetime, warnings, configparser
import pandas as pd
import numpy as np
import concurrent.futures
import tushare as ts
import talib
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

cur_path = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    root_path = cur_path[0:cur_path.rfind('/', 0, len(cur_path))]
    cur_path = root_path
sys.path.append(root_path + "/" + 'Source/FetchData/')
sys.path.append(root_path + "/" + 'Source/DataBase/')

from Fetch_Data_Stock_CHN_Daily import updateStockData_CHN_Daily
from Fetch_Data_Stock_CHN_StockList import getStocksList_CHN
from DB_API import queryStock

def get_single_stock_data_daily(root_path, symbol):
    '''
    All data is from quandl wiki dataset
    Feature set: [Open  High    Low  Close    Volume  Ex-Dividend  Split Ratio Adj. Open  Adj. High  Adj. Low
    Adj. Close  Adj. Volume]
    '''
    df, lastUpdateTime = queryStock(root_path, "DB_STOCK", "SHEET_CHN", "_DAILY", symbol, "daily_update")

    if df.empty: 
        #print("stock delisted", symbol)
        return df
    
    df.index = pd.to_datetime(df.index)
    suspended_day = pd.Timestamp((datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d"))
    if df.index[-1] < suspended_day:
        #print("stock suspended", symbol)
        return pd.DataFrame()

    return df

def get_single_stock_data_daily_date(root_path, symbol, start):
    '''
    All data is from quandl wiki dataset
    Feature set: [Open  High    Low  Close    Volume  Ex-Dividend  Split Ratio Adj. Open  Adj. High  Adj. Low
    Adj. Close  Adj. Volume]
    '''
    df = get_single_stock_data_daily(root_path, symbol)

    if df.empty: 
        #print("stock delisted", symbol)
        return []

    df = df[df.index >= pd.Timestamp(start)]
    datelist = df.index.strftime('%Y-%m-%d')

    return datelist

def get_single_stock_data_tick(root_path, symbol, date):
    '''
    All data is from quandl wiki dataset
    Feature set: [Open  High    Low  Close    Volume  Ex-Dividend  Split Ratio Adj. Open  Adj. High  Adj. Low
    Adj. Close  Adj. Volume]
    '''
    return ts.get_tick_data(symbol ,date=date, src ='tt')

def processing_cashflow(df):
    buy = df[df.type == '买盘']
    sell = df[df.type == '卖盘']
    even = df[df.type == '中性盘']

    buy.is_copy = False
    sell.is_copy = False
    even.is_copy = False
    
    #print("cash flow count: ", buy['count'].sum() - sell['count'].sum())
    buy_amount, sell_amount, even_amount, buy_volume, sell_volume, even_volume = buy['amount'].sum(), sell['amount'].sum(), even['amount'].sum(), buy['volume'].sum() * 100, sell['volume'].sum() * 100, even['volume'].sum() * 100
    if buy_volume == 0:
        buy_max, buy_min, buy_average = 0, 0, 0
    else:
        buy_max, buy_min, buy_average = buy['price'].max(), buy['price'].min(), round(buy_amount / buy_volume, 2)

    if sell_volume == 0:
        sell_max, sell_min, sell_average = 0, 0, 0
    else:
        sell_max, sell_min, sell_average = sell['price'].max(), sell['price'].min(), round(sell_amount / sell_volume, 2)

    if even_volume == 0:
        even_max, even_min, even_average = 0, 0, 0
    else:
        even_max, even_min, even_average = even['price'].max(), even['price'].min(), round(even_amount / even_volume, 2)

    return buy_amount, sell_amount, even_amount, buy_volume, sell_volume, even_volume, buy_max, buy_min, buy_average, sell_max, sell_min, sell_average, even_max, even_min, even_average
    

def processing_stock_data(root_path, start, df, symbol):
    startTime = time.time()

    date_list = get_single_stock_data_daily_date(root_path, symbol, start)
    df_symbol = df[(df.symbol == int(symbol))]
    if df_symbol.empty == False:
        df_symbol_date_list = df_symbol['date'].values.tolist()
        #date_list = list( (set(date_list) | set(df_symbol_date_list)) - (set(date_list) & set(df_symbol_date_list)) )
        date_list = list( set(date_list) - set(df_symbol_date_list) )
        
    pbar = trange(len(date_list), mininterval=0.1, smoothing=1, leave=False)
    for i in pbar:
        date = date_list[i]
        start = time.time()
        
        file_name = "backup/" + symbol + "_" + date + ".csv"
        if os.path.exists(file_name):
            data = pd.read_csv(file_name, index_col=0)

        else:
            data = get_single_stock_data_tick(root_path, symbol, date)
            if data is not None:
                data.to_csv(file_name)

        if (data is None) or data.empty or len(data) < 4:
            buy, sell, even = 0, 0, 0
        else:
            buy_amount, sell_amount, even_amount, buy_volume, sell_volume, even_volume, buy_max, buy_min, buy_average, sell_max, sell_min, sell_average, even_max, even_min, even_average = processing_cashflow(data)
            df.loc[len(df)] = [date, symbol, buy_amount, sell_amount, even_amount, buy_volume, sell_volume, even_volume, buy_max, buy_min, buy_average, sell_max, sell_min, sell_average, even_max, even_min, even_average]

        outMessage = '%s processed in: %.3s seconds' % (date, (time.time() - start))
        pbar.set_description(outMessage)
        

    #print(symbol, buy, sell, even)
    #print(date_list)
    return df, startTime, len(date_list) > 0

def update_all_stocks_data(root_path, symbols):
    start = (datetime.datetime.now() - datetime.timedelta(days=12*30)).strftime("%Y-%m-%d")

    filename = 'cashflow.csv'
    if os.path.exists(filename) == False:
        database = pd.DataFrame(columns=['date', 'symbol', 'buy_amount', 'sell_amount', 'even_amount', 'buy_volume', 'sell_volume', 'even_volume', 'buy_max', 'buy_min', 'buy_average', 'sell_max', 'sell_min', 'sell_average', 'even_max', 'even_min', 'even_average'])
        database.index.name = 'index'
    else:
        database = pd.read_csv(filename, index_col=["index"])

    pbar = tqdm(total=len(symbols))

    count = 0
    startTime_1 = time.time()
    for symbol in symbols:
        database, startTime, update = processing_stock_data(root_path, start, database, symbol)
        outMessage = '%-*s processed in:  %.4s seconds' % (6, symbol, (time.time() - startTime))
        pbar.set_description(outMessage)
        pbar.update(1)
        if update:
            count += 1
            if True:#count % 10 == 0: 
                database.to_csv(filename) 
                count = 0

    database = database.sort_values(['symbol','date'], ascending=[True, True])
    database.to_csv(filename)
    print('total processing in:  %.4s seconds' % ((time.time() - startTime_1)))

    # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    #     # Start the load operations and mark each future with its URL
    #     future_to_stock = {executor.submit(processing_stock_data, root_path, symbol, window, day_selection, week_selection, month_selection): symbol for symbol in symbols}
    #     for future in concurrent.futures.as_completed(future_to_stock):
    #         stock = future_to_stock[future]
    #         try:
    #             startTime = future.result()
    #         except Exception as exc:
    #             startTime = time.time()
    #             print('%r generated an exception: %s' % (stock, exc))
    #         outMessage = '%-*s processed in:  %.4s seconds' % (6, stock, (time.time() - startTime))
    #         pbar.set_description(outMessage)
    #         pbar.update(1)

# def cal_stock_data(root_path, df, db_count, symbol):
#     startTime = time.time()
#     df_symbol = df[(df.symbol == int(symbol))]
#     if df_symbol.empty:
#         return startTime
    
#     df_symbol = df_symbol.sort_values(['date'], ascending=[True])

#     df_stock = get_single_stock_data_daily(root_path, symbol)

#     days = [3, 5, 7, 10, 15, 20, 30, 40]
#     result = [symbol]

#     for day in days:
#         result.append(("%.2f" % (df_symbol[-day:]['cashflow'].sum() / 10000)))
#         if df_stock.empty | (len(df_stock) < day):
#             result.append("NaN")
#         else:
#             percentage = ("%.3f" % (100 * (df_stock['close'][-1] - df_stock['open'][-day]) / df_stock['open'][-day]))
#             #result.append(str(percentage) + "%")    
#             result.append(percentage)    

#     db_count.loc[len(db_count)] = result
        
#     return startTime

# def process_all_stocks_data(root_path, symbols, update):
#     filename = 'cashflow.csv'
#     filename_1 = 'cashflow_count.csv'
#     filename_2 = 'cashflow_count_filter.csv'

#     startTime_1 = time.time()

#     if (update == '1') | (os.path.exists(filename_1) == False):

#         database = pd.read_csv(filename, index_col=["index"])

#         db_count = pd.DataFrame(columns=['symbol', '3day', '3-price', '5day', '5-price', '7day', '7-price', '10day', '10-price', '15day', '15-price', '20day', '20-price', '30day', '30-price', '40day', '40-price'])
#         db_count.index.name = 'index'
#         count = 0

#         pbar = tqdm(total=len(symbols))

#         for symbol in symbols:
#             startTime = cal_stock_data(root_path, database, db_count, symbol)
#             outMessage = '%-*s processed in:  %.4s seconds' % (6, symbol, (time.time() - startTime))
#             pbar.set_description(outMessage)
#             pbar.update(1)
#             count += 1
#             if count % 10 == 0: 
#                 db_count.to_csv(filename_1) 
#                 count = 0

#         db_count.to_csv(filename_1) 
#     else:
#         db_count = pd.read_csv(filename_1, index_col=["index"], float_precision='round_trip')

#     db_count_filter = db_count[((db_count["3-price"].astype(float).fillna(0.0)) < 0.0)]
#     # db_count_filter = db_count[((db_count["3-price"].astype(float).fillna(0.0)) < 0.0) & \
#     #                            (abs(db_count["10-price"].astype(float).fillna(0.0)) < 5.0) & \
#     #                            (abs(db_count["20-price"].astype(float).fillna(0.0)) < 5.0) & \
#     #                            (abs(db_count["40-price"].astype(float).fillna(0.0)) < 5.0)]
#     db_count_filter.to_csv(filename_2)

#     print('total processing in:  %.4s seconds' % ((time.time() - startTime_1)))


def cal_stock_data_1(root_path, df, db_count, symbol):
    startTime = time.time()
    df_symbol = df[(df.symbol == int(symbol))]
    if df_symbol.empty:
        return startTime
    
    df_symbol = df_symbol.sort_values(['date'], ascending=[True])
    
    df_stock = get_single_stock_data_daily(root_path, symbol)

    days = [1, 2, 3]
    result = [symbol]

    for day in days:
        buy_count = df_symbol[-day:]['cashflow_buy'].sum()
        sell_count = df_symbol[-day:]['cashflow_sell'].sum()
        even_count = df_symbol[-day:]['cashflow_even'].sum()
        pure_buy = (buy_count - sell_count) / 10000
        pect_buy = 100 * abs(pure_buy) / ((buy_count + sell_count) / 10000)
        result.append(("%.2f" % pure_buy))
        result.append(("%.2f" % pect_buy))
        if df_stock.empty | (len(df_stock) < day):
            result.append("NaN")
        else:
            percentage = ("%.3f" % (100 * (df_stock['close'][-1] - df_stock['open'][-day]) / df_stock['close'][-day]))
            #result.append(str(percentage) + "%")    
            result.append(percentage)    
    
    db_count.loc[len(db_count)] = result
        
    return startTime

def process_all_stocks_data_1(root_path, symbols, update):
    filename = 'cashflow.csv'
    filename_1 = 'cashflow_count.csv'
    filename_2 = 'cashflow_count_filter.csv'

    startTime_1 = time.time()

    if (update == '1') | (os.path.exists(filename_1) == False):

        database = pd.read_csv(filename, index_col=["index"])

        db_count = pd.DataFrame(columns=['symbol', '1day', '1-pect', '1-price', '2day', '2-pect', '2-price', '3day', '3-pect', '3-price'])
        db_count.index.name = 'index'
        count = 0

        pbar = tqdm(total=len(symbols))

        for symbol in symbols:
            startTime = cal_stock_data_1(root_path, database, db_count, symbol)
            outMessage = '%-*s processed in:  %.4s seconds' % (6, symbol, (time.time() - startTime))
            pbar.set_description(outMessage)
            pbar.update(1)
            count += 1
            if count % 10 == 0: 
                db_count.to_csv(filename_1) 
                count = 0

        db_count.to_csv(filename_1) 
    else:
        db_count = pd.read_csv(filename_1, index_col=["index"], float_precision='round_trip')

    db_count_filter = db_count[((db_count["1-price"].astype(float).fillna(0.0)) < 0.0)]
    # db_count_filter = db_count[((db_count["3-price"].astype(float).fillna(0.0)) < 0.0) & \
    #                            (abs(db_count["10-price"].astype(float).fillna(0.0)) < 5.0) & \
    #                            (abs(db_count["20-price"].astype(float).fillna(0.0)) < 5.0) & \
    #                            (abs(db_count["40-price"].astype(float).fillna(0.0)) < 5.0)]
    db_count_filter.to_csv(filename_2)

    print('total processing in:  %.4s seconds' % ((time.time() - startTime_1)))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Input parameter error")
        exit()

    pd.set_option('precision', 3)
    pd.set_option('display.width',1000)
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    update = str(sys.argv[1])
    
    df = getStocksList_CHN(root_path)
    df.index = df.index.astype(str).str.zfill(6)
    symbols = df.index.values.tolist()

    if update == '1':
        print("updating Daily data...")
        #updateStockData_CHN_Daily(root_path)

        print("updating CashFlow data...")
        update_all_stocks_data(root_path, symbols)

    print("Processing data...")
    process_all_stocks_data_1(root_path, symbols, update)


    
    

