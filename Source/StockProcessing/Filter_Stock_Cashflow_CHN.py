import sys, os, time, datetime, warnings, configparser
import pandas as pd
import numpy as np
import concurrent.futures
import tushare as ts
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

cur_path = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    root_path = cur_path[0:cur_path.rfind('/', 0, len(cur_path))]
    cur_path = root_path
sys.path.append(root_path + "/" + 'Source/FetchData/')
sys.path.append(root_path + "/" + 'Source/DataBase/')

#from Fetch_Data_Stock_CHN_Daily import updateStockData_CHN_Daily
from Fetch_Data_Stock_CHN_StockList import getStocksList_CHN
#from DB_API import queryStock

def get_single_stock_data_daily(root_path, symbol):
    '''
    All data is from quandl wiki dataset
    Feature set: [Open  High    Low  Close    Volume  Ex-Dividend  Split Ratio Adj. Open  Adj. High  Adj. Low
    Adj. Close  Adj. Volume]
    '''
    #df, lastUpdateTime = queryStock(root_path, "DB_STOCK", "SHEET_CHN", "_DAILY", symbol, "daily_update")
    try:
        df = ts.get_k_data(symbol)
        df.set_index('date', inplace=True)
        df.sort_index(ascending=True, inplace=True)
    except:
        print("stock delisted", symbol)
        return pd.DataFrame()

    if df.empty:
        print("stock delisted", symbol)
        return pd.DataFrame()

    out_path = root_path + "/Data/CSV/symbols/"

    if os.path.exists(out_path) == False:
        os.mkdir(out_path)

    out_file = out_path + symbol + ".csv"
    df.to_csv(out_file)

    return df


def get_single_stock_data_daily_date(root_path, symbol, start_date):
    '''
    All data is from quandl wiki dataset
    Feature set: [Open  High    Low  Close    Volume  Ex-Dividend  Split Ratio Adj. Open  Adj. High  Adj. Low
    Adj. Close  Adj. Volume]
    '''
    df = get_single_stock_data_daily(root_path, symbol)

    if df.empty: 
        #print("stock delisted", symbol)
        return pd.DataFrame(), pd.DataFrame(), []

    out_path = root_path + "/Data/CSV/cashflow/"
    exception_path = root_path + "/Data/CSV/exception/"
    
    if os.path.exists(out_path) == False:
        os.mkdir(out_path)
    if os.path.exists(exception_path) == False:
        os.mkdir(exception_path)

    out_file = out_path + symbol + ".csv"
    exception_file =exception_path + symbol + ".csv"

    df.index = pd.to_datetime(df.index)
    df = df[df.index >= pd.Timestamp(start_date)]
    datelist = df.index.strftime('%Y-%m-%d')

    if os.path.exists(exception_file):
        exception_df = pd.read_csv(exception_file, index_col=0)
        temp_df = exception_df[(exception_df['retry'] > 0)] #or (exception_df['last_update']=='True')]
        df_symbol_date_list = temp_df['date'].values.tolist()
        datelist = list( set(datelist) - set(df_symbol_date_list) )
    else:
        exception_df = pd.DataFrame(columns=['date', 'retry', 'last_update'])

    if os.path.exists(out_file) == False:
        df = pd.DataFrame(columns=['date', 'symbol', 'buy_amount', 'sell_amount', 'even_amount', 'buy_volume', 'sell_volume', 'even_volume', 'buy_max', 'buy_min', 'buy_average', 'sell_max', 'sell_min', 'sell_average', 'even_max', 'even_min', 'even_average'])
        df.index.name = 'index'
        return df, exception_df, datelist

    df = pd.read_csv(out_file, index_col=["index"])
    df_symbol_date_list = df['date'].values.tolist()
    #date_list = list( (set(date_list) | set(df_symbol_date_list)) - (set(date_list) & set(df_symbol_date_list)) )
    datelist = list( set(datelist) - set(df_symbol_date_list) )

    return df, exception_df, datelist


def group_tick_data_to_cashflow(df):
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


def updating_stock_tick_data(root_path, exception_df, symbol, date_list):
    file_path = root_path + "/Data/CSV/tick/" + symbol + "/"
    exception_file = root_path + "/Data/CSV/exception/" + symbol + ".csv"

    if os.path.exists(file_path) == False:
        os.mkdir(file_path)

    now_date = (datetime.datetime.now()).strftime("%Y-%m-%d")
    need_update_exception = False
    #pbar = trange(len(date_list), leave=False)
    #for i in pbar: 
    for date in date_list:
        #start = time.time()
        #date = date_list[i]

        new_file_name = file_path + symbol + "_" + date + ".csv"
        if os.path.exists(new_file_name):
            continue
        try:
            data = ts.get_tick_data(symbol ,date=date, src ='tt')
        except:
            print("stock:", symbol, " date:", date, "get data failed")
            
        if data is not None:
            data.to_csv(new_file_name)
        else:
            need_update_exception = True
            exception_df.loc[len(exception_df)] = [date, 1, now_date]
            # print("tick data", symbol, date, "is None")

        #outMessage = '%s processed in: %.4s seconds' % (date, (time.time() - start))
        #pbar.set_description(outMessage)

    if need_update_exception:
        exception_df = exception_df.groupby(["date"]).agg({'retry':'sum', 'last_update':'max'}).reset_index()
        exception_df.to_csv(exception_file)

    return not need_update_exception


def summary_stock_tick_data(root_path, df, symbol, date_list):
    file_path = root_path + "/Data/CSV/tick/" + symbol + "/"    
    out_file = root_path + "/Data/CSV/cashflow/" + symbol + ".csv"

    #pbar = trange(len(date_list), mininterval=0.1, smoothing=1, leave=False)
    #for i in pbar:
    for date in date_list:
        #date = date_list[i]
        start = time.time()
        file_name = file_path + symbol + "_" + date + ".csv"

        if os.path.exists(file_name) == False:
            continue

        try:
            data = pd.read_csv(file_name, index_col=0)
        except:
            print("error on symbol:", symbol, "  date:", date)
            continue

        if (data is None) or data.empty or len(data) < 4:
            buy, sell, even = 0, 0, 0
        else:
            buy_amount, sell_amount, even_amount, buy_volume, sell_volume, even_volume, buy_max, buy_min, buy_average, sell_max, sell_min, sell_average, even_max, even_min, even_average = group_tick_data_to_cashflow(data)
            df.loc[len(df)] = [date, symbol, buy_amount, sell_amount, even_amount, buy_volume, sell_volume, even_volume, buy_max, buy_min, buy_average, sell_max, sell_min, sell_average, even_max, even_min, even_average]

        #outMessage = '%s processed in: %.3s seconds' % (date, (time.time() - start))
        #pbar.set_description(outMessage)
        
    df = df.sort_values(['symbol','date'], ascending=[True, True])
    df.to_csv(out_file)


def parallel_processing(root_path, start_date, symbol):
    startTime = time.time()
    
    df, exception_df, date_list = get_single_stock_data_daily_date(root_path, symbol, start_date)

    if len(date_list) == 0: return startTime  

    need_update = updating_stock_tick_data(root_path, exception_df, symbol, date_list)
    
    if need_update: summary_stock_tick_data(root_path, df, symbol, date_list)

    return startTime


def update_all_stocks_data(root_path, symbols, start_date):
    pbar = tqdm(total=len(symbols))

    # for symbol in symbols:
    #     startTime = parallel_processing(root_path, start_date, symbol)
    #     outMessage = '%-*s processed in:  %.4s seconds' % (6, symbol, (time.time() - startTime))
    #     pbar.set_description(outMessage)
    #     pbar.update(1)
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Start the load operations and mark each future with its URL
        future_to_stock = {executor.submit(parallel_processing, root_path, start_date, symbol): symbol for symbol in symbols}
        for future in concurrent.futures.as_completed(future_to_stock):
            stock = future_to_stock[future]
            try:
                startTime = future.result()
            except Exception as exc:
                startTime = time.time()
                print('%r generated an exception: %s' % (stock, exc))
            outMessage = '%-*s processed in:  %.4s seconds' % (6, stock, (time.time() - startTime))
            pbar.set_description(outMessage)
            pbar.update(1)
    
    pbar.close()


def cal_stock_data(root_path, db_count, symbol, stock_memory, symbol_memory, day_range, index, range_len):
    startTime = time.time()

    if symbol in stock_memory:
        df_stock = stock_memory[symbol]
    else:
        symbol_file = root_path + "/Data/CSV/symbols/" + symbol + ".csv"
        if os.path.exists(symbol_file) == False:
            return startTime
        df_stock = pd.read_csv(symbol_file, index_col=["date"])
        df_stock.index = pd.to_datetime(df_stock.index)
        stock_memory[symbol] = df_stock

    if symbol in symbol_memory:
        df_symbol = symbol_memory[symbol]
    else:
        out_file = root_path + "/Data/CSV/cashflow/" + symbol + ".csv"
        
        if os.path.exists(out_file) == False:
            return startTime
            
        df_symbol = pd.read_csv(out_file, index_col=["index"])
        df_symbol = df_symbol.sort_values(['date'], ascending=[True])
        symbol_memory[symbol] = df_symbol

    if df_stock.empty or df_symbol.empty: 
        return startTime

    df_stock_min = df_stock[ df_stock.index.isin(day_range) ]
    df_symbol_min = df_symbol[ df_symbol['date'].isin(day_range) ]

    if len(df_stock_min) < (range_len+1) or len(df_symbol_min) < (range_len+1): 
        return startTime
    
    percentage = (float)("%.3f" % (100 * (df_stock_min['close'][-1] - df_stock_min['open'][-1]) / df_stock_min['close'][-1]))
    result = [symbol, percentage]

    for day in range(-2, -range_len-2, -1):
        buy_count = df_symbol_min[day:]['buy_amount'].sum()
        sell_count = df_symbol_min[day:]['sell_amount'].sum()
        even_count = df_symbol_min[day:]['even_amount'].sum()
        pure_buy = (buy_count - sell_count) / 10000
        pect_buy = 100 * pure_buy / ((buy_count + sell_count) / 10000)
        result.append((float)("%.2f" % pure_buy))
        result.append((float)("%.2f" % pect_buy))
        percentage = (float)("%.3f" % (100 * (df_stock_min['close'][-2] - df_stock_min['open'][day]) / df_stock_min['close'][day]))
        #result.append(str(percentage) + "%")    
        result.append(percentage)  

    
    db_count.loc[len(db_count)] = result
        
    return startTime

def process_all_stocks_data(root_path, symbols, day_range, stock_memory, symbol_memory, index, range_len):
    startTime_1 = time.time()
    filename = 'cashflow_count.csv'

    db_count = pd.DataFrame(columns=['symbol', '0-price', '1day', '1-pect', '1-price', '2day', '2-pect', '2-price', '3day', '3-pect', '3-price'])
    db_count.index.name = 'index'

    #pbar = tqdm(total=len(symbols))
    for symbol in symbols:
        startTime = cal_stock_data(root_path, db_count, symbol, stock_memory, symbol_memory, day_range, index, range_len)
        # outMessage = '%-*s processed in:  %.4s seconds' % (6, symbol, (time.time() - startTime))
        # pbar.set_description(outMessage)
        # pbar.update(1)

    #db_count.to_csv(filename)

    # else:
    #     db_count = pd.read_csv(filename, index_col=["index"], float_precision='round_trip')
    #pbar.close()
    #print('total processing in:  %.4s seconds' % ((time.time() - startTime_1)))

    return db_count

def get_result(filter_stock):
    if len(filter_stock) == 0: return 'N/A'
    filter_stock_negative = filter_stock[(filter_stock["0-price"] < 0.0)]
    result = [len(filter_stock_negative), len(filter_stock), 0.0]
    result[2] = round(result[0] / result[1], 2)
    return result

def filter_cashflow(db_count):
    # filename_1 = 'cashflow_count_filter_1.csv'
    # filename_2 = 'cashflow_count_filter_2.csv'
    # filename_3 = 'cashflow_count_filter_3.csv'
    # filename_4 = 'cashflow_count_filter_4.csv'
    # db_count_filter_1 = db_count[(db_count["1-pect"] < 0.0) & (db_count["2-pect"] < 0.0) & (db_count["3-pect"] < 0.0)]

    # db_count_filter_2 = db_count[(db_count["1-pect"] > 0.0) & (db_count["2-pect"] > 0.0) & (db_count["3-pect"] > 0.0)]

    # db_count_filter_3 = db_count_filter_1[(db_count_filter_1["3-price"].astype(float).fillna(0.0) > 0.0) & \
    #                                       (db_count_filter_1["2-price"].astype(float).fillna(0.0) > 0.0)]

    # db_count_filter_4 = db_count_filter_3[(db_count_filter_3["3-pect"] > db_count_filter_3["2-pect"]) & \
    #                                       (db_count_filter_3["2-pect"] > db_count_filter_3["1-pect"])]


    # db_count_filter_2 = db_count[(db_count["2-pect"].astype(float).fillna(0.0) > 0.0) & \
    #                              (db_count["3-pect"].astype(float).fillna(0.0) > 0.0)]


                                #  (db_count["3-price"].astype(float).fillna(0.0) > 0.0) & \
                                #  (db_count["2-price"].astype(float).fillna(0.0) > 0.0) & \
                                #  (db_count["1-price"].astype(float).fillna(0.0) > 0.0) & \
                                # (db_count["3-price"].astype(float).fillna(0.0) < db_count["2-price"].astype(float).fillna(0.0)) & \
    # db_count_filter_1 = db_count[(db_count["2-pect"].astype(float).fillna(0.0) > 0.0) & \
    #                              (db_count["3-pect"].astype(float).fillna(0.0) > 0.0) & \
    #                              (db_count["3-pect"] < db_count["2-pect"]) & \
    #                              (db_count["1-pect"] < db_count["2-pect"]) & \
    #                              (db_count["3-price"].astype(float).fillna(0.0) > 0.0) & \
    #                              (db_count["2-price"].astype(float).fillna(0.0) > 0.0) & \
    #                              (db_count["1-price"].astype(float).fillna(0.0) > 0.0) & \
    #                              (db_count["1day"].astype(float).fillna(0.0) > 0.0)]
    db_count_filter = db_count[(db_count["1-pect"] > 0.0)  & (db_count["2-pect"] > 0.0)  & (db_count["3-pect"] > 0.0)  & \
                               (db_count["1-price"] > 0.0) & (db_count["2-price"] > 0.0) & \
                               (db_count["2-price"] > db_count["1-price"]) & \
                               (db_count["1day"] > 0.0)]
    db_count_filter_1 = db_count_filter.sort_values(['0-price'], ascending=[False]).reset_index(drop=True)
        
    #print(db_count_filter_1)

    return db_count_filter_1


    # db_count_filter = db_count[((db_count["3-price"].astype(float).fillna(0.0)) < 0.0) & \
    #                            (abs(db_count["10-price"].astype(float).fillna(0.0)) < 5.0) & \
    #                            (abs(db_count["20-price"].astype(float).fillna(0.0)) < 5.0) & \
    #                            (abs(db_count["40-price"].astype(float).fillna(0.0)) < 5.0)]
    # db_count_filter_1.to_csv(filename_1)
    # db_count_filter_2.to_csv(filename_2)
    # db_count_filter_3.to_csv(filename_3)
    # db_count_filter_4.to_csv(filename_4)

def process_data(root_path, symbols, dates):
    
    negative_pect = {}
    stock_memory = {}
    symbol_memory = {}
    range_len = 3

    my_range = range(-1, -200, -1)
    #pbar = tqdm(total=len(my_range))
    pbar = trange(len(my_range))

    out_path = root_path + "/Data/CSV/target/"
    if os.path.exists(out_path) == False:
        os.mkdir(out_path)

    for index in my_range:
        day_range = [ dates[idx] for idx in range(index-range_len, index+1) ]
        file_name = out_path + day_range[-1] + ".csv"

        if os.path.exists(file_name):
            stock_filter = pd.read_csv(file_name, index_col=0)
        else:
            db_cashflow = process_all_stocks_data(root_path, symbols, day_range, stock_memory, symbol_memory, index, range_len)
            stock_filter = filter_cashflow(db_cashflow)

            if len(stock_filter) > 0: 
                stock_filter.to_csv(file_name)

        negative_pect[day_range[-1]] = get_result(stock_filter)

        # outMessage = '%-*s processed in:  %.4s seconds' % (6, index, (time.time() - startTime))
        # pbar.set_description(outMessage)    
        pbar.update(1)

    pbar.close()

    print(negative_pect)

def processing_sector_cashflow_count(root_path, symbols, dates):
    stock_info = ts.get_stock_basics()

    sector_columns = list(set(stock_info['industry'].values.tolist()))

    sector_count = pd.DataFrame(columns=sector_columns, index=dates)
    sector_count.index.name = 'date'
    sector_count = sector_count.fillna(0)

    pbar = tqdm(total=len(symbols))

    for symbol in symbols:
        startTime = time.time()
        out_file = root_path + "/Data/CSV/cashflow/" + symbol + ".csv"
        column = stock_info[stock_info.index == symbol]["industry"][0]

        if os.path.exists(out_file) == False:
            pbar.update(1)
            #print(symbol, column)
            continue
        
        df_symbol = pd.read_csv(out_file, index_col=["date"])
        df = df_symbol['buy_amount'] - df_symbol["sell_amount"]
        sector_count[column] = sector_count[column].add(df, fill_value=0)

        outMessage = '%-*s processed in:  %.4s seconds' % (6, symbol, (time.time() - startTime))
        pbar.set_description(outMessage)
        pbar.update(1)

    pbar.close()

    sector_count = sector_count.sort_index(ascending=False)
    sector_count.to_csv("cashflow_sector.csv")


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
    df = df.sort_index(ascending = True)
    symbols = df.index.values.tolist()

    sh = ts.get_k_data("sh")

    months = 13
    start_date = (datetime.datetime.now() - datetime.timedelta(days=months*30)).strftime("%Y-%m-%d")
    #start_date = sh['date'][0]

    if update == '1':
        print("Updating cashflow data...")
        update_all_stocks_data(root_path, symbols, start_date)

    # print("Processing cashflow sector data...")
    # processing_sector_cashflow_count(root_path, symbols, sh['date'])

    print("Processing data...")
    process_data(root_path, symbols, sh['date'].values.tolist())
    
