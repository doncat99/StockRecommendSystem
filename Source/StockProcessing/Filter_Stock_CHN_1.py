import sys, os, time, datetime, warnings, configparser
import pandas as pd
import numpy as np
import talib
import concurrent.futures
import tushare as ts
import matplotlib.pyplot as plt
from stockstats import StockDataFrame as Sdf
from tqdm import tqdm

cur_path = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    root_path = cur_path[0:cur_path.rfind('/', 0, len(cur_path))]
    cur_path = root_path
sys.path.append(root_path + "/" + 'Source/FetchData/')
sys.path.append(root_path + "/" + 'Source/DataBase/')

from Fetch_Data_Stock_CHN_Daily import updateStockData_CHN_Daily
from DB_API import queryStock, queryStockList

def KDJ(df):
    low_list = df['low'].rolling(center=False,window=9).min()
    low_list.fillna(value=df['low'].expanding(min_periods=1).min(), inplace=True)
    high_list = df['high'].rolling(center=False,window=9).max()
    high_list.fillna(value=df['high'].expanding(min_periods=1).max(), inplace=True)
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    df['kdj_k'] = rsv.ewm(min_periods=0,adjust=True,ignore_na=False,com=2).mean()
    df['kdj_d'] = df['kdj_k'].ewm(min_periods=0,adjust=True,ignore_na=False,com=2).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    return df

def RSI(df, n=14):
    prices = df['close'].values.tolist()
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    key = 'rsi_' + str(n)
    df[key] = rsi
    return df

def MACD(df, short_win=12, long_win=26, macd_win=9):
    # talib计算MACD
    prices = np.array(df['close'])
    macd_tmp = talib.MACD(prices, fastperiod=short_win, slowperiod=long_win, signalperiod=macd_win)
    df['macd_dif'] = macd_tmp[0]
    df['macd_dea'] = macd_tmp[1]
    df['macd'] = macd_tmp[2]
    return df


def corssover(input_1, input_2):
    index = -1
    return (input_1[index] > input_2[index]) & (input_1[index-1] < input_2[index-1])

def ma_rule(df):
    df['ma5']   = df['close'].rolling(window=5, center=False).mean()
    df['ma10']  = df['close'].rolling(window=10, center=False).mean()
    df['ma20']  = df['close'].rolling(window=20, center=False).mean()
    df['ma30']  = df['close'].rolling(window=30, center=False).mean()
    df['ma60']  = df['close'].rolling(window=60, center=False).mean()
    df['ma120'] = df['close'].rolling(window=120, center=False).mean()
    df['ma250'] = df['close'].rolling(window=250, center=False).mean()

    index = -1
    fit_count = 0
    delta = 0.05 #ma5 / 60

    ma5, ma10, ma20, ma30, ma60, ma120, ma250 = df['ma5'][index], df['ma10'][index], df['ma20'][index], df['ma30'][index], df['ma60'][index], df['ma120'][index], df['ma250'][index]

    if abs(ma5 - ma10)  < delta: fit_count += 1
    if abs(ma5 - ma20)  < delta: fit_count += 1
    if abs(ma5 - ma30)  < delta: fit_count += 1
    if abs(ma5 - ma60)  < delta: fit_count += 1
    if abs(ma5 - ma120) < delta: fit_count += 1
    if abs(ma5 - ma250) < delta: fit_count += 1

    return fit_count > 4

def kdj_rule(df):
    try: df = KDJ(df)
    except: return False

    if len(df) < 2: return False
    index = -1
    return corssover(df['kdj_j'], df['kdj_d']) & (df['kdj_d'][index] > df['kdj_d'][index-1])
    
def kdj_rule_1(df):
    try: df = KDJ(df)
    except: return False

    return df['kdj_d'][-1] < 20

def macd_rule(df):
    try:  df = MACD(df)
    except: return False

    input_1 = 0.2
    input_2 = -0.8
    input_3 = 22 * 3
    index = -1
    df['macd_dif_1'] = df['macd_dif'].shift(1)
    df['macd_dea_1'] = df['macd_dea'].shift(1)

    return (abs(df['macd_dea'][index]) < input_1) & \
           (abs(df['macd_dif'][index]) < input_1) & \
           (df['macd_dif'][-input_3:].min() < input_2) & \
           (df['macd_dif'][index] > df['macd_dea'][index]) & \
           ((df['macd_dea_1'][index] > df['macd_dif_1'][index]) | (abs(df['macd_dea_1'][index] - df['macd_dif_1'][index]) < 0.007))

def macd_rule_1(df):
    try:  df = MACD(df)
    except: return False

    input_1 = 0
    input_2 = -0.8
    input_3 = 0.05

    dif_len = len(df['macd_dif'])
    if dif_len < 2: return False

    if abs(df['macd_dif'][-1]) > input_3:
        return False

    for idx in range(dif_len-1, 1, -1):
        if ((df['macd_dif'][idx] - df['macd_dif'][idx-1]) > input_1):
            continue

        if df['macd_dif'][idx] <= input_2:
            return True
        else: return False
        
def macd_rule_2(df, symbol):
    try:  df = MACD(df)
    except: return False

    input_1 = -3
    input_2 = -0.2

    index = -1

    return (df['macd_dif'][index] > input_1) & \
           (df['macd_dif'][index] < input_2) & \
           (df['macd_dif'][index] > df['macd_dea'][index]) & \
           ((df['macd_dea'][index-1] > df['macd_dif'][index-1]) | (abs(df['macd_dea'][index-1] - df['macd_dif'][index-1]) < 0.007))


def rsi_rule(df):
    try:  
        df = RSI(df, 6)
        df = RSI(df, 12)
        df = RSI(df, 24)
    except: return False

    index = -1
    rsi_6, rsi_12, rsi_24 = df['rsi_6'][index], df['rsi_12'][index], df['rsi_24'][index]

    return (rsi_6 < 20) & (rsi_12 < 20) & (rsi_24 < 30)
    

def judge_rule(symbol, dataset, window, selection, str):
    #if kdj_rule(dataset) & macd_rule(dataset):
    if kdj_rule_1(dataset):
        selection.append(symbol)

def get_single_stock_data(root_path, symbol):
    '''
    All data is from quandl wiki dataset
    Feature set: [Open  High    Low  Close    Volume  Ex-Dividend  Split Ratio Adj. Open  Adj. High  Adj. Low
    Adj. Close  Adj. Volume]
    '''
    # file_name = stock_folder + ticker + '.csv'
    # COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    # RENAME_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume']

    # if os.path.exists(file_name) == False: 
    #     print("get stock: " + ticker + " failed")
    #     return pd.DataFrame()

    # df = pd.read_csv(
    #     file_name,
    #     #names=COLUMNS,
    #     skipinitialspace=True,
    #     engine='python',
    #     index_col=['Date'],
    #     #usecols=COLUMNS,
    #     parse_dates=['Date'],
    #     #skiprows=1,
    #     memory_map=True,
    #     #chunksize=300,
    # ).sort_index()
    df, lastUpdateTime = queryStock(root_path, "DB_STOCK", "SHEET_CHN", "_DAILY", symbol, "daily_update")
    df.index = pd.to_datetime(df.index)

    suspended_day = pd.Timestamp((datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d"))

    if df.empty: 
        #print("stock delisted", symbol)
        return df

    if df.index[-1] < suspended_day:
        #print("stock suspended", symbol)
        return pd.DataFrame()

    if 'adj_close' in df:
        df = df.drop('close', 1)
        df = df.rename(columns = {'adj_close':'close'})

    return df

def inner_processing_stock_data(symbol, input_data, window, day_selection, week_selection, month_selection):
    # start_date = pd.Timestamp(paras.start_date)
    # end_date   = pd.Timestamp(paras.end_date)
    # input_data = input_data.loc[(input_data.index >= start_date) & (input_data.index <= end_date)]
    day_data = input_data[input_data['volume'] > 0].copy()
    #week_data = convert_week_based_data(day_data)
    #month_data = convert_month_based_data(day_data)

    judge_rule(symbol, day_data, window, day_selection, "day based")
    #judge_rule(symbol, week_data, window, week_selection, "week based")
    #judge_rule(symbol, month_data, window, month_selection, "month based")


def processing_stock_data(root_path, symbol, window, day_selection, week_selection, month_selection):
    startTime = time.time()
    data = get_single_stock_data(root_path, symbol)
    if data.empty: return startTime
    if len(data) < 60 + window: return startTime
    
    inner_processing_stock_data(symbol, data, window, day_selection, week_selection, month_selection)

    return startTime

def process_all_stocks_data(root_path, window = 1):
    df = queryStockList(root_path, "DB_STOCK", "SHEET_CHN_DAILY")
    df.index = df.index.astype(str).str.zfill(6)
    symbols = df.index.values.tolist()

    pbar = tqdm(total=len(symbols))

    day_selection = []
    week_selection = []
    month_selection = []

    # for index in range(0, window):
    #     day_window = []
    #     day_selection.append(day_window)
    #     week_window = []
    #     week_selection.append(week_window)
    #     month_window = []
    #     month_selection.append(month_window)

    startTime = time.time()
    for symbol in symbols:
        startTime = processing_stock_data(root_path, symbol, window, day_selection, week_selection, month_selection)
        outMessage = '%-*s processed in:  %.4s seconds' % (6, symbol, (time.time() - startTime))
        pbar.set_description(outMessage)
        pbar.update(1)
    print('total processing in:  %.4s seconds' % ((time.time() - startTime)))

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

    # day_week_selection = []
    # week_month_selection = []
    # day_month_selection = []
    # all_selection = []

    #count = []

    day_week_selection   = list(set(day_selection)      & set(week_selection      ))
    week_month_selection = list(set(week_selection)     & set(month_selection     ))
    day_month_selection  = list(set(day_selection)      & set(month_selection     ))
    all_selection        = list(set(day_week_selection) & set(week_month_selection))

        #day_selection = list(set(day_selection) - set(all_selection))
        #week_selection = list(set(week_selection) - set(all_selection))
        #month_selection = list(set(month_selection) - set(all_selection))

        # sumUp = len(day_week_selection[index]) + len(week_month_selection[index]) + len(day_month_selection[index]) + len(all_selection[index])
        # count.insert(0,sumUp)

    print("all_selection", len(all_selection), sorted(all_selection))
    print("day_week_selection", len(day_week_selection), sorted(day_week_selection))
    print("week_month_selection", len(week_month_selection), sorted(week_month_selection))
    print("day_month_selection", len(day_month_selection), sorted(day_month_selection))
    print("/n ------------------------ /n")

    # plt.plot(range(0, len(count)), count)
    # plt.title('A simple chirp')
    # plt.show()
    print("day_selection", len(day_selection), sorted(day_selection))
    print("week_selection", len(week_selection), sorted(week_selection))
    print("month_selection", len(month_selection), sorted(month_selection))

def calBasic():
    pe = 40
    gpr = 30  # 毛利率
    npr = 15  # 净利率
    nav = 20
    roe = 40  # 净资产收益率 三年

    df_base = ts.get_stock_basics()
    baseData = pd.DataFrame(df_base)

    baseData = baseData[(baseData.pe < pe) & (baseData.gpr > gpr) & (baseData.npr > npr)]
    baseData = baseData.index.values.tolist()

    years = [2016, 2015, 2014]
    main_symbols = []
    grow_symbols = []

    if os.path.exists("year_2016.csv") == False:
        df_main = ts.get_report_data(2016, 4)
        mainData = pd.DataFrame(df_main)
        mainData.to_csv("year_2016.csv")

    if os.path.exists("grow_2016.csv") == False:
        df_main = ts.get_growth_data(2016, 4)
        mainData = pd.DataFrame(df_main)
        mainData.to_csv("grow_2016.csv")
        
    if os.path.exists("year_2015.csv") == False:
        df_main = ts.get_report_data(2015, 4)
        mainData = pd.DataFrame(df_main)
        mainData.to_csv("year_2015.csv")

    if os.path.exists("grow_2015.csv") == False:
        df_main = ts.get_growth_data(2015, 4)
        mainData = pd.DataFrame(df_main)
        mainData.to_csv("grow_2015.csv")

    if os.path.exists("year_2014.csv") == False:
        df_main = ts.get_report_data(2014, 4)
        mainData = pd.DataFrame(df_main)
        mainData.to_csv("year_2014.csv")

    if os.path.exists("grow_2014.csv") == False:
        df_main = ts.get_growth_data(2014, 4)
        mainData = pd.DataFrame(df_main)
        mainData.to_csv("grow_2014.csv")

    for year in years:
        mainData = pd.read_csv("year_" + str(year) + ".csv")
        mainData = mainData[mainData.roe > roe]
        main_symbols.append(mainData.code.values.tolist())

    for year in years:
        mainData = pd.read_csv("grow_" + str(year) + ".csv")
        mainData = mainData[mainData.nav > nav]
        grow_symbols.append(mainData.code.values.tolist())

    roe_list = list(set(main_symbols[0]) & set(main_symbols[1]) & set(main_symbols[2]))
    roe_list = [str(item).zfill(6) for item in roe_list]

    nav_list = list(set(grow_symbols[0]) & set(grow_symbols[1]) & set(grow_symbols[2]))
    nav_list = [str(item).zfill(6) for item in nav_list]

    output = list(set(roe_list) & set(nav_list) & set(baseData))
    print(output)
    
    
if __name__ == "__main__":
    pd.set_option('precision', 3)
    pd.set_option('display.width',1000)
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    now = datetime.datetime.now().strftime("%Y-%m-%d")

    config = configparser.ConfigParser()
    config.read(root_path + "/" + "config.ini")
    storeType = int(config.get('Setting', 'StoreType'))

    # if storeType == 1:
    #     from Start_DB_Server import StartServer, ShutdownServer
    #     # start database server (async)
    #     thread = StartServer(root_path)
        
    #     # wait for db start, the standard procedure should listen to 
    #     # the completed event of function "StartServer"
    #     time.sleep(5)
    
    print("updating data...")
    #updateStockData_CHN_Daily(root_path, storeType)
    
    print("Processing data...")
    #calBasic()
    process_all_stocks_data(root_path)

    # if storeType == 1:
    #     # stop database server (sync)
    #     time.sleep(5)
    #     ShutdownServer()

