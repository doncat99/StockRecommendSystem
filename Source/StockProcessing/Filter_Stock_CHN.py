import sys, os, time, datetime, warnings, configparser
import pandas as pd
import numpy as np
import concurrent.futures
import tushare as ts
import talib
import matplotlib.pyplot as plt
from tqdm import tqdm

cur_path = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    root_path = cur_path[0:cur_path.rfind('/', 0, len(cur_path))]
    cur_path = root_path
sys.path.append(root_path + "/" + 'Source/FetchData/')
sys.path.append(root_path + "/" + 'Source/DataBase/')

from Fetch_Data_Stock_CHN_StockList import getStocksList_CHN
from Fetch_Data_Stock_CHN_Daily import updateStockData_CHN_Daily
from Fetch_Data_Stock_CHN_Weekly import updateStockData_CHN_Weekly
from Fetch_Data_Stock_CHN_Monthly import updateStockData_CHN_Monthly
from DB_API import queryStock

def get_single_stock_data_daily(root_path, symbol):
    '''
    All data is from quandl wiki dataset
    Feature set: [Open  High    Low  Close    Volume  Ex-Dividend  Split Ratio Adj. Open  Adj. High  Adj. Low
    Adj. Close  Adj. Volume]
    '''
    df, lastUpdateTime = queryStock(root_path, "DB_STOCK", "SHEET_CHN", "_DAILY", symbol, "daily_update")
    df.index = pd.to_datetime(df.index)

    suspended_day = pd.Timestamp((datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d"))

    if df.empty: 
        #print("stock delisted", symbol)
        return df

    if df.index[-1] < suspended_day:
        #print("stock suspended", symbol)
        return pd.DataFrame()

    return df


def get_single_stock_data_weekly(root_path, symbol):
    '''
    All data is from quandl wiki dataset
    Feature set: [Open  High    Low  Close    Volume  Ex-Dividend  Split Ratio Adj. Open  Adj. High  Adj. Low
    Adj. Close  Adj. Volume]
    '''
    df, lastUpdateTime = queryStock(root_path, "DB_STOCK", "SHEET_CHN", "_WEEKLY", symbol, "weekly_update")
    df.index = pd.to_datetime(df.index)

    #suspended_day = pd.Timestamp((datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d"))

    if df.empty: 
        #print("stock delisted", symbol)
        return df

    # if df.index[-1] < suspended_day:
    #     #print("stock suspended", symbol)
    #     return pd.DataFrame()

    return df


def get_single_stock_data_monthly(root_path, symbol):
    '''
    All data is from quandl wiki dataset
    Feature set: [Open  High    Low  Close    Volume  Ex-Dividend  Split Ratio Adj. Open  Adj. High  Adj. Low
    Adj. Close  Adj. Volume]
    '''
    df, lastUpdateTime = queryStock(root_path, "DB_STOCK", "SHEET_CHN", "_MONTHLY", symbol, "monthly_update")
    df.index = pd.to_datetime(df.index)

    #suspended_day = pd.Timestamp((datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d"))

    if df.empty: 
        #print("stock delisted", symbol)
        return df

    # if df.index[-1] < suspended_day:
    #     #print("stock suspended", symbol)
    #     return pd.DataFrame()

    return df

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

def corssover(input_1, input_2, index = -1):
    return (input_1[index] > input_2[index]) & (input_1[index-1] < input_2[index-1])

def ma_rule(df, type = 0, index = -1):
    default_parameters = [5, 10, 20, 30, 60, 120, 250]
    if type == 0:
        min_parameters, delta = 4, 0.15
    elif type == 1:
        min_parameters, delta = 4, 0.15
    else:
        min_parameters, delta = 3, 0.15

    len_cnt = len(df)
    ma_parameters = [item for item in default_parameters if item <= len_cnt]
    item_cnt = len(ma_parameters)

    if item_cnt < min_parameters: return False
    ma_names = ['ma'+str(item) for item in ma_parameters]

    try:
        if not set(ma_names).issubset(df.columns): 
            for idx, item in enumerate(ma_names):
                df[item] = df['close'].rolling(window=ma_parameters[idx], center=False).mean()
    except Exception as e: 
        print(e)
        return False

    ma_array = []
    for item in ma_names:
        ma_array.append(df[item][index])

    ma_array = sorted(ma_array)
    min_lines_required = min_parameters - 1

    for index in range(min_lines_required, item_cnt):
        if (ma_array[index] - ma_array[index - min_lines_required]) < delta:
            return True
    return False

def kdj_rule(df, index = -1):
    if len(df) < 2: return False

    try:
        if not {'kdj_k', 'kdj_d', 'kdj_j'}.issubset(df.columns): 
            df = KDJ(df)
    except Exception as e: 
        print(e)
        return False

    return corssover(df['kdj_j'], df['kdj_d']) & (df['kdj_d'][index] > df['kdj_d'][index-1]) & (df['kdj_d'][index] < 45)
    
def kdj_rule_1(df, index = -1):
    if len(df) < 2: return False

    try:
        if not {'kdj_k', 'kdj_d', 'kdj_j'}.issubset(df.columns): 
            df = KDJ(df)
    except Exception as e: 
        print(e)
        return False

    return (df['kdj_d'][index] < 45)

def kdj_rule_2(df, index = -1):
    if len(df) < 2: return False

    try:
        if not {'kdj_k', 'kdj_d', 'kdj_j'}.issubset(df.columns): 
            df = KDJ(df)
    except Exception as e: 
        print(e)
        return False

    return (df['kdj_j'][index] < df['kdj_d'][index]) & (df['kdj_j'][index-1] < df['kdj_d'][index-1]) & (df['kdj_j'][index-1] < df['kdj_j'][index]) & (df['kdj_d'][index] < 35)

def macd_rule(df, index = -1):
    try:  
        if not {'macd_dif', 'macd_dea', 'macd'}.issubset(df.columns):
            df = MACD(df)
    except Exception as e: 
        print(e)
        return False

    input_1 = -3
    input_2 = 0.05
    
    # df['macd_dif_1'] = df['macd_dif'].shift(1)
    # df['macd_dea_1'] = df['macd_dea'].shift(1)

#(df['macd_dif'][-input_3:].min() < input_2) & \

    return (df['macd_dif'][index] > input_1) & \
           (df['macd_dif'][index] < input_2) & \
           (df['macd_dif'][index] > df['macd_dea'][index]) & \
           ((df['macd_dea'][index-1] > df['macd_dif'][index-1]) | (abs(df['macd_dea'][index-1] - df['macd_dif'][index-1]) < 0.007))

def macd_rule_1(df, index = -1):
    try:  
        if not {'macd_dif', 'macd_dea', 'macd'}.issubset(df.columns):
            df = MACD(df)
    except Exception as e: 
        print(e)
        return False

    return (df['macd_dif'][index] > df['macd_dea'][index]) & \
           ((df['macd_dea'][index-1] > df['macd_dif'][index-1]) | (abs(df['macd_dea'][index-1] - df['macd_dif'][index-1]) < 0.007))

def macd_rule_2(df, index = -1):
    try:  
        if not {'macd_dif', 'macd_dea', 'macd'}.issubset(df.columns):
            df = MACD(df)
    except Exception as e: 
        print(e)
        return False

    input = 0.05

    return (df['macd_dif'][index] < input) & (df['macd_dea'][index] < input) 

def rsi_rule(df, index = -1):
    try:  
        df = RSI(df, 6)
        df = RSI(df, 12)
        df = RSI(df, 24)
    except Exception as e: 
        print(e)
        return False

    rsi_6, rsi_12, rsi_24 = df['rsi_6'][index], df['rsi_12'][index], df['rsi_24'][index]

    return (rsi_6 < 20) & (rsi_12 < 20) & (rsi_24 < 30)
    

def judge_rule_daily(symbol, dataset, window, selection):
    #if (kdj_rule(dataset) | kdj_rule(dataset, -2)) & (macd_rule(dataset) | macd_rule(dataset, -2)):
    #if (kdj_rule(dataset) | kdj_rule(dataset, -2)) & ma_rule(dataset):
    #if macd_rule(dataset):
    if kdj_rule_2(dataset) & macd_rule_2(dataset) & rsi_rule(dataset):
        selection.append(symbol)

def judge_rule_weekly(symbol, dataset, window, selection):
    #if (kdj_rule(dataset) | kdj_rule(dataset, -2)) & (macd_rule(dataset) | macd_rule(dataset, -2)):
    if ma_rule(dataset) & kdj_rule_1(dataset):
    #if ma_rule(dataset):
        selection.append(symbol)

def judge_rule_monthly(symbol, dataset, window, selection):
    if ma_rule(dataset) & kdj_rule_1(dataset):
        selection.append(symbol)

def inner_processing_stock_data(symbol, input_data, window, day_selection, week_selection, month_selection):
    day_data =  input_data['daily'] #input_data[input_data['volume'] > 0].copy()
    # week_data = input_data['weekly'] #convert_week_based_data(day_data)
    # month_data = input_data['monthly'] #convert_month_based_data(day_data)

    judge_rule_daily(symbol, day_data, window, day_selection)
    # judge_rule_weekly(symbol, week_data, window, week_selection)
    # judge_rule_monthly(symbol, month_data, window, month_selection)


def processing_stock_data(root_path, symbol, window, day_selection, week_selection, month_selection):
    startTime = time.time()
    data_daily = get_single_stock_data_daily(root_path, symbol)
    data_weekly = get_single_stock_data_weekly(root_path, symbol)
    data_monthly = get_single_stock_data_monthly(root_path, symbol)

    if data_daily.empty: return startTime
    if len(data_daily) < 60 + window: return startTime

    data = { "daily":data_daily, "weekly":data_weekly, "monthly":data_monthly }
    
    inner_processing_stock_data(symbol, data, window, day_selection, week_selection, month_selection)

    return startTime

def process_all_stocks_data(root_path, window = 1):
    df = getStocksList_CHN(root_path)
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

    startTime_1 = time.time()
    for symbol in symbols:
        startTime = processing_stock_data(root_path, symbol, window, day_selection, week_selection, month_selection)
        outMessage = '%-*s processed in:  %.4s seconds' % (6, symbol, (time.time() - startTime))
        pbar.set_description(outMessage)
        pbar.update(1)

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

    # day_week_selection = []
    # week_month_selection = []
    # day_month_selection = []
    # all_selection = []

    #count = []

    day_week_selection   = list(set(day_selection)      & set(week_selection      ))
    week_month_selection = list(set(week_selection)     & set(month_selection     ))
    day_month_selection  = list(set(day_selection)      & set(month_selection     ))
    all_selection        = list(set(day_week_selection) & set(week_month_selection))

    print("all_selection", len(all_selection), sorted(all_selection))
    print("day_week_selection", len(day_week_selection), sorted(day_week_selection))
    print("week_month_selection", len(week_month_selection), sorted(week_month_selection))
    print("day_month_selection", len(day_month_selection), sorted(day_month_selection))
    print("/n ------------------------ /n")
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
    if len(sys.argv) != 2:
        print("Input parameter error")
        exit()

    pd.set_option('precision', 3)
    pd.set_option('display.width',1000)
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    update = str(sys.argv[1])
    
    if update == '1':
        print("updating Daily data...")
        updateStockData_CHN_Daily(root_path)

        print("updating Weekly data...")
        updateStockData_CHN_Weekly(root_path)

        print("updating Monthly data...")
        updateStockData_CHN_Monthly(root_path)
    

    print("Processing data...")
    #calBasic()
    process_all_stocks_data(root_path)

