import sys, os, time, datetime, warnings, configparser
import pandas as pd
import concurrent.futures
from itertools import combinations
from tqdm import tqdm

cur_path = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    root_path = cur_path[0:cur_path.rfind('/', 0, len(cur_path))]
    cur_path = root_path
sys.path.append(root_path + "/" + 'Source/FetchData/')
sys.path.append(root_path + "/" + 'Source/DataBase/')

from Fetch_Data_Stock_US_Daily import updateStockData_US, getStocksList
from DB_API import queryStock, queryCoorelation, storeCoorelation


def get_single_stock_data(root_path, symbol, dates_range):
    
    df, lastUpdateTime = queryStock(root_path, "STOCK_US", symbol)
    if df.empty: return pd.DataFrame()
    
    df.index = pd.to_datetime(df.index)
    #df = pd.read_csv(filename, index_col=["Date"], parse_dates=['Date'], usecols=['Date', 'Adj Close'])
    df = df[df.index.isin(dates_range)].sort_index()
    df.loc[:, 'Close_Shift_1'] = df.loc[:, 'Adj Close'].shift(1)
    
    df.loc[:, 'Return'] = df.loc[:, 'Adj Close']/df.loc[:, 'Close_Shift_1'] - 1
    return df
    


def get_all_stocks_correlation(root_path, dates_range):
    startTime = time.time()

    try:
        df = queryCoorelation(root_path, "COORELATION_US")
    except:
        df = pd.DataFrame()

    if df.empty == False: return df

    symbols = getStocksList(root_path)['Symbol'].values.tolist()

    stockData = []
    stockList = []
    print("get stock data...")
    #count = 10
    for symbol in symbols:
        df = get_single_stock_data(root_path, symbol, dates_range)
        if df.empty: continue
        stockData.append(df['Return'])
        stockList.append(symbol)
        #count -= 1
        #if count == 0: break
    
    print("merge stock data...")
    df_returns = pd.concat(stockData, axis=1).fillna(0)
    df_returns.columns = stockList
    df_correlations = df_returns.corr()

    print("cal correlationship...")
    pairwise_correlations = []
    stockCount = len(stockList)
    pbar = tqdm(total=stockCount*stockCount)
    for i in range(stockCount):
        for j in range(stockCount):
            if j > i:
                pairwise_correlations.append(df_correlations.iloc[i][j])
            pbar.set_description(str(i) + " " + str(j))
            pbar.update(1)

    us_company_pairs = combinations(stockList, 2)
    df_us_company_pairs = pd.DataFrame(list(us_company_pairs))
    df_us_company_pairs.columns = ['Company1', 'Company2']
    df_us_company_pairs.loc[:, 'Correlation'] = pd.Series(pairwise_correlations).T
    df_us_company_pairs = df_us_company_pairs.sort_values(['Correlation'], ascending=[False]).reset_index(drop=True)#.reset_index(drop=True)

    storeCoorelation(root_path, "COORELATION_US", df_us_company_pairs)
    
    #print(df_us_company_pairs.head(30))

    print('total processing in:  %.4s seconds' % ((time.time() - startTime)))

    return df_us_company_pairs


if __name__ == "__main__":
    #np.seterr(divide='ignore', invalid='ignore')
    #np.set_printoptions(precision=3, suppress=True)
    pd.set_option('precision', 3)
    pd.set_option('display.width',1000)
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    start_date = "2016-01-03"
    end_date = "2017-06-10"
    
    config = configparser.ConfigParser()
    config.read(root_path + "/" + "config.ini")
    storeType = int(config.get('Setting', 'StoreType'))

    if storeType == 1:
        from Start_DB_Server import StartServer, ShutdownServer
        # start database server (async)
        thread = StartServer(root_path)
        
        # wait for db start, the standard procedure should listen to 
        # the completed event of function "StartServer"
        time.sleep(5)
    
    df = get_all_stocks_correlation(root_path, pd.date_range(start_date, end_date))

    df_amd = df[df['Company1'] == 'AMD'].reset_index(drop=True)
    print(df_amd.head(30))

    if storeType == 1:
        # stop database server (sync)
        time.sleep(5)
        ShutdownServer()
    print("Processing data...")
    

 
