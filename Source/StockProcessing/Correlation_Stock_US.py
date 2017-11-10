import sys, os, time, datetime, warnings, configparser
import pandas as pd
import concurrent.futures
from itertools import combinations
from tqdm import tqdm

cur_path = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    root_path = cur_path[0:cur_path.rfind('/', 0, len(cur_path))]
    cur_path = root_path
sys.path.append(root_path + "/" + 'Source/DataBase/')

from DB_API import queryStockList, queryStock, queryCorrelation, storeCorrelation


def get_single_stock_data(root_path, symbol, dates_range):
    
    df, lastUpdateTime = queryStock(root_path, "DB_STOCK", "SHEET_US", "_DAILY", symbol, "daily_update")
    if df.empty: return pd.DataFrame()
    
    df.index = pd.to_datetime(df.index)
    df = df[df.index.isin(dates_range)].sort_index()
    df.loc[:, 'Close_Shift_1'] = df.loc[:, 'adj_close'].shift(1)
    
    df.loc[:, 'Return'] = df.loc[:, 'adj_close']/df.loc[:, 'Close_Shift_1'] - 1
    return df
    

def get_all_stocks_correlation(root_path, dates_range):
    df = queryCorrelation(root_path, "DB_STOCK", "SHEET_US_RELA")

    if df.empty == False: return df
    
    df = queryStockList(root_path, "DB_STOCK", "SHEET_US_DAILY")
    symbols = df.index.values.tolist()

    pbar = tqdm(total=len(symbols))

    stockData = []
    stockList = []
    print("get stock data...")
    # count = 500
    for symbol in symbols:
        startTime = time.time()
        df = get_single_stock_data(root_path, symbol, dates_range)
        if df.empty: continue
        stockData.append(df['Return'])
        stockList.append(symbol)
        outMessage = '%-*s fetched in:  %.4s seconds' % (12, symbol, (time.time() - startTime))
        pbar.set_description(outMessage)
        pbar.update(1)
        # count -= 1
        # if count == 0: break
    pbar.close()
    
    print("merge stock data...")
    startTime = time.time()
    df_returns = pd.concat(stockData, axis=1).fillna(0)
    df_returns.columns = stockList
    df_correlations = df_returns.corr()
    print('total processing in:  %.4s seconds' % ((time.time() - startTime)))

    print("cal correlationship...")
    startTime = time.time()
    pairwise_correlations = []
    stockCount = len(stockList)
    pbar = tqdm(total=stockCount*stockCount)
    for i in range(stockCount):
        for j in range(stockCount):
            if j > i:
                pairwise_correlations.append(df_correlations.iloc[i][j])
            pbar.set_description(str(i) + " " + str(j))
            pbar.update(1)

    print("arrange matrix...")
    us_company_pairs = combinations(stockList, 2)
    df_us_company_pairs = pd.DataFrame(list(us_company_pairs))
    df_us_company_pairs.columns = ['company1', 'company2']
    df_us_company_pairs.loc[:, 'correlation'] = pd.Series(pairwise_correlations).T
    df_us_company_pairs = df_us_company_pairs.sort_values(['correlation'], ascending=[False]).reset_index(drop=True)

    storeCorrelation(root_path, "DB_STOCK", "SHEET_US_RELA", df_us_company_pairs)

    print('total processing in:  %.4s seconds' % ((time.time() - startTime)))

    pbar.close()

    return df_us_company_pairs


if __name__ == "__main__":
    #np.seterr(divide='ignore', invalid='ignore')
    #np.set_printoptions(precision=3, suppress=True)
    pd.set_option('precision', 3)
    pd.set_option('display.width',1000)
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    now_date = datetime.datetime.now()
    start_date = (now_date - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
    end_date = now_date.strftime("%Y-%m-%d")

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
    
    df = get_all_stocks_correlation(root_path, pd.date_range(start_date, end_date))

    df_amd = df[df['company1'] == 'AMD'].reset_index(drop=True)
    print(df_amd.head(30))

    # if storeType == 1:
    #     # stop database server (sync)
    #     time.sleep(5)
    #     ShutdownServer()
    

 
