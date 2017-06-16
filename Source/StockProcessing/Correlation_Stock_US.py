import sys
sys.path.append('../FetchData/')

import os, time, datetime, warnings, configparser
import pandas as pd
import concurrent.futures
from itertools import combinations
from tqdm import tqdm

from Fetch_Data_Stock_US_Daily import updateStockData_US, getStocksList


def get_single_stock_data(ticker, dates_range, stock_folder):
    filename = stock_folder + ticker + '.csv'
    if os.path.exists(filename) == False: 
        return pd.DataFrame()

    df = pd.read_csv(filename, index_col=["Date"], parse_dates=['Date'], usecols=['Date', 'Adj Close'])
    df = df[df.index.isin(dates_range)].sort_index()
    df.loc[:, 'Close_Shift_1'] = df.loc[:, 'Adj Close'].shift(1)
    df.loc[:, 'Return'] = df.loc[:, 'Adj Close']/df.loc[:, 'Close_Shift_1'] - 1
    return df
    


def get_all_stocks_data(dates_range):
    Config = configparser.ConfigParser()
    Config.read("../../config.ini")
    dir_stock = Config.get('Paths', 'STOCK_US')
    dir_result = Config.get('Paths', 'RESULT_COORELATION')

    if os.path.exists(dir_stock) == False: 
        os.makedirs(dir_stock)

    if os.path.exists(dir_result) == False: 
        os.makedirs(dir_result)

    startTime = time.time()

    symbols = getStocksList()['Symbol'].values.tolist()

    stockData = []
    stockList = []
    print("get stock data...")
    #count = 10
    for symbol in symbols:
        df = get_single_stock_data(symbol, dates_range, dir_stock)
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
    df_us_company_pairs = df_us_company_pairs.sort_values(['Correlation'], ascending=[False])

    filename = dir_result + "us_company_coorelation.csv"
    df_us_company_pairs.to_csv(filename)
    
    print(df_us_company_pairs.head(30))

    print('total processing in:  %.4s seconds' % ((time.time() - startTime)))

    # startTime = time.time()
    # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    #     # Start the load operations and mark each future with its URL
    #     future_to_stock = {executor.submit(processing_stock_data, dir, stock, week_selection, month_selection): stock for stock in stocklist}
    #     for future in concurrent.futures.as_completed(future_to_stock):
    #         stock = future_to_stock[future]
    #         try:
    #             subStartTime = future.result()
    #         except Exception as exc:
    #             print('%r generated an exception: %s' % (stock, exc))
    #         else:
    #             outMessage = '%-*s processing in:  %.4s seconds' % (6, stock, (time.time() - subStartTime))
    #             print(outMessage)
    # print('total processing in:  %.4s seconds' % ((time.time() - startTime)))

    return


if __name__ == "__main__":
    #np.seterr(divide='ignore', invalid='ignore')
    #np.set_printoptions(precision=3, suppress=True)
    pd.set_option('precision', 3)
    pd.set_option('display.width',1000)
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    start_date = "2016-01-03"
    end_date = "2017-06-10"
    
    print("Processing data...")
    get_all_stocks_data(pd.date_range(start_date, end_date))
 
