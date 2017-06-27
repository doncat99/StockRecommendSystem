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
from DB_API import queryStock, storeStock, queryStockList, storeStockList, queryStockPublishDay, storePublishDay

def getStocksList(root_path):
    try:
        df = queryStockList(root_path, "STOCK_CHN")
        df.index = df.index.astype(str).str.zfill(6)
    except Exception as e:
        df = pd.DataFrame()

    if df.empty == False: return df
    
    stock_info = ts.get_stock_basics()
    listData = pd.DataFrame(stock_info)
    #listData.index.name = 'Symbol'
    #listData.index = listData.index.astype(str).str.zfill(6) #[str(symbol).zfill(6) for symbol in listData.index] #listData.index.astype(str).str.zfill(6)
    #print(listData.index)
    #listData['Symbol'] = listData['Symbol'].str.strip()
    storeStockList(root_path, "STOCK_CHN", listData)
    df = queryStockList(root_path, "STOCK_CHN")
    df.index = df.index.astype(str).str.zfill(6)
    return df

def getSingleStock(symbol):
    repeat_times = 1
    message = ""
    df = pd.DataFrame()

    for _ in range(repeat_times): 
        try:
            data = ts.get_hist_data(symbol)
            data = data.rename(columns = {'date':'Date'})
            data.sort_index(ascending=True, inplace=True)
            return data, ""
        except Exception as e:
            message = symbol + " fetch exception: " + str(e)
            continue   
    return df, message

def getSingleStockByTime(symbol, from_date, till_date):
    start = from_date.split('-')
    start_y, start_m, start_d = start[0], start[1], start[2] # starting date

    end = till_date.split('-')
    end_y, end_m, end_d = end[0], end[1], end[2] # until now
    
    repeat_times = 1
    message = ""
    df = pd.DataFrame()

    for _ in range(repeat_times): 
        try:
            data = ts.get_hist_data(symbol, from_date, till_date)
            data = data.rename(columns = {'date':'Date'})
            data.sort_index(ascending=True, inplace=True)
            return data, ""
        except Exception as e:
            message = symbol + " fetch exception: " + str(e)
            continue   
    return df, message

def judgeOpenDaysInRange(from_date, to_date):
    holidays=["2017-01-01", "2017-01-02",
              "2017-01-27", "2017-01-28", "2017-01-29", "2017-01-30", "2017-01-31", "2017-02-01", "2017-02-02",
              "2017-04-02", "2017-04-03", "2017-04-04",
              "2017-05-01",
              "2017-05-28", "2017-05-29", "2017-05-30",
              "2017-10-01", "2017-10-02", "2017-10-03", "2017-10-04", "2017-10-05","2017-10-06","2017-10-07","2017-10-08"]

    #holidays = cal.holidays(from_date, to_date)
    duedays = pd.bdate_range(from_date, to_date)
    df = pd.DataFrame()
    df['date'] = duedays
    df['Holiday'] = duedays.isin(holidays)
    opendays = df[df['Holiday'] == False]
    return opendays

def judgeNeedPostDownload(from_date, to_date):
    today = datetime.datetime.now()
    start_date = pd.Timestamp(from_date)
    end_date = pd.Timestamp(to_date)

    if start_date > today: return False    
    if end_date > today: to_date = today.strftime("%Y-%m-%d")
    dateList = judgeOpenDaysInRange(from_date, to_date)
    if len(dateList) > 0: return True
    return False


def updateSingleStockData(root_path, symbol, force_check):
    startTime = time.time()
    message = ""

    if len(symbol) == 0: return startTime, message

    till_date = (datetime.datetime.now()).strftime("%Y-%m-%d")
    end_date  = pd.Timestamp(till_date)
    
    stockData, lastUpdateTime = queryStock(root_path, "STOCK_CHN", symbol)

    if stockData.empty:
        stockData, message = getSingleStock(symbol)
        if stockData.empty == False:
            storeStock(root_path, "STOCK_CHN", symbol, stockData)
        return startTime, message

    modified = False
    first_date = pd.Timestamp(stockData.index[0])
    last_date  = pd.Timestamp(stockData.index[-1])
    updateOnce = end_date > lastUpdateTime
     
    if end_date > last_date and (updateOnce or force_check):
        to_date = (last_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        if judgeNeedPostDownload(to_date, till_date):
            message = message + ", download post data from " + to_date + " to " + till_date
            moreStockData, tempMessage = getSingleStockByTime(symbol, to_date, till_date)
            message = message + tempMessage
            if len(moreStockData) > 0:
                if isinstance(moreStockData.index, pd.DatetimeIndex):
                    moreStockData.index = moreStockData.index.strftime("%Y-%m-%d")
                modified = True
                stockData = pd.concat([stockData, moreStockData])
                stockData.index.name = 'date'
        
    if modified:
        stockData = stockData[~stockData.index.duplicated(keep='first')]
        storeStock(root_path, "STOCK_CHN", symbol, stockData)
    elif updateOnce:
        stockData = stockData[~stockData.index.duplicated(keep='first')]
        storeStock(root_path, "STOCK_CHN", symbol, stockData)
        message = message + ", nothing updated"
    else:
        message = ""

    return startTime, message

def updateStockData_CHN(root_path, force_check = False):

    symbols = getStocksList(root_path).index.values.tolist()

    pbar = tqdm(total=len(symbols))
    log_errors = []
    log_update = []

    # debug only
    # for symbol in symbols:
    #     startTime, message = updateSingleStockData(root_path, symbol, force_check)
    #     outMessage = '%-*s fetched in:  %.4s seconds' % (6, symbol, (time.time() - startTime))
    #     pbar.set_description(outMessage)
    #     pbar.update(1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Start the load operations and mark each future with its URL
        future_to_stock = {executor.submit(updateSingleStockData, root_path, symbol, force_check): symbol for symbol in symbols}
        for future in concurrent.futures.as_completed(future_to_stock):
            stock = future_to_stock[future]
            try:
                startTime, message = future.result()
            except Exception as exc:
                startTime = time.time()
                log_errors.append('%r generated an exception: %s' % (stock, exc))
            else:
                if len(message) > 0: log_update.append(message)
            outMessage = '%-*s fetched in:  %.4s seconds' % (6, stock, (time.time() - startTime))
            pbar.set_description(outMessage)
            pbar.update(1)

    pbar.close()
    if len(log_errors) > 0:
        print(log_errors)
    # if len(log_update) > 0:
    #     print(log_update)

    return symbols

if __name__ == "__main__":
    pd.set_option('precision', 3)
    pd.set_option('display.width',1000)
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

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
    
    updateStockData_CHN(root_path, True)

    if storeType == 1:
        # stop database server (sync)
        time.sleep(5)
        ShutdownServer()
