import os, time, datetime, requests, warnings, configparser
import pandas as pd
import numpy as np
import quandl
import concurrent.futures
from tqdm import tqdm

def getSingleStock(symbol, from_date, till_date):    
    repeat_times = 3
    message = ""
    for _ in range(repeat_times): 
        try:
            data = quandl.get("HKEX/"+symbol, start_date=from_date, end_date=till_date)
            data.index = pd.to_datetime(data.index)
            return data, ""
        except Exception as e:
            message = ", fetch exception: " + str(e)
            continue   
        else:
            time.sleep(0.1)
    return '', message

def getStockPublishDay(dir, symbol):
    filename = dir + 'StockPublishDay.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        publishDay = df[df['Code'] == symbol]
        if len(publishDay) == 1:
            return publishDay['Date'].values[0]
    return ''

def saveStockPublishDay(dir, symbol, date):
    filename = dir + 'StockPublishDay.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=["index"])
        publishDate = df[df['Code'] == symbol]
        if publishDate.empty:
            df.loc[len(df)] = [symbol, date]
    else: 
        df = pd.DataFrame(columns = ['Code', 'Date'])
        df.index.name = 'index'
        df.loc[len(df)] = [symbol, date]
    df.to_csv(filename)

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
    df['Date'] = duedays
    df['Holiday'] = duedays.isin(holidays)
    opendays = df[df['Holiday'] == False]
    return opendays

def judgeNeedPreDownload(dir, symbol, from_date, to_date):
    dateList = judgeOpenDaysInRange(from_date, to_date)
    if len(dateList) > 0:
        publishDay = pd.Timestamp(getStockPublishDay(dir, symbol))
        lastDay = pd.Timestamp(dateList['Date'].index[-1])
        if pd.isnull(publishDay) or lastDay > publishDay: 
            return True
    return False

def judgeNeedPostDownload(from_date, to_date):
    today = pd.Timestamp(datetime.datetime.now().strftime("%Y-%m-%d"))
    start_date = pd.Timestamp(from_date)
    end_date = pd.Timestamp(to_date)

    if start_date >= today:
        return False
    
    if end_date > today:
        to_date = today.strftime("%Y-%m-%d")

    dateList = judgeOpenDaysInRange(from_date, to_date)
    if len(dateList) > 0:
        return True
    return False

def updateSingleStockData(dir, symbol, from_date, till_date, force_check):
    startTime = time.time()
    message = ""

    if len(symbol) == 0:
        return startTime, message
    
    now_date   = pd.Timestamp((datetime.datetime.now()).strftime("%Y-%m-%d"))
    end_date  = pd.Timestamp(till_date)

    filename = dir + symbol + '.csv'
    
    try:
        stockData = pd.read_csv(filename, index_col=["Date"])
    except Exception as e:
        #print(symbol, " read csv exception, ", e)
        if str(e) == 'Index Date invalid':
            stockData = pd.read_csv(filename,index_col=0)
            stockData.index.name = 'Date'
            stockData.sort_index(ascending=True, inplace=True)
            stockData.to_csv(filename)
        else:
            stockData, message = getSingleStock(symbol, from_date, till_date)
            if len(stockData) > 0: 
                stockData['lastUpdate'] = now_date.strftime("%Y-%m-%d")
                stockData.to_csv(filename)
                message = message + ", database updated"
            return startTime, message

    modified = False
    first_date = pd.Timestamp(stockData.index[0])
    last_date  = pd.Timestamp(stockData.index[-1])
    
    if 'lastUpdate' in stockData:
        lastUpdateTime = pd.Timestamp(stockData['lastUpdate'].iloc[0])
    else:
        lastUpdateTime = pd.Timestamp('1970-01-01')

    updateOnce = now_date > lastUpdateTime

    if end_date > last_date and (updateOnce or force_check):
        to_date = (last_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        if judgeNeedPostDownload(to_date, till_date):
            message = message + ", download post data from " + to_date + " to " + till_date
            moreStockData, tempMessage = getSingleStock(symbol, to_date, till_date)
            message = message + tempMessage
            if len(moreStockData) > 0:
                modified = True
                stockData = pd.concat([stockData, moreStockData])
                stockData.index.name = 'date'
    
    if modified:
        stockData['lastUpdate'] = now_date.strftime("%Y-%m-%d")
        stockData = stockData[~stockData.index.duplicated(keep='first')]
        stockData.sort_index(ascending=True, inplace=True)
        stockData.to_csv(filename)
    elif updateOnce:
        stockData['lastUpdate'] = now_date.strftime("%Y-%m-%d")
        stockData = stockData[~stockData.index.duplicated(keep='first')]
        stockData.sort_index(ascending=True, inplace=True)
        stockData.to_csv(filename)
        message = message + ", nothing updated"
    else:
        message = ""
    return startTime, message


def getStocksList():
    # https://www.quandl.com/api/v3/databases/HKEX/codes?api_key=X1xf_1YJLkT9mdxDj13v
    # down the zip file above and unzip to the _share folder
    # rename the stock code name from "HKEX/XXXX" to "XXXX"
    Config = configparser.ConfigParser()
    Config.read("../../config.ini")
    dir = Config.get('Paths', 'STOCK_HK')
    
    if os.path.exists(dir) == False: 
        os.makedirs(dir)

    share_dir = dir + Config.get('Paths', 'STOCK_SHARE')
    if os.path.exists(share_dir) == False: 
        os.makedirs(share_dir)

    filename = share_dir + 'HKEX-datasets-codes.csv'

    if os.path.exists(filename):
        listData = pd.read_csv(filename)
        listData = listData[listData['Code'].astype(str).str.isdigit()]
        return listData['Code'].values.tolist()
    return []

def updateStockData_HK(symbols, from_date, till_date, force_check = False):        
    Config = configparser.ConfigParser()
    Config.read("../../config.ini")
    dir = Config.get('Paths', 'STOCK_HK')
    quandl.ApiConfig.api_key = Config.get('Quandl', 'KEY')

    if os.path.exists(dir) == False: 
        os.makedirs(dir)

    stocklist = getStocksList()

    for symbol in symbols:
        if symbol not in stocklist:
            stocklist.append(symbol)

    symbols = stocklist
    pbar = tqdm(total=len(symbols))
    log_errors = []
    log_update = []
    
    # debug only
    for symbol in symbols:
        startTime, message = updateSingleStockData(dir, symbol, from_date, till_date, force_check)
        outMessage = '%-*s fetched in:  %.4s seconds' % (6, symbol, (time.time() - startTime))
        pbar.set_description(outMessage)
        pbar.update(1)
            
    # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    #     # Start the load operations and mark each future with its URL
    #     future_to_stock = {executor.submit(updateSingleStockData, dir, symbol, from_date, till_date, force_check): symbol for symbol in symbols}
    #     for future in concurrent.futures.as_completed(future_to_stock):
    #         stock = future_to_stock[future]
    #         try:
    #             startTime, message = future.result()
    #         except Exception as exc:
    #             log_errors.append('%r generated an exception: %s' % (stock, exc))
    #         else:
    #             if len(message) > 0: log_update.append(message)
    #         outMessage = '%-*s fetched in:  %.4s seconds' % (6, stock, (time.time() - startTime))
    #         pbar.set_description(outMessage)
    #         pbar.update(1)
    
    pbar.close()
    if len(log_errors) > 0:
        print(log_errors)
    return symbols


if __name__ == "__main__":
    pd.set_option('precision', 3)
    pd.set_option('display.width',1000)
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    now = datetime.datetime.now().strftime("%Y-%m-%d")
    updateStockData_HK([], "1990-01-01", now, True)


 