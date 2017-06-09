import os, time, datetime, requests, warnings, configparser
import pandas as pd
import numpy as np
import tushare as ts
import concurrent.futures
from tqdm import tqdm

def getSingleStock(symbol):    
    repeat_times = 3
    message = ""
    for _ in range(repeat_times): 
        try:
            data = ts.get_hist_data(symbol)
            data.sort_index(ascending=True, inplace=True)
            return data, ""
        except Exception as e:
            message = symbol + " fetch exception: " + str(e)
            continue   
        else:
            time.sleep(0.1)
    return '', message

def getSingleStockByTime(symbol, from_date, till_date):
    start = from_date.split('-')
    start_y, start_m, start_d = start[0], start[1], start[2] # starting date

    end = till_date.split('-')
    end_y, end_m, end_d = end[0], end[1], end[2] # until now
    
    repeat_times = 3
    message = ""
    for _ in range(repeat_times): 
        try:
            data = ts.get_hist_data(symbol, from_date, till_date)
            data.sort_index(ascending=True, inplace=True)
            return data, ""
        except Exception as e:
            message = symbol + " fetch exception: " + str(e)
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
            return publishDay['date'].values[0]
    return ''


def saveStockPublishDay(dir, symbol, date):
    filename = dir + 'StockPublishDay.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=["index"])
        publishDate = df[df['Code'] == symbol]
        if publishDate.empty:
            df.loc[len(df)] = [symbol, date]
    else: 
        df = pd.DataFrame(columns = ['Code', 'date'])
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
    df['date'] = duedays
    df['Holiday'] = duedays.isin(holidays)
    opendays = df[df['Holiday'] == False]
    return opendays


# def judgeNeedPreDownload(dir, symbol, from_date, to_date):
#     dateList = judgeOpenDaysInRange(from_date, to_date)
#     if len(dateList) > 0:
#         publishDay = pandas.Timestamp(getStockPublishDay(dir, symbol))
#         lastDay = pandas.Timestamp(dateList['Date'].index[-1])
#         if pandas.isnull(publishDay) or lastDay > publishDay: 
#             return True
#     return False


def judgeNeedPostDownload(from_date, to_date):
    today = datetime.datetime.now()
    start_date = pd.Timestamp(from_date)
    end_date = pd.Timestamp(to_date)

    if start_date > today:
        return False
    
    if end_date > today:
        to_date = today.strftime("%Y-%m-%d")

    dateList = judgeOpenDaysInRange(from_date, to_date)
    if len(dateList) > 0:
        return True
    return False


def updateSingleStockData(dir, stock, force_check):
    startTime = time.time()
    symbol = str(stock[0]).zfill(6)
    message = ""

    if len(symbol) == 0:
        return startTime, message

    till_date = (datetime.datetime.now()).strftime("%Y-%m-%d")
    end_date  = pd.Timestamp(till_date)

    filename = dir + symbol + '.csv'
    
    try:
        stockData = pd.read_csv(filename, index_col=["date"])
    except Exception as e:
        #print(symbol, " read csv exception, ", e)
        if str(e) == 'Index Date invalid':
            stockData = pd.read_csv(filename,index_col=0)
            stockData.index.name = 'date'
            stockData.sort_index(ascending=True, inplace=True)
            stockData.to_csv(filename)
        else:
            stockData, message = getSingleStock(symbol)
            if len(stockData) > 0: 
                stockData['lastUpdate'] = till_date
                stockData.sort_index(ascending=True, inplace=True)
                stockData.to_csv(filename)
                message = symbol + " database updated"
            return startTime, message

    modified = False
    first_date = pd.Timestamp(stockData.index[0])
    last_date  = pd.Timestamp(stockData.index[-1])

    if 'lastUpdate' in stockData:
        lastUpdateTime = pd.Timestamp(stockData['lastUpdate'].iloc[0])
    else:
        lastUpdateTime = pd.Timestamp('1970-01-01')

    updateOnce = end_date > lastUpdateTime
     
    if end_date > last_date and (updateOnce or force_check):
        to_date = (last_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        if judgeNeedPostDownload(to_date, till_date):
            message = message + ", download post data from " + to_date + " to " + till_date
            moreStockData, tempMessage = getSingleStockByTime(symbol, to_date, till_date)
            message = message + tempMessage
            if len(moreStockData) > 0:
                modified = True
                stockData = pd.concat([stockData, moreStockData])
                stockData.index.name = 'date'
        
    if modified:
        stockData['lastUpdate'] = till_date
        stockData = stockData[~stockData.index.duplicated(keep='first')]
        stockData.sort_index(ascending=True, inplace=True)
        stockData.to_csv(filename)
    elif updateOnce:
        stockData['lastUpdate'] = till_date
        stockData = stockData[~stockData.index.duplicated(keep='first')]
        stockData.sort_index(ascending=True, inplace=True)
        stockData.to_csv(filename)
        message = message + ", nothing updated"
    else:
        message = ""

    return startTime, message


def getStocksList(share_dir):
    filename = share_dir + 'StockList_China.csv'

    if os.path.exists(filename):
        return pd.read_csv(filename) 

    stock_info = ts.get_stock_basics()
    listData = pd.DataFrame(stock_info)
    listData.to_csv(filename)
    return listData

def updateStockList(dir):
    share_dir = dir + '_share/'

    if os.path.exists(share_dir) == False: 
        os.makedirs(share_dir)

    return getStocksList(share_dir)

def updateStockData_CHN(dir, force_check = False):

    if os.path.exists(dir) == False: 
        os.makedirs(dir)

    stocklist = updateStockList(dir)

    pbar = tqdm(total=len(stocklist))
    log_errors = []
    log_update = []

    # debug only
    for index, stock in stocklist.iterrows():
        startTime, message = updateSingleStockData(dir, stock, force_check)
        outMessage = '%-*s fetched in:  %.4s seconds' % (6, str(stock[0]).zfill(6), (time.time() - startTime))
        pbar.set_description(outMessage)
        pbar.update(1)

    # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    #     # Start the load operations and mark each future with its URL
    #     future_to_stock = {executor.submit(updateSingleStockData, dir, stock, force_check): stock for index, stock in stocklist.iterrows()}
    #     for future in concurrent.futures.as_completed(future_to_stock):
    #         stock = future_to_stock[future]
    #         try:
    #             startTime, message = future.result()
    #         except Exception as exc:
    #             log_errors.append('%r generated an exception: %s' % (str(stock[0]).zfill(6), exc))
    #         else:
    #             if len(message) > 0: log_update.append(message)
    #         outMessage = '%-*s fetched in:  %.4s seconds' % (6, str(stock[0]).zfill(6), (time.time() - startTime))
    #         pbar.set_description(outMessage)
    #         pbar.update(1)

    pbar.close()
    if len(log_errors) > 0:
        print(log_errors)
    # if len(log_update) > 0:
    #     print(log_update)

    return stocklist

if __name__ == "__main__":
    pd.set_option('precision', 3)
    pd.set_option('display.width',1000)
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    Config = configparser.ConfigParser()
    Config.read("../../config.ini")
    dir = Config.get('Paths', 'STOCK_CHN')

    updateStockData_CHN(dir, True)