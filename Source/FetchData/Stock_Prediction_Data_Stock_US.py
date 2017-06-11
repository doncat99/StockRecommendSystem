import sys
sys.path.append('../../Utility/')

import os, io, pandas, time, datetime, requests, warnings, configparser
import pandas as pd
import numpy as np
#import pandas_datareader.data as pandasReader
import fix_yahoo_finance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar
import concurrent.futures
from tqdm import tqdm

def getSingleStock(symbol, from_date, till_date):
    repeat_times = 1
    message = ""

    if len(symbol) == 0:
        return '', message
    
    for _ in range(repeat_times): 
        try:
            #data = pandasReader.DataReader(symbol, "yahoo", from_date, till_date)
            data = yf.download(symbol, from_date, till_date, progress=False)
            data.sort_index()
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
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(from_date, to_date)
    duedays = pd.bdate_range(from_date, to_date)
    df = pd.DataFrame()
    df['Date'] = duedays
    df['Holiday'] = duedays.isin(holidays)
    opendays = df[df['Holiday'] == False]
    return opendays


def judgeNeedPreDownload(dir, symbol, first_date, from_date, to_date):
    publishDay = pd.Timestamp(getStockPublishDay(dir, symbol))

    if pd.isnull(publishDay) == False and publishDay == first_date:
        return False

    dateList = judgeOpenDaysInRange(from_date, to_date)
    if len(dateList) > 0:
        lastDay = pd.Timestamp(dateList['Date'].index[-1])
        if pd.isnull(publishDay) or lastDay > publishDay: 
            return True
    return False

def judgeNeedPostDownload(now_date, from_date, to_date):
    start_date = pd.Timestamp(from_date)
    end_date = pd.Timestamp(to_date)

    if start_date >= now_date:
        return False
    
    if end_date > now_date:
        to_date = now_date

    dateList = judgeOpenDaysInRange(from_date, to_date)
    if len(dateList) > 0:
        return True
    return False


def updateSingleStockData(dir, share_dir, symbol, from_date, till_date, force_check):
    startTime = time.time()
    message = ""

    if len(symbol) == 0:
        return startTime, message

    now_date   = pd.Timestamp((datetime.datetime.now()).strftime("%Y-%m-%d"))
    start_date = pd.Timestamp(from_date)
    end_date   = pd.Timestamp(till_date)
    
    if end_date == now_date:
        end_date = end_date - datetime.timedelta(days=1)
    
    filename = dir + symbol + '.csv'
    
    try:
        #stockData = pd.read_csv(filename, index_col=["Date"], parse_dates=["Date"]).sort_index()
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
                stockData.sort_index(ascending=True, inplace=True)
                stockData.to_csv(filename)
                message = message + ", database updated"
            return startTime, message

    modified = False
    savePublishDay = False
    first_date = pd.Timestamp(stockData.index[0])
    last_date  = pd.Timestamp(stockData.index[-1])
    
    if start_date < first_date:
        to_date = (first_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        if judgeNeedPreDownload(share_dir, symbol, first_date, from_date, to_date):
            #print("pre-download", symbol, start_date, first_date)
            message = message + ", download pre data from " + from_date + " to " + to_date
            moreStockData, tempMessage = getSingleStock(symbol, from_date, to_date)
            message = message + tempMessage
            if len(moreStockData) > 0:
                modified = True
                stockData = pd.concat([moreStockData, stockData])
                stockData.index.name = 'Date'
            else:
                savePublishDay = True
                saveStockPublishDay(share_dir, symbol, first_date.strftime("%Y-%m-%d"))
                message = message + ", save stock publish(IPO) day, next time won't check it again"

    if 'lastUpdate' in stockData:

        lastUpdateTime = pd.Timestamp(stockData['lastUpdate'].iloc[0])
        # try:
        #     lastUpdateTime = pd.Timestamp(stockData['lastUpdate'].iloc[0])
        # except Exception as e:
        #     print("lastUpdateTime", e)
        #     lastUpdateTime = pd.Timestamp('1970-01-01')
    else:
        lastUpdateTime = pd.Timestamp('1970-01-01')

    updateOnce = now_date > lastUpdateTime

    if (end_date > last_date) and (updateOnce or force_check):
        to_date = (last_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        if judgeNeedPostDownload(now_date, to_date, till_date):
            #print("post-download", symbol, to_date, till_date)
            message = message + ", download post data from " + to_date + " to " + till_date
            moreStockData, tempMessage = getSingleStock(symbol, to_date, till_date)
            message = message + tempMessage
            if len(moreStockData) > 0:
                if isinstance(moreStockData.index, pd.DatetimeIndex):
                    moreStockData.index = moreStockData.index.strftime("%Y-%m-%d")
                modified = True
                stockData = pd.concat([stockData, moreStockData])
                stockData.index.name = 'Date'

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
    elif savePublishDay == False:
        message = ""

    return startTime, message


def getStocksList():
    Config = configparser.ConfigParser()
    Config.read("../../config.ini")
    dir = Config.get('Paths', 'STOCK_US')
    
    if os.path.exists(dir) == False: 
        os.makedirs(dir)

    share_dir = dir + Config.get('Paths', 'STOCK_SHARE_FOLDER')
    if os.path.exists(share_dir) == False: 
        os.makedirs(share_dir)

    filename = share_dir + 'StockList.csv'

    if os.path.exists(filename):
        return pd.read_csv(filename)
        #return listData['Code'].values.tolist()

    df = pd.DataFrame()
    for exchange in ["NASDAQ", "NYSE"]:
        url = "http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=%s&render=download" % exchange
        repeat_times = 3 # repeat downloading in case of http error
        for _ in range(repeat_times): 
            try:
                urlData = requests.get(url, timeout=15).content
                if df.empty:
                    df = pd.read_csv(io.StringIO(urlData.decode('utf-8')))
                else:
                    df = pd.concat([df, pd.read_csv(io.StringIO(urlData.decode('utf-8')))])                    
                break
            except Exception as e:
                print ("exception in getStocks:" + exchange, str(e))
                continue

    df = df[(df['MarketCap'] > 100000000)]    
    listData = df[['Symbol', 'MarketCap', 'Sector', 'Industry']]
    listData.to_csv(filename)
    return listData#listData['Code'].values.tolist()


def updateStockData_US(symbols, from_date, till_date, force_check = False):
    Config = configparser.ConfigParser()
    Config.read("../../config.ini")
    dir = Config.get('Paths', 'STOCK_US')
    
    if os.path.exists(dir) == False: 
        os.makedirs(dir)

    share_dir = dir + Config.get('Paths', 'STOCK_SHARE_FOLDER')
    if os.path.exists(share_dir) == False: 
        os.makedirs(share_dir)


    stocklist = getStocksList()['Symbol'].values.tolist()

    for symbol in symbols:
        if symbol not in stocklist:
            stocklist.append(symbol)

    symbols = stocklist

    pbar = tqdm(total=len(symbols))
    log_errors = []
    log_update = []

    # debug only
    # for stock in symbols:
    #     #stock = 'cizn'
    #     startTime, message = updateSingleStockData(dir, share_dir, stock, from_date, till_date, force_check)
    #     outMessage = '%-*s fetched in:  %.4s seconds' % (6, stock, (time.time() - startTime))
    #     pbar.set_description(outMessage)
    #     pbar.update(1)
    #     #break
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Start the load operations and mark each future with its URL
        future_to_stock = {executor.submit(updateSingleStockData, dir, share_dir, symbol, from_date, till_date, force_check): symbol for symbol in symbols}
        for future in concurrent.futures.as_completed(future_to_stock):
            stock = future_to_stock[future]
            try:
                startTime, message = future.result()
            except Exception as exc:
                startTime = time.time()
                log_errors.append('%r generated an exception: %s' % (stock, exc))
                len_errors = len(log_errors)
                if len_errors % 5 == 0: print(log_errors[(len_errors-5):]) 
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

    now = datetime.datetime.now().strftime("%Y-%m-%d")
    updateStockData_US([], "1990-01-01", now, True)