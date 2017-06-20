import sys
sys.path.append('Utility/')
sys.path.append('DataBase/')
import os, io, time, datetime, requests, warnings, configparser
import pandas as pd
import numpy as np
import fix_yahoo_finance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar
import concurrent.futures
from tqdm import tqdm

from DB_API import queryStock, storeStock, queryStockList, storeStockList, queryStockPublishDay, storePublishDay

def getStocksList(root_path):
    try:
        df = queryStockList(root_path, "STOCK_US")
    except Exception as e:
        df = pd.DataFrame()

    if df.empty == False:
        return df

    for exchange in ["NASDAQ", "NYSE"]:
        print("fetching " + exchange + " stocklist...")
        url = "http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=%s&render=download" % exchange
        repeat_times = 1 # repeat downloading in case of http error
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
    df = df.drop_duplicates(subset=['Symbol'], keep='first')
    df.sort_index(ascending=True, inplace=True)

    listData = df[['Symbol', 'Name', 'MarketCap', 'Sector', 'Industry']].copy()
    listData.loc[len(listData)] = ['SPY', 'SPDR S&P 500 ETF Trust', 0.0, '', '']
    listData.loc[len(listData)] = ['^VIX', 'VOLATILITY S&P 500', 0.0, '', '']
    listData['Symbol'] = listData['Symbol'].str.strip()
    storeStockList(root_path, "STOCK_US", listData)
    return queryStockList(root_path, "STOCK_US")


def getSingleStock(symbol, from_date, till_date):
    repeat_times = 1
    message = ""
    df = pd.DataFrame()

    if len(symbol) == 0: return df, message
    
    for _ in range(repeat_times): 
        try:
            data = yf.download(symbol, from_date, till_date, progress=False)
            data.sort_index()
            return data, message
        except Exception as e:
            message = symbol + " fetch exception: " + str(e)
            continue   
    return df, message

def judgeOpenDaysInRange(from_date, to_date):
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(from_date, to_date)
    duedays = pd.bdate_range(from_date, to_date)
    df = pd.DataFrame()
    df['Date'] = duedays
    df['Holiday'] = duedays.isin(holidays)
    opendays = df[df['Holiday'] == False]
    return opendays

def judgeNeedPreDownload(root_path, symbol, first_date, from_date, to_date):
    publishDay = pd.Timestamp(queryStockPublishDay(root_path, "STOCK_US", symbol))
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

    if start_date >= now_date: return False
    if end_date > now_date: to_date = now_date

    dateList = judgeOpenDaysInRange(from_date, to_date)
    
    if len(dateList) > 0: return True
    return False


def updateSingleStockData(root_path, symbol, from_date, till_date, force_check):
    startTime = time.time()
    message = ""

    if len(symbol) == 0: return startTime, message

    now_date   = pd.Timestamp((datetime.datetime.now()).strftime("%Y-%m-%d"))
    start_date = pd.Timestamp(from_date)
    end_date   = pd.Timestamp(till_date)
    
    if end_date == now_date: 
        end_date = end_date - datetime.timedelta(days=1)
     
    stockData, lastUpdateTime = queryStock(root_path, symbol, "STOCK_US")

    if stockData.empty:
        stockData, message = getSingleStock(symbol, from_date, till_date)
        if stockData.empty == False:
            storeStock(root_path, "STOCK_US", symbol, stockData)
            first_date = pd.Timestamp(stockData.index[0])
            to_date = (first_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            if judgeNeedPreDownload(root_path, symbol, first_date, from_date, to_date):
                storePublishDay(root_path, "STOCK_US", symbol, first_date.strftime("%Y-%m-%d"))
            message = message + ", database updated"
        return startTime, message

    modified = False
    savePublishDay = False

    first_date = pd.Timestamp(stockData.index[0])
    last_date  = pd.Timestamp(stockData.index[-1])
    
    if start_date < first_date:
        to_date = (first_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        if judgeNeedPreDownload(root_path, symbol, first_date, from_date, to_date):
            message = message + ", download pre data from " + from_date + " to " + to_date
            moreStockData, tempMessage = getSingleStock(symbol, from_date, to_date)
            message = message + tempMessage
            if len(moreStockData) > 0:
                if isinstance(moreStockData.index, pd.DatetimeIndex):
                    moreStockData.index = moreStockData.index.strftime("%Y-%m-%d")
                modified = True
                stockData = pd.concat([moreStockData, stockData])
                stockData.index.name = 'Date'
            else:
                savePublishDay = True
                storePublishDay(root_path, "STOCK_US", symbol, first_date.strftime("%Y-%m-%d"))
                message = message + ", save stock publish(IPO) day, next time won't check it again"

    updateOnce = now_date > lastUpdateTime

    if (end_date > last_date) and (updateOnce or force_check):
        to_date = (last_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        if judgeNeedPostDownload(now_date, to_date, till_date):
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
        print("data modified: ", symbol)
        stockData = stockData[~stockData.index.duplicated(keep='first')]
        storeStock(root_path, "STOCK_US", symbol, stockData)
        message = message + ", database updated"
    elif updateOnce:
        print("data modified upda: ", symbol)
        stockData = stockData[~stockData.index.duplicated(keep='first')]
        storeStock(root_path, "STOCK_US", symbol, stockData)
        message = message + ", nothing updated"
    elif savePublishDay == False:
        message = ""
    return startTime, message

def updateStockData_US(root_path, from_date, till_date, force_check = False):
    symbols = getStocksList(root_path)['Symbol'].values.tolist()

    pbar = tqdm(total=len(symbols))
    log_errors = []
    log_update = []

    # # debug only
    # for stock in symbols:
    #     startTime, message = updateSingleStockData(root_path, stock, from_date, till_date, force_check)
    #     outMessage = '%-*s fetched in:  %.4s seconds' % (6, stock, (time.time() - startTime))
    #     pbar.set_description(outMessage)
    #     pbar.update(1)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Start the load operations and mark each future with its URL
        future_to_stock = {executor.submit(updateSingleStockData, root_path, symbol, from_date, till_date, force_check): symbol for symbol in symbols}
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
    updateStockData_US("1990-01-01", now, True)