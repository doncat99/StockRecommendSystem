import sys, os, io, time, datetime, requests, warnings, configparser
import pandas as pd
import numpy as np
import pandas_datareader as pdr
from pandas.tseries.holiday import USFederalHolidayCalendar
import concurrent.futures
from tqdm import tqdm

cur_path = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    root_path = cur_path[0:cur_path.rfind('/', 0, len(cur_path))]
    cur_path = root_path
sys.path.append(root_path + "/" + 'Source/DataBase/')
sys.path.append(root_path + "/" + 'Source/Utility/')
from Fetch_Data_Stock_US_StockList import getStocksList_US
from DB_API import queryStock, storeStock, queryStockList, storeStockList, queryStockPublishDay, storePublishDay
import fix_yahoo_finance as yf

def getSingleStock(symbol, from_date, till_date):
    repeat_times = 1
    message = ""
    df = pd.DataFrame()

    if len(symbol) == 0: return df, message
    for _ in range(repeat_times): 
        try:
            data = yf.download(symbol, start=from_date, end=till_date, interval='1wk')
            #data = pdr.get_data_yahoo(symbol, start=from_date, end=till_date, interval='d')
            data = data.rename(columns = {'Date':'date', 'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', "Adj Close":'adj_close', 'Volume':'volume'})
            data.index.name = 'date'
            data.sort_index()
            return data, ""
        except Exception as e:
            message = symbol + " fetch exception: " + str(e)
            continue   
    return df, message

def judgeOpenDaysInRange(from_date, to_date):
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(from_date, to_date)
    duedays = pd.bdate_range(from_date, to_date)
    df = pd.DataFrame()
    df['date'] = duedays
    df['holiday'] = duedays.isin(holidays)
    opendays = df[df['holiday'] == False]
    return opendays

def judgeNeedPreDownload(root_path, symbol, first_date, from_date, to_date):
    publishDay = pd.Timestamp(queryStockPublishDay(root_path, "DB_STOCK", "SHEET_US", symbol))
    if pd.isnull(publishDay) == False and publishDay == first_date:
        return False

    dateList = judgeOpenDaysInRange(from_date, to_date)
    if len(dateList) > 0:
        lastDay = pd.Timestamp(dateList['date'].index[-1])
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
     
    stockData, lastUpdateTime = queryStock(root_path, "DB_STOCK", "SHEET_US", "_WEEKLY", symbol, "weekly_update")
    
    if stockData.empty:
        stockData, message = getSingleStock(symbol, from_date, till_date)
        if stockData.empty == False:
            storeStock(root_path, "DB_STOCK", "SHEET_US", "_WEEKLY", symbol, stockData, "weekly_update")
            first_date = pd.Timestamp(stockData.index[0])
            to_date = (first_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            if judgeNeedPreDownload(root_path, symbol, first_date, from_date, to_date):
                storePublishDay(root_path, "DB_STOCK", "SHEET_US", symbol, first_date.strftime("%Y-%m-%d"))
            message = message + ", database updated"
        else:
            print("get stock from network failed", symbol)
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
                stockData.index.name = 'date'
            else:
                savePublishDay = True
                storePublishDay(root_path, "DB_STOCK", "SHEET_US", symbol, first_date.strftime("%Y-%m-%d"))
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
                stockData.index.name = 'date'

    if modified:
        stockData = stockData[~stockData.index.duplicated(keep='first')]
        storeStock(root_path, "DB_STOCK", "SHEET_US", "_WEEKLY", symbol, stockData, "weekly_update")
    elif updateOnce:
        now_date = datetime.datetime.now().strftime("%Y-%m-%d")
        stockList = queryStockList(root_path, "DB_STOCK", "SHEET_US")
        if stockList[stockList.index == symbol]['daily_update'][0] != now_date:
            stockList.set_value(symbol, 'weekly_update', now_date)
            storeStockList(root_path, "DB_STOCK", "SHEET_US", stockList, symbol)
    elif savePublishDay == False:
        message = ""
    return startTime, message

def updateStockData_US_Weekly(root_path, from_date, till_date, storeType, force_check = False):
    symbols = getStocksList_US(root_path).index

    pbar = tqdm(total=len(symbols))

    if storeType == 2:# or storeType == 1:
        # count = 10
        for stock in symbols:
            startTime, message = updateSingleStockData(root_path, stock, from_date, till_date, force_check)
            outMessage = '%-*s fetched in:  %.4s seconds' % (6, stock, (time.time() - startTime))
            pbar.set_description(outMessage)
            pbar.update(1)
            # count = count - 1
            # if count == 0: break
    
    if storeType == 1:
        log_errors = []
        log_update = []
        # Parallel mode is not suitable in CSV storage mode, since no lock is added to limit csv file IO.
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
        if len(log_errors) > 0: print(log_errors)
        # if len(log_update) > 0: print(log_update)
    
    pbar.close()
    return symbols


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
    
    updateStockData_US_Weekly(root_path, "2014-01-01", now, storeType)

    # if storeType == 1:
    #     # stop database server (sync)
    #     time.sleep(5)
    #     ShutdownServer()


    