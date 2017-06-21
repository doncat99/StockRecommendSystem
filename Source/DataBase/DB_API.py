import os, datetime, configparser
import pandas as pd

global_config = None
global_store = None

def queryStockList(root_path, database):
    global global_config
    global global_store

    #symbol_exception = ['AXON', 'CTT', 'ARL']

    if global_config is None:
        global_config = configparser.ConfigParser()
        global_config.read(root_path + "/" + "config.ini")
    storeType = int(global_config.get('Setting', 'StoreType'))
    stockList = pd.DataFrame()

    stockListKey = "StockList"

    if storeType == 1:
        if global_store is None:
            from arctic import Arctic
            global_store = Arctic('localhost')

        try:
            library = global_store[database]
        except:
            global_store.initialize_library(database)
            library = global_store[database]

        try:
            item = library.read(stockListKey)
            stockList = item.data
            #stockList = stockList[~stockList.Symbol.isin(symbol_exception)]
            return stockList
        except Exception as e:
            return stockList

    if storeType == 2:
        csv_dir = root_path + "/" + global_config.get('Paths', database) + global_config.get('Paths', 'STOCK_SHARE')
        filename = csv_dir + 'StockList.csv'
        stockList = pd.read_csv(filename, index_col=0)
        #stockList = stockList[~stockList.Symbol.isin(symbol_exception)]
        return stockList

    return stockList

def storeStockList(root_path, database, df):
    global global_config
    global global_store

    if global_config is None:
        global_config = configparser.ConfigParser() 
        global_config.read(root_path + "/" + "config.ini")
    storeType = int(global_config.get('Setting', 'StoreType'))

    stockListKey = "StockList"

    if storeType == 1:
        if global_store is None:
            from arctic import Arctic
            global_store = Arctic('localhost')

        try:
            library = global_store[database]
        except:
            global_store.initialize_library(database)
            library = global_store[database]

        now_date = datetime.datetime.now().strftime("%Y-%m-%d")
        library.delete(stockListKey)
        library.write(stockListKey, df, metadata={'lastUpdate': now_date})

    if storeType == 2:
        csv_dir = root_path + "/" + global_config.get('Paths', database) + global_config.get('Paths', 'STOCK_SHARE')
        if os.path.exists(csv_dir) == False:
            os.makedirs(csv_dir)
        filename = csv_dir + 'StockList.csv'
        df.to_csv(filename)


def queryStockPublishDay(root_path, database, symbol):
    global global_config
    global global_store

    if global_config is None:    
        global_config = configparser.ConfigParser()
        global_config.read(root_path + "/" + "config.ini")
    storeType = int(global_config.get('Setting', 'StoreType'))

    PublishDayKey = "StockPublishDay"

    if storeType == 1:
        if global_store is None:
            from arctic import Arctic
            global_store = Arctic('localhost')

        try:
            library = global_store[database]
        except:
            global_store.initialize_library(database)
            library = global_store[database]

        try:
            item = library.read(PublishDayKey)
            df = item.data
        except Exception as e:
            return ''
            
        if df.empty == False:
            publishDay = df[df['Code'] == symbol]
            if len(publishDay) == 1:
                return publishDay['Date'].values[0]
        return ''

    if storeType == 2:
        csv_dir = root_path + "/" + global_config.get('Paths', database) + global_config.get('Paths', 'STOCK_SHARE')
        filename = csv_dir + 'StockPublishDay.csv'
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            publishDay = df[df['Code'] == symbol]
            if len(publishDay) == 1:
                return publishDay['Date'].values[0]
        return ''

    return ''

def storePublishDay(root_path, database, symbol, date):
    global global_config
    global global_store

    if global_config is None:
        global_config = configparser.ConfigParser()
        global_config.read(root_path + "/" + "config.ini")
    storeType = int(global_config.get('Setting', 'StoreType'))

    PublishDayKey = "StockPublishDay"

    if storeType == 1:
        if global_store is None:
            from arctic import Arctic
            global_store = Arctic('localhost')
        try:
            library = global_store[database]
        except:
            global_store.initialize_library(database)
            library = global_store[database]

        df = pd.DataFrame(columns = ['Code', 'Date'])
        df.index.name = 'index'
        df.loc[len(df)] = [symbol, date]
        now_date = datetime.datetime.now().strftime("%Y-%m-%d")
        library.append(PublishDayKey, df, metadata={'lastUpdate': now_date})

    if storeType == 2:
        csv_dir = root_path + "/" + global_config.get('Paths', database) + global_config.get('Paths', 'STOCK_SHARE')
        if os.path.exists(csv_dir) == False:
            os.makedirs(csv_dir)
        filename = csv_dir + 'StockPublishDay.csv'
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
    

def queryStock(root_path, database, symbol):
    global global_config
    global global_store

    if global_config is None:
        global_config = configparser.ConfigParser()
        global_config.read(root_path + "/" + "config.ini")
    storeType = int(global_config.get('Setting', 'StoreType'))
    
    stockData = pd.DataFrame()
    lastUpdateTime = pd.Timestamp('1970-01-01')

    if storeType == 1:
        if global_store is None:
            from arctic import Arctic
            global_store = Arctic('localhost')
        try:
            library = global_store[database]
        except:
            global_store.initialize_library(database)
            library = global_store[database]

        try:
            item = library.read(symbol)
            metadata = item.metadata
            return item.data, pd.Timestamp(metadata['lastUpdate'])
        except Exception as e:
            return stockData, lastUpdateTime

    if storeType == 2:
        csv_dir = root_path + "/" + global_config.get('Paths', database)
        filename = csv_dir + symbol + '.csv'
        try:
            stockData = pd.read_csv(filename, index_col=["Date"])
            if 'lastUpdate' in stockData:
                lastUpdateTime = pd.Timestamp(stockData['lastUpdate'].iloc[0])
            return stockData, lastUpdateTime
        except Exception as e:
            return stockData, lastUpdateTime

    return stockData, pd.Timestamp('1970-01-01')


def storeStock(root_path, database, symbol, stockData):
    global global_config
    global global_store

    if global_config is None:
        global_config = configparser.ConfigParser()
        global_config.read(root_path + "/" + "config.ini")

    now_date = datetime.datetime.now().strftime("%Y-%m-%d")
    storeType = int(global_config.get('Setting', 'StoreType'))
    stockData.index.name = 'Date'
    
    if 'Date' in stockData:
        stockData.set_index('Date')  

    stockData.index = stockData.index.astype(str)
    stockData.sort_index(ascending=True, inplace=True)

    if storeType == 1:
        if global_store is None:
            from arctic import Arctic
            global_store = Arctic('localhost')

        try:
            library = global_store[database]
        except:
            global_store.initialize_library(database)
            library = global_store[database]

        #library.delete(symbol)
        library.write(symbol, stockData, metadata={'lastUpdate': now_date})

    if storeType == 2:
        csv_dir = root_path + "/" + global_config.get('Paths', database)
        if os.path.exists(csv_dir) == False:
            os.makedirs(csv_dir)
        filename = csv_dir + symbol + '.csv'
        stockData['lastUpdate'] = now_date
        stockData.to_csv(filename)


def queryNews(root_path, database, symbol):
    global global_config
    global global_store

    if global_config is None:
        global_config = configparser.ConfigParser()
        global_config.read(root_path + "/" + "config.ini")
    storeType = int(global_config.get('Setting', 'StoreType'))
    stockNews = pd.DataFrame()

    stockListKey = "StockList"

    if storeType == 1:
        if global_store is None:
            from arctic import Arctic
            global_store = Arctic('localhost')

        try:
            library = global_store[database]
        except:
            global_store.initialize_library(database)
            library = global_store[database]

        try:
            item = library.read(symbol)
            return item.data
        except Exception as e:
            return stockNews

    if storeType == 2:
        dir = root_path + "/" + global_config.get('Paths', database)
        filename = dir + symbol + '.csv'
        stockNews = pd.read_csv(filename)
        return stockNews

    return stockNews


def storeNews(root_path, database, symbol, df):
    global global_config
    global global_store

    if global_config is None:
        global_config = configparser.ConfigParser()
        global_config.read(root_path + "/" + "config.ini")

    now_date = datetime.datetime.now().strftime("%Y-%m-%d")
    storeType = int(global_config.get('Setting', 'StoreType'))
    
    df = df.drop_duplicates(subset=['Uri'], keep='first')
    df.set_index(['Date'], inplace=True)
    df.sort_index(ascending=True, inplace=True)
    
    if storeType == 1:
        if global_store is None:
            from arctic import Arctic
            global_store = Arctic('localhost')

        try:
            library = global_store[database]
        except:
            global_store.initialize_library(database)
            library = global_store[database]

        #library.delete(symbol)
        library.write(symbol, df)

    if storeType == 2:
        csv_dir = root_path + "/" + global_config.get('Paths', database)
        if os.path.exists(csv_dir) == False:
            os.makedirs(csv_dir)
        filename = csv_dir + symbol + '.csv'
        df.to_csv(filename)


def queryEarnings(root_path, database, date):
    global global_config
    global global_store

    if global_config is None:
        global_config = configparser.ConfigParser()
        global_config.read(root_path + "/" + "config.ini")
    storeType = int(global_config.get('Setting', 'StoreType'))
    stockEarnings = pd.DataFrame()

    if storeType == 1:
        if global_store is None:
            from arctic import Arctic
            global_store = Arctic('localhost')

        try:
            library = global_store[database]
        except:
            global_store.initialize_library(database)
            library = global_store[database]

        try:
            item = library.read(date)
            return item.data
        except Exception as e:
            return stockEarnings

    if storeType == 2:
        dir = root_path + "/" + global_config.get('Paths', database)
        filename = dir + date + ".csv"
        stockEarnings = pd.read_csv(filename)
        return stockEarnings

    return stockEarnings


def storeEarnings(root_path, database, date, df):
    global global_config
    global global_store

    if global_config is None:
        global_config = configparser.ConfigParser()
        global_config.read(root_path + "/" + "config.ini")

    now_date = datetime.datetime.now().strftime("%Y-%m-%d")
    storeType = int(global_config.get('Setting', 'StoreType'))
    
    if storeType == 1:
        if global_store is None:
            from arctic import Arctic
            global_store = Arctic('localhost')

        try:
            library = global_store[database]
        except:
            global_store.initialize_library(database)
            library = global_store[database]

        #library.delete(symbol)
        library.write(date, df)

    if storeType == 2:
        csv_dir = root_path + "/" + global_config.get('Paths', database)
        if os.path.exists(csv_dir) == False:
            os.makedirs(csv_dir)
        filename = csv_dir + date + '.csv'
        df.to_csv(filename)


def queryTweets(root_path, database, symbol, col):
    global global_config
    global global_store

    if global_config is None:
        global_config = configparser.ConfigParser()
        global_config.read(root_path + "/" + "config.ini")
    storeType = int(global_config.get('Setting', 'StoreType'))
    stockTweets = pd.DataFrame()

    if storeType == 1:
        if global_store is None:
            from arctic import Arctic
            global_store = Arctic('localhost')

        try:
            library = global_store[database]
        except:
            global_store.initialize_library(database)
            library = global_store[database]

        try:
            item = library.read(symbol)
            return item.data
        except Exception as e:
            return stockTweets

    if storeType == 2:
        dir = root_path + "/" + global_config.get('Paths', database)
        filename = dir + date + ".csv"
        stockTweets = pd.read_csv(filename, usecols=col)
        return stockTweets

    return stockTweets


def storeTweets(root_path, database, symbol, df):
    global global_config
    global global_store

    if global_config is None:
        global_config = configparser.ConfigParser()
        global_config.read(root_path + "/" + "config.ini")

    now_date = datetime.datetime.now().strftime("%Y-%m-%d")
    storeType = int(global_config.get('Setting', 'StoreType'))

    df = df.drop_duplicates(keep='last')
    df = df.sort_values(['Date'], ascending=[False]).reset_index(drop=True)
    
    if storeType == 1:
        if global_store is None:
            from arctic import Arctic
            global_store = Arctic('localhost')

        try:
            library = global_store[database]
        except:
            global_store.initialize_library(database)
            library = global_store[database]

        #library.delete(symbol)
        library.write(symbol, df)

    if storeType == 2:
        csv_dir = root_path + "/" + global_config.get('Paths', database)
        if os.path.exists(csv_dir) == False:
            os.makedirs(csv_dir)
        filename = csv_dir + date + '.csv'
        df.to_csv(filename)