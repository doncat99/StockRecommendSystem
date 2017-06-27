
import os, datetime, configparser
import pandas as pd
from bson import json_util

global_config = None
global_client = None

def getConfig(root_path):
    global global_config
    if global_config is None:
        global_config = configparser.ConfigParser()
        global_config.read(root_path + "/" + "config.ini")
    return global_config

def getClient():
    global global_client
    from pymongo import MongoClient
    if global_client is None: global_client = MongoClient('localhost', 27017)
    return global_client

def getCollection(database, collection):
    client = getClient()
    db = client[database]
    return db[collection]

def readFromCollection(collection):
    return pd.DataFrame(list(collection.find()))

def writeToCollection(collection, df):
    jsonStrings = df.to_json(orient='records')
    bsonStrings = json_util.loads(jsonStrings)
    collection.insert_many(bsonStrings)


def queryStockList(root_path, database):
    #symbol_exception = ['AXON', 'CTT', 'ARL']
    CollectionKey = "StockList"
    stockList = pd.DataFrame()
    config = getConfig(root_path)
    storeType = int(config.get('Setting', 'StoreType'))
    
    if storeType == 1:
        collection = getCollection(database, CollectionKey)
        try:
            return readFromCollection(collection)
        except Exception as e:
            print("queryStockList Exception", e)
            return stockList

    if storeType == 2:
        csv_dir = root_path + "/" + config.get('Paths', database) + config.get('Paths', 'STOCK_SHARE')
        filename = csv_dir + CollectionKey + '.csv'
        return pd.read_csv(filename, index_col=0)

    return stockList

def storeStockList(root_path, database, df):
    CollectionKey = "StockList"
    config = getConfig(root_path)
    storeType = int(config.get('Setting', 'StoreType')) 

    df.index.name = 'index'
    df = df.reset_index(drop=True)

    if storeType == 1:
        collection = getCollection(database, CollectionKey)
        collection.remove()
        writeToCollection(collection, df)

    if storeType == 2:
        csv_dir = root_path + "/" + config.get('Paths', database) + config.get('Paths', 'STOCK_SHARE')
        if os.path.exists(csv_dir) == False:
            os.makedirs(csv_dir)
        filename = csv_dir + CollectionKey + '.csv'
        df.to_csv(filename)


def queryStockPublishDay(root_path, database, symbol):
    CollectionKey = "StockPublishDay"
    config = getConfig(root_path)
    storeType = int(config.get('Setting', 'StoreType'))

    if storeType == 1:
        collection = getCollection(database, CollectionKey)

        try:
            df = readFromCollection(collection)
        except Exception as e:
            print("queryStockPublishDay Exception", e)
            return ''
            
        if df.empty == False:
            publishDay = df[df['Code'] == symbol]
            if len(publishDay) == 1:
                return publishDay['Date'].values[0]
        return ''

    if storeType == 2:
        csv_dir = root_path + "/" + config.get('Paths', database) + config.get('Paths', 'STOCK_SHARE')
        filename = csv_dir + CollectionKey + '.csv'
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            publishDay = df[df['Code'] == symbol]
            if len(publishDay) == 1:
                return publishDay['Date'].values[0]
        return ''
    return ''

def storePublishDay(root_path, database, symbol, date):
    CollectionKey = "StockPublishDay"
    config = getConfig(root_path)
    storeType = int(config.get('Setting', 'StoreType'))

    if storeType == 1:
        collection = getCollection(database, CollectionKey)

        df = pd.DataFrame(columns = ['Code', 'Date'])
        df.index.name = 'index'
        df.loc[len(df)] = [symbol, date]
        writeToCollection(collection, df)

    if storeType == 2:
        csv_dir = root_path + "/" + config.get('Paths', database) + config.get('Paths', 'STOCK_SHARE')
        if os.path.exists(csv_dir) == False:
            os.makedirs(csv_dir)
        filename = csv_dir + CollectionKey + '.csv'
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
    CollectionKey = symbol
    config = getConfig(root_path)
    storeType = int(config.get('Setting', 'StoreType'))
    
    stockData = pd.DataFrame()
    lastUpdateTime = pd.Timestamp('1970-01-01')

    if storeType == 1:
        collection = getCollection(database, CollectionKey)

        try:
            stockData = readFromCollection(collection)
            lastUpdateTime = pd.Timestamp(stockData['lastUpdate'].iloc[0])
            stockData = stockData.set_index('Date')
            return stockData, lastUpdateTime
        except Exception as e:
            print("queryStock Exception", e)
            return pd.DataFrame(), lastUpdateTime

    if storeType == 2:
        csv_dir = root_path + "/" + config.get('Paths', database)
        filename = csv_dir + CollectionKey + '.csv'
        try:
            stockData = pd.read_csv(filename, index_col=["Date"])
            if 'lastUpdate' in stockData:
                lastUpdateTime = pd.Timestamp(stockData['lastUpdate'].iloc[0])
            return stockData, lastUpdateTime
        except Exception as e:
            return pd.DataFrame(), lastUpdateTime

    return stockData, lastUpdateTime


def storeStock(root_path, database, symbol, stockData):
    CollectionKey = symbol
    config = getConfig(root_path)
    storeType = int(config.get('Setting', 'StoreType'))
    now_date = datetime.datetime.now().strftime("%Y-%m-%d")
    stockData.index.name = 'Date'
    
    if 'Date' in stockData: stockData.set_index('Date')  

    stockData['lastUpdate'] = now_date
    stockData.index = stockData.index.astype(str)
    stockData.sort_index(ascending=True, inplace=True)

    if storeType == 1:
        collection = getCollection(database, CollectionKey)
        collection.remove()
        stockData = stockData.reset_index() 
        writeToCollection(collection, stockData)

    if storeType == 2:
        csv_dir = root_path + "/" + config.get('Paths', database)
        if os.path.exists(csv_dir) == False:
            os.makedirs(csv_dir)
        filename = csv_dir + symbol + '.csv'
        stockData.to_csv(filename)


def queryNews(root_path, database, symbol):
    global global_config
    global global_client

    if global_config is None:
        global_config = configparser.ConfigParser()
        global_config.read(root_path + "/" + "config.ini")
    storeType = int(global_config.get('Setting', 'StoreType'))
    stockNews = pd.DataFrame()

    stockListKey = "StockList"

    if storeType == 1:
        if global_client is None: global_client = createClient()

        try:
            collection = global_client[database]
        except:
            global_client.initialize_collection(database)
            collection = global_client[database]

        try:
            item = collection.read(symbol)
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
    global global_client

    if global_config is None:
        global_config = configparser.ConfigParser()
        global_config.read(root_path + "/" + "config.ini")

    now_date = datetime.datetime.now().strftime("%Y-%m-%d")
    storeType = int(global_config.get('Setting', 'StoreType'))
    
    df = df.drop_duplicates(subset=['Uri'], keep='first')
    df.set_index(['Date'], inplace=True)
    df.sort_index(ascending=True, inplace=True)
    
    if storeType == 1:
        if global_client is None: global_client = createClient()

        try:
            collection = global_client[database]
        except:
            global_client.initialize_collection(database)
            collection = global_client[database]

        #collection.delete(symbol)
        collection.write(symbol, df)

    if storeType == 2:
        csv_dir = root_path + "/" + global_config.get('Paths', database)
        if os.path.exists(csv_dir) == False:
            os.makedirs(csv_dir)
        filename = csv_dir + symbol + '.csv'
        df.to_csv(filename)


def queryEarnings(root_path, database, date):
    global global_config
    global global_client

    if global_config is None:
        global_config = configparser.ConfigParser()
        global_config.read(root_path + "/" + "config.ini")
    storeType = int(global_config.get('Setting', 'StoreType'))
    stockEarnings = pd.DataFrame()

    if storeType == 1:
        if global_client is None: global_client = createClient()

        try:
            collection = global_client[database]
        except:
            global_client.initialize_collection(database)
            collection = global_client[database]

        try:
            item = collection.read(date)
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
    global global_client

    if global_config is None:
        global_config = configparser.ConfigParser()
        global_config.read(root_path + "/" + "config.ini")

    now_date = datetime.datetime.now().strftime("%Y-%m-%d")
    storeType = int(global_config.get('Setting', 'StoreType'))
    
    if storeType == 1:
        if global_client is None: global_client = createClient()

        try:
            collection = global_client[database]
        except:
            global_client.initialize_collection(database)
            collection = global_client[database]

        #collection.delete(symbol)
        collection.write(date, df)

    if storeType == 2:
        csv_dir = root_path + "/" + global_config.get('Paths', database)
        if os.path.exists(csv_dir) == False:
            os.makedirs(csv_dir)
        filename = csv_dir + date + '.csv'
        df.to_csv(filename)


def queryTweets(root_path, database, symbol, col):
    global global_config
    global global_client

    if global_config is None:
        global_config = configparser.ConfigParser()
        global_config.read(root_path + "/" + "config.ini")
    storeType = int(global_config.get('Setting', 'StoreType'))
    stockTweets = pd.DataFrame()

    if storeType == 1:
        if global_client is None: global_client = createClient()

        try:
            collection = global_client[database]
        except:
            global_client.initialize_collection(database)
            collection = global_client[database]

        try:
            item = collection.read(symbol)
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
    global global_client

    if global_config is None:
        global_config = configparser.ConfigParser()
        global_config.read(root_path + "/" + "config.ini")

    now_date = datetime.datetime.now().strftime("%Y-%m-%d")
    storeType = int(global_config.get('Setting', 'StoreType'))

    df = df.drop_duplicates(keep='last')
    df = df.sort_values(['Date'], ascending=[False]).reset_index(drop=True)
    
    if storeType == 1:
        if global_client is None: global_client = createClient()

        try:
            collection = global_client[database]
        except:
            global_client.initialize_collection(database)
            collection = global_client[database]

        #collection.delete(symbol)
        collection.write(symbol, df)

    if storeType == 2:
        csv_dir = root_path + "/" + global_config.get('Paths', database)
        if os.path.exists(csv_dir) == False:
            os.makedirs(csv_dir)
        filename = csv_dir + date + '.csv'
        df.to_csv(filename)


def queryCoorelation(root_path, database):
    global global_config
    global global_client

    if global_config is None:
        global_config = configparser.ConfigParser()
        global_config.read(root_path + "/" + "config.ini")
    storeType = int(global_config.get('Setting', 'StoreType'))
    stockCoorelation = pd.DataFrame()

    Key = "us_company_coorelation"

    if storeType == 1:
        if global_client is None: global_client = createClient()

        try:
            collection = global_client[database]
        except:
            global_client.initialize_collection(database)
            collection = global_client[database]

        try:
            item = collection.read(Key)
            return item.data
        except Exception as e:
            return stockCoorelation

    if storeType == 2:
        dir = root_path + "/" + global_config.get('Paths', database)
        filename = dir + Key + ".csv"
        stockCoorelation = pd.read_csv(filename, index_col=0)
        return stockCoorelation

    return stockCoorelation


def storeCoorelation(root_path, database, df):
    global global_config
    global global_client

    if global_config is None:
        global_config = configparser.ConfigParser()
        global_config.read(root_path + "/" + "config.ini")

    now_date = datetime.datetime.now().strftime("%Y-%m-%d")
    storeType = int(global_config.get('Setting', 'StoreType'))

    Key = "us_company_coorelation"

    if storeType == 1:
        if global_client is None: global_client = createClient()

        try:
            collection = global_client[database]
        except:
            global_client.initialize_collection(database)
            collection = global_client[database]

        #collection.delete(symbol)
        collection.write(Key, df)

    if storeType == 2:
        csv_dir = root_path + "/" + global_config.get('Paths', database)
        if os.path.exists(csv_dir) == False:
            os.makedirs(csv_dir)
        filename = csv_dir + Key + '.csv'
        df.to_csv(filename)