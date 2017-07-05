#install eventregistry using "pip install eventregistry" - see https://github.com/gregorleban/EventRegistry    
import sys, os, time, datetime, configparser
import pandas as pd
from eventregistry import *
from googletrans import Translator

cur_path = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    root_path = cur_path[0:cur_path.rfind('/', 0, len(cur_path))]
    cur_path = root_path
sys.path.append(root_path + "/" + 'Source/DataBase/')
from DB_API import queryNews, storeNews
from Fetch_Data_Stock_US_Daily import getStocksList

global_eventRegistry = None

def getSingleStockNewsArticle(root_path, stock, from_date, till_date, count):
    global global_eventRegistry

    if global_eventRegistry is None:
        config = configparser.ConfigParser()
        config.read(root_path + "/" + "config.ini")
        global_eventRegistry = EventRegistry(apiKey = config.get('EventRegistry', 'KEY'))

    start = from_date.split('-')
    start_y, start_m, start_d = int(start[0]), int(start[1]), int(start[2]) # starting date

    end = till_date.split('-')
    end_y, end_m, end_d = int(end[0]), int(end[1]), int(end[2]) # until now

    q = QueryArticles(lang='eng')
    q.setDateLimit(datetime.datetime(start_y, start_m, start_d), datetime.datetime(end_y, end_m, end_d))
    q.addKeyword(stock)
    q.addRequestedResult(RequestArticlesInfo(count = count, 
    returnInfo = ReturnInfo(
        articleInfo = ArticleInfoFlags(
            bodyLen = -1, duplicateList = True, concepts = True, 
            categories = True, location = True, image = True))))
    res = global_eventRegistry.execQuery(q)

    df = pd.DataFrame(columns=['Date', 'Title', 'Source', 'Body', 'Uri'])

    if 'info' in res:
        print(stock, res["info"])
        return df

    #translator = Translator()
    #count = 1
    for art in res["articles"]["results"]:
        df.loc[len(df)] = [art['dateTime'], art['title'], art['source']['title'], art['body'], art['uri']]
        # print("\n-------- " + str(count) + " --------\n")
        # print("title: ", art['title'])
        # print("source: ", art['source']['title'])
        # print("dateTime: ", art['dateTime'])
        # print("body: ")
        # lines = art['body'].splitlines()

        # for line in lines:
        #     if len(line) < 1: continue
        #     trans = translator.translate(line, src='en', dest='zh-CN').text
        #     print(line + "\n\n" + trans + "\n")
        #count += 1
    return df
    
    
def updateNewsArticle(root_path, symbol, from_date, till_date, count):
    startTime = time.time()
    message = ""

    if len(symbol) == 0: return startTime, message

    df, lastUpdateTime = queryNews(root_path, "DB_STOCK", "SHEET_US_NEWS", symbol)

    if (datetime.datetime.now() - lastUpdateTime) < datetime.timedelta(hours=24):
        return

    if df.empty:
        df = getSingleStockNewsArticle(root_path, symbol, from_date, till_date, count)
        storeNews(root_path, "DB_STOCK", "SHEET_US_NEWS", symbol, df)
        return 
    
    print(df)
    first_date = pd.Timestamp(df['Date'].iloc[0]).tz_localize(None)
    last_date  = pd.Timestamp(df['Date'].iloc[-1]).tz_localize(None)

    modified = False

    # require pre download
    if first_date > pd.Timestamp(from_date):
        pre_df = getSingleStockNewsArticle(root_path, symbol, from_date, first_date.strftime("%Y-%m-%d"), count)
        #print("pre_df", from_date, first_date.strftime("%Y-%m-%d"))
        #print(pre_df)
        df = pd.concat([pre_df, df])
        modified = True
    
    if last_date < pd.Timestamp(till_date):
        post_df = getSingleStockNewsArticle(root_path, symbol, last_date.strftime("%Y-%m-%d"), till_date, count)
        #print("post_df", last_date.strftime("%Y-%m-%d"), till_date)
        #print(post_df)
        df = pd.concat([df, post_df])
        modified = True

    if modified:
        storeNews(root_path, "DB_STOCK", "SHEET_US_NEWS", symbol, df)

    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("please input Stock symbol after python file")
        exit()

    pd.set_option('precision', 3)
    pd.set_option('display.width',1000)
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    #stocklist = getStocksList()
    # stocklist = ['AMD', 'WDC', 'SINA', 'WB', 'CTRP', 'NTES', 'ATVI', 'FB', 'GLUU', 'NVDA', 'NFLX', 'GPRO',
    #              'MRVL', 'SMCI', 'JD', 'INTC', 'AMZN', 'BIDU', 'BGNE', 'QIWI', 'XNET', 'MOMO', 'YY']

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
    
    symbol = str(sys.argv[1])
    print("fetching news of stock:", symbol)
    updateNewsArticle(root_path, symbol, "2016-06-01", now, 200)

    # if storeType == 1:
    #     # stop database server (sync)
    #     time.sleep(5)
    #     ShutdownServer()
    
 
    
