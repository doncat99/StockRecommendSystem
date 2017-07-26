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
from DB_API import queryStockList, queryNews, storeNews

global_eventRegistry = None

def getEventRegistry(root_path):
    global global_eventRegistry
    if global_eventRegistry is None:
        config = configparser.ConfigParser()
        config.read(root_path + "/" + "config.ini")
        global_eventRegistry = EventRegistry(apiKey = config.get('EventRegistry', 'KEY'))
    return global_eventRegistry

# def getSingleStockNewsArticle(root_path, symbol, name, from_date, till_date, count):
#     er = getEventRegistry(root_path)

#     conceptUri = er.getConceptUri(symbol)
#     businessUri = er.getCategoryUri("Business")
#     financeUri = er.getCategoryUri("Finance")
   
#     qStr = ComplexArticleQuery(
#         CombinedQuery.AND([
#             BaseQuery(dateStart = from_date, dateEnd = till_date),
#             # CombinedQuery.OR([
#             #     BaseQuery(conceptUri = QueryItems.OR([searchUri])),
#             #     BaseQuery(keyword = name)
#             # ]),
#             BaseQuery(keyword = QueryItems.OR(["NASDAQ:"+symbol, "NYSE:"+symbol, "("+symbol+")", name])),
#             #BaseQuery(conceptUri = conceptUri),
#             BaseQuery(categoryUri = QueryItems.OR([businessUri, financeUri])),
#             BaseQuery(lang = "eng")
#         ])
#     )
#     q = QueryArticles.initWithComplexQuery(qStr)
#     q.addRequestedResult(RequestArticlesInfo(count = count, 
#     returnInfo = ReturnInfo(
#         conceptInfo = ConceptInfoFlags(lang = "eng"),
#         articleInfo = ArticleInfoFlags(
#             bodyLen = -1, duplicateList = True, concepts = True, 
#             categories = True, location = False, image = False))))
#     res = er.execQuery(q)

#     df = pd.DataFrame(columns=['date', 'time', 'title', 'source', 'uri', 'body_eng', 'body_chn'])

#     if 'info' in res:
#         print(symbol, res["info"])
#         return df

#     translator = Translator()
#     #count = 1
#     #print(res)
#     for art in res["articles"]["results"]:
#         # print(res)
#         # print("\n-------- " + str(count) + " --------\n")
#         # print("title: ", art['title'])
#         # print("source: ", art['source']['title'])
#         # print("dateTime: ", art['dateTime'])
#         # print("body: ")
#         lines = art['body'].splitlines()
#         eng = [line for line in lines if len(line) > 0]
#         chn = []
        
#         for line in eng:
#             try:
#                 chn.append(translator.translate(line, src='en', dest='zh-CN').text)
#             except Exception as e:
#                 chn.append(str(e))

#         # for line in lines:
#         #     if len(line) < 1: continue
#         #     trans = translator.translate(line, src='en', dest='zh-CN').text
#         #     transLines.append(trans)
#             #print(line + "\n\n" + trans + "\n")
#         #count += 1
#         df.loc[len(df)] = [art['date'], art['time'], art['title'], art['source']['title'], art['uri'], eng, chn]
#     #print("article", len(df))
#     return df

def getSingleStockNewsArticle(root_path, symbol, name, from_date, till_date, count):
    config = configparser.ConfigParser()
    config.read(root_path + "/" + "config.ini")

    url = "https://api.newsriver.io/v2/search?query=title%3Anasdaq%20" + symbol + "%20language%3Aen%20%20AND%20discoverDate%3A%5B" + from_date + "%20TO%20" + till_date + "%5D%0A&sortBy=_score&sortOrder=DESC&limit=" + str(count)

    response = requests.get(url, headers={"Authorization": config.get('NewsRiver', 'KEY')})
    jsonFile = response.json()

    df = pd.DataFrame(columns=['date', 'time', 'title', 'source', 'ranking', 'sentiment', 'uri', 'url', 'body_eng', 'body_chn'])
    translator = Translator()
    
    for art in jsonFile:
        try:
            finSentiment = art['metadata']['finSentiment']['sentiment']
        except:
            finSentiment = "0.0"
        
        try: 
            source = art['website']['hostName']
        except:
            source = "N/A"

        try:
            ranking = art['website']['rankingGlobal']
        except:
            ranking = "N/A"

        try:
            trans = ""#translator.translate(art['text'], src='en', dest='zh-CN').text
        except:
            trans = ""

        df.loc[len(df)] = [art['discoverDate'][:10], 
                           art['discoverDate'][11:19], 
                           art['title'], 
                           source, 
                           ranking, 
                           finSentiment,
                           art['id'], 
                           art['url'],
                           art['text'], 
                           trans
                          ]

    return df

def updateNewsArticle(root_path, symbol, name, from_date, till_date, count):
    startTime = time.time()
    message = ""

    if len(symbol) == 0: return startTime, message

    #df, lastUpdateTime = queryNews(root_path, "DB_STOCK", "SHEET_US_NEWS", symbol)

    # if (datetime.datetime.now() - lastUpdateTime) < datetime.timedelta(hours=24):
    #     return
    
    #if df.empty:
    df = getSingleStockNewsArticle(root_path, symbol, name, from_date, till_date, count)
    
    storeNews(root_path, "DB_STOCK", "SHEET_US_NEWS", symbol, df)
    
    # first_date = pd.Timestamp(df.index[0])#.tz_localize(None)
    # last_date  = pd.Timestamp(df.index[-1])#.tz_localize(None)

    # modified = False

    # # require pre download
    # if first_date > pd.Timestamp(from_date):
    #     pre_df = getSingleStockNewsArticle(root_path, symbol, name, from_date, first_date.strftime("%Y-%m-%d"), count)
    #     #print("pre_df", from_date, first_date.strftime("%Y-%m-%d"))
    #     #print(pre_df)
    #     df = pd.concat([pre_df, df])
    #     modified = True
    
    # if last_date < pd.Timestamp(till_date):
    #     post_df = getSingleStockNewsArticle(root_path, symbol, name, last_date.strftime("%Y-%m-%d"), till_date, count)
    #     #print("post_df", last_date.strftime("%Y-%m-%d"), till_date)
    #     #print(post_df)
    #     df = pd.concat([df, post_df])
    #     modified = True

    # if modified:
    #     storeNews(root_path, "DB_STOCK", "SHEET_US_NEWS", symbol, df)

    
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("please input Stock symbol and start date, end date after python file")
        exit()

    pd.set_option('precision', 3)
    pd.set_option('display.width',1000)
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    symbol = str(sys.argv[1])

    stocklist = queryStockList(root_path, "DB_STOCK", "SHEET_US_DAILY")
    
    result = stocklist[stocklist.index == symbol]

    if result.empty:
        print("symbol not exist.")
        exit()

    start_date = str(sys.argv[2])
    end_date = str(sys.argv[3])

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
    
    name = result['name'].values[0]
    print("fetching news of stock:", symbol, name)
    updateNewsArticle(root_path, symbol, name, start_date, end_date, 5)

    # if storeType == 1:
    #     # stop database server (sync)
    #     time.sleep(5)
    #     ShutdownServer()
    

    
