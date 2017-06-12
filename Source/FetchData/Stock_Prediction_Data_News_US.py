#install eventregistry using "pip install eventregistry" - see https://github.com/gregorleban/EventRegistry
import os, configparser
import pandas as pd
from eventregistry import *
from googletrans import Translator
from Stock_Prediction_Data_Stock_US import getStocksList

def getSingleStockNewsArticle(er, stock, from_date, till_date, count):
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
    res = er.execQuery(q)

    df = pd.DataFrame(columns=['date', 'title', 'source', 'body', 'uri'])

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
    
    

def updateNewsArticle(er, dir, stock, from_date, till_date, count):
    if os.path.exists(dir) == False: 
        os.makedirs(dir)
    
    filename = dir + stock + '.csv'

    try:
        df = pd.read_csv(filename)
    except:
        df = pd.DataFrame()

    if df.empty:
        df = getSingleStockNewsArticle(er, stock, from_date, till_date, count)
        df.set_index(['date'], inplace=True)
        df.sort_index(ascending=True, inplace=True)
        df.to_csv(filename)
        return 
        
    first_date = pd.Timestamp(df['date'].iloc[0]).tz_localize(None)
    last_date  = pd.Timestamp(df['date'].iloc[-1]).tz_localize(None)

    modified = False

    # require pre download
    if first_date > pd.Timestamp(from_date):
        pre_df = getSingleStockNewsArticle(er, stock, from_date, first_date.strftime("%Y-%m-%d"), count)
        print("pre_df", from_date, first_date.strftime("%Y-%m-%d"))
        print(pre_df)
        df = pd.concat([pre_df, df])
        modified = True
    
    if last_date < pd.Timestamp(till_date):
        post_df = getSingleStockNewsArticle(er, stock, last_date.strftime("%Y-%m-%d"), till_date, count)
        print("post_df", last_date.strftime("%Y-%m-%d"), till_date)
        print(post_df)
        df = pd.concat([df, post_df])
        modified = True

    if modified:
        df.drop_duplicates(subset=['uri'], keep=False)
        df.set_index(['date'], inplace=True)
        df.sort_index(ascending=True, inplace=True)
        print("modified", df)
        df.to_csv(filename)

    
if __name__ == "__main__":
    pd.set_option('precision', 3)
    pd.set_option('display.width',1000)
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    Config = configparser.ConfigParser()
    Config.read("../../config.ini")
    dir = Config.get('Paths', 'NEWS_US')

    #stocklist = getStocksList()
    stocklist = ['AMD', 'WDC', 'SINA', 'WB', 'CTRP', 'NTES', 'ATVI', 'FB', 'GLUU', 'NVDA', 'NFLX', 'GPRO',
                 'MRVL', 'SMCI', 'JD', 'INTC', 'AMZN', 'BIDU', 'BGNE', 'QIWI', 'XNET', 'MOMO', 'YY']


    er = EventRegistry(apiKey = Config.get('EventRegistry', 'KEY'))
    now = datetime.datetime.now().strftime("%Y-%m-%d")

    updateNewsArticle(er, dir, 'MEETME', "2017-06-01", now, 200)

    # for symbol in stocklist:
    #     updateNewsArticle(er, dir, symbol, "2017-05-01", now, 200)
    
