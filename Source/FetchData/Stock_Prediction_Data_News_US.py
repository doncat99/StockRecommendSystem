#install eventregistry using "pip install eventregistry" - see https://github.com/gregorleban/EventRegistry
import os, configparser
import pandas as pd
from eventregistry import *
from googletrans import Translator

def updateNewsArticle(er, dir, stock, from_date, till_date, count):
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

    if 'info' in res:
        print(res["info"])
        return

    if os.path.exists(dir) == False: 
        os.makedirs(dir)
    
    filename = dir + stock + '.csv'

    df = getNewsDataFrame(filename, stock)

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
    
    df.drop_duplicates(subset=['uri'], keep=False)
    df.set_index(['date'], inplace=True)
    df.sort_index(ascending=True, inplace=True)
    df.to_csv(filename)

def getNewsDataFrame(dir, stock):
    try:
        return pd.read_csv(filename)
    except:
        return pd.DataFrame(columns=['date', 'title', 'source', 'body', 'uri'])

    
if __name__ == "__main__":
    Config = configparser.ConfigParser()
    Config.read("../../config.ini")
    dir = Config.get('Paths', 'NEWS_US')

    er = EventRegistry(apiKey = Config.get('EventRegistry', 'KEY'))
    updateNewsArticle(er, dir, "GPRO", "2017-06-01", "2017-06-10", 100)
    
