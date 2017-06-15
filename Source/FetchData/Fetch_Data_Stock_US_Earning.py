import os, time, datetime, requests, warnings, configparser
import pandas as pd
from bs4 import BeautifulSoup
import concurrent.futures
from tqdm import tqdm

URL_EARNING = 'http://www.nasdaq.com/earnings/earnings-calendar.aspx?date='

def getStockEarningByDate(date):
    repeat_times = 1
    message = ""
    
    for _ in range(repeat_times): 
        try:
            resp = requests.get(URL_EARNING + date, timeout=15)
            return resp.text, message
        except Exception as e:
            message = date + " fetch exception: " + str(e)
            continue   
        else:
            time.sleep(0.1)
    return '', message


def getStockCodeFromCompanyName(companyName):
    leftIndex = companyName.rfind('(')
    if leftIndex == -1: return ''

    rightIndex = companyName.rfind(')')
    if rightIndex == -1: return ''

    return companyName[leftIndex+1:rightIndex]


def parseEarningsColumn(column):
    earning_data = []
    
    if len(column) < 7: return earning_data
    
    # Date
    earning_data.append(column[1])

    # Symbol
    symbol = getStockCodeFromCompanyName(column[0])
    earning_data.append(symbol)

    # Num of Analyist
    earning_data.append(column[4])
    
    # Estimate value
    column[3] = column[3][1:]
    if column[3] == 'n/a': column[3] = '0'
    earning_data.append(column[3])

    # Actual value
    column[5] = column[5][1:]
    if column[5] == 'n/a': column[5] = '0'
    earning_data.append(column[5])

    # Surprise
    if column[6] == 'N/A' or column[6] == 'Met': column[6] = '0'
    earning_data.append(column[6])

    return earning_data


def convertEarningsToDataFrame(responseText):
    df = pd.DataFrame(columns=["Date", "Code", "Analyist", "Estimate", "Actual", "Surprise"])

    soup = BeautifulSoup(responseText, 'html.parser')
    table = soup.find('table', {'class': 'USMN_EarningsCalendar'})
    
    if table == None:
        return df

    tree = table.find_all('tr')

    for tr in tree:
        if tr.find('th'):
            continue
        tds = tr.find_all('td')
        column = []
        for td in tds:
            text = td.get_text().strip()
            if len(text) == 0 or text == 'n/a':
                continue
            column.append(text)
        
        row = parseEarningsColumn(column)

        if len(row) == 0: continue
        df.loc[len(df)] = row

    return df

def updateEarnings_US(duedays):
    Config = configparser.ConfigParser()
    Config.read("../../config.ini")
    dir = Config.get('Paths', 'EARNING_US')
    
    if os.path.exists(dir) == False: 
        os.makedirs(dir)

    pbar = tqdm(total=len(duedays))

    for date in duedays:
        date = date.strftime("%Y-%m-%d")
        startTime = time.time()
        filename = dir + date + ".csv"

        if os.path.exists(filename): continue

        text, message = getStockEarningByDate(date)
        if len(message) > 0: continue

        df = convertEarningsToDataFrame(text)
        if df.empty: continue
        df.to_csv(filename)

        outMessage = '%-*s fetched in:  %.4s seconds' % (20, date, (time.time() - startTime))
        pbar.set_description(outMessage)
        pbar.update(1)


if __name__ == "__main__":
    pd.set_option('precision', 3)
    pd.set_option('display.width',1000)
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    date_start = datetime.datetime(2012, 1, 3)
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    duedays = pd.date_range(date_start, now)
    updateEarnings_US(duedays)

