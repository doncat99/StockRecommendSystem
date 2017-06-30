import sys, os, time, datetime, requests, warnings, configparser
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from bs4 import BeautifulSoup
import concurrent.futures
from tqdm import tqdm

cur_path = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    root_path = cur_path[0:cur_path.rfind('/', 0, len(cur_path))]
    cur_path = root_path
sys.path.append(root_path + "/" + 'Source/DataBase/')
from DB_API import queryEarnings, storeEarnings

def getStockEarningByDate(date):
    repeat_times = 1
    message = ""
    
    for _ in range(repeat_times): 
        try:
            URL_EARNING = 'http://www.nasdaq.com/earnings/earnings-calendar.aspx?date='
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

def updateEarningByDate(root_path, date):
    startTime = time.time()
    date = date.strftime("%Y-%m-%d")
    try:
        df = queryEarnings(root_path, "DB_STOCK", "SHEET_US_EARN", date)
        if df.empty == False: return startTime
    except:
        df = pd.DataFrame()
    
    if df.empty:
        text, message = getStockEarningByDate(date)
        if len(message) > 0: return startTime

    df = convertEarningsToDataFrame(text)
    storeEarnings(root_path, "DB_STOCK", "SHEET_US_EARN", date, df)
    return startTime


def updateEarnings_US(duedays, storeType):

    pbar = tqdm(total=len(duedays))

    if storeType == 2:
        for date in duedays:
            startTime = updateEarningByDate(root_path, date)
            outMessage = '%-*s fetched in:  %.4s seconds' % (12, date, (time.time() - startTime))
            pbar.set_description(outMessage)
            pbar.update(1)
        
    if storeType == 1:
        log_errors = []
        log_update = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Start the load operations and mark each future with its URL
            future_to_stock = {executor.submit(updateEarningByDate, root_path, date): date for date in duedays}
            for future in concurrent.futures.as_completed(future_to_stock):
                date = future_to_stock[future]
                try:
                    startTime = future.result()
                except Exception as exc:
                    startTime = time.time()
                    log_errors.append('%r generated an exception: %s' % (stock, exc))
                    len_errors = len(log_errors)
                    if len_errors % 5 == 0: print(log_errors[(len_errors-5):]) 
                outMessage = '%-*s fetched in:  %.4s seconds' % (12, date, (time.time() - startTime))
                pbar.set_description(outMessage)
                pbar.update(1)
        if len(log_errors) > 0: print(log_errors)
        # if len(log_update) > 0: print(log_update)
    
    pbar.close()

if __name__ == "__main__":
    pd.set_option('precision', 3)
    pd.set_option('display.width',1000)
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    date_start = "2012-01-03"
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    duedays = pd.DatetimeIndex(start=date_start, end=now, freq=us_bd)

    config = configparser.ConfigParser()
    config.read(root_path + "/" + "config.ini")
    storeType = int(config.get('Setting', 'StoreType'))

    if storeType == 1:
        from Start_DB_Server import StartServer, ShutdownServer
        # start database server (async)
        thread = StartServer(root_path)
        
        # wait for db start, the standard procedure should listen to 
        # the completed event of function "StartServer"
        time.sleep(5)
    
    updateEarnings_US(duedays, storeType)

    if storeType == 1:
        # stop database server (sync)
        time.sleep(5)
        ShutdownServer()


    

