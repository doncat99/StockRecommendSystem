import os, time, datetime, logging, warnings
import pandas as pd

from DataBase.Start_DB_Server import SetAdminUser, StartServer, ShutdownServer
from FetchData.Fetch_Data_Stock_US_Daily import updateStockData_US, queryStockList

if __name__ == "__main__":    
    pd.set_option('precision', 3)
    pd.set_option('display.width',1000)
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    logging.basicConfig(level=logging.ERROR)
    cur_path = os.path.dirname(os.path.abspath(__file__))
    root_path = cur_path[0:cur_path.rfind('/', 0, len(cur_path))]

    # start database server (async)
    
    thread = StartServer(root_path)
    
    # wait for db start, the standard procedure should listen to 
    # the completed event of function "StartServer"
    time.sleep(3)


    # symbols = queryStockList(root_path, "STOCK_US")


    from arctic import Arctic
    store = Arctic("localhost")
    database = "STOCK_US"
    #store.delete_library(database)
    try:
        library = store[database]
    except:
        store.initialize_library(database)
        library = store[database]
    

    # library.delete("StockList")
    # library.delete("StockPublishDay")
    
    # item = library.read('CIZN')
    # print(item.data)
    # item = library.read('ELEC')
    # print(item.data)
    # item = library.read('DJCO')
    # print(item.data)
    # item = library.read('BLVD')
    # print(item.data)
    # item = library.read('KEN')
    # print(item.data)

    now = datetime.datetime.now().strftime("%Y-%m-%d")
    updateStockData_US(root_path, "1990-01-01", now)


    # stop database server (sync)
    time.sleep(3)
    ShutdownServer()

