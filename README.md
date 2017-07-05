# StockRecommendSystem


## Main Requirement:
Python 3.5.2   
TensorFlow 1.0   
pymongo   
nltk   

## Install
brew install mongodb --with-openssl   
brew services start mongodb   
mongod --dbpath (Your Porject Folder)/Data/DB  

When you storing stock data with mongodb mode, you may meet [too many open files](https://superuser.com/questions/433746/is-there-a-fix-for-the-too-many-open-files-in-system-error-on-os-x-10-7-1) problem, try the following codes in command line:   
> sysctl -w kern.maxfiles=20480 (or whatever number you choose)   
> sysctl -w kern.maxfilesperproc=18000 (or whatever number you choose)   
> launchctl limit maxfiles 1000000 (or whatever number you choose)   
> brew services restart mongodb   
> mongodump -h localhost:27017 -d DB_STOCK -o ./

## Data Fetching:
Cover stock related data fetching, storaging in either MongoDB or CSV mode (See config.ini [Setting] sector for more detail).   
1. Stock:（NSDQ, NYSE）-> US, (HKSE) -> HK, (SSE，SZSE) -> CHN
2. Earning: US stock market earning info.
3. Short: US stock market short squeeze info. (Require Multi IP Routing Support)
4. News: EventRegistry (6.4.3)
5. Media: Twitter Data

### Data Structure

 **US Stock List**   
 DB   : DB_STOCK   
 SHEET: SHEET_US_DAILY_LIST   
 ITEM : symbol, name, market_cap, sector, industry   
   
 **US Stock Daily**   
 DB   : DB_STOCK   
 SHEET: SHEET_US_DAILY_DATA   
 ITEM : symbol (stock symbol)   
        data -> [{date, open, high, low, close, adj_close, volume}]   
        metadata -> {name, last_update}   
      
 **US Stock Earning**   
 DB   : DB_STOCK   
 SHEET: SHEET_US_EARN   
 ITEM : symbol （date）   
        data -> [{date, symbol, analyist, estimate, actual, surprise}]   
   
 **US News**   
 DB   : DB_STOCK   
 SHEET: SHEET_US_NEWS   
 ITEM : symbol (stock symbol)   
        data -> [{date, title, source, body, uri}]   
        metadata -> {last_update}   

### Run
cd Source/FetchData   
python Fetch_Data_Stock_US_Daily.py   

## Stock Prediction:

Under Development...

## Stock Processing:
Correlation
>        Company1 Company2  Correlation  
>            QQQ     TQQQ        0.999
>            IBB      BIB        0.999
>           INSE     XBKS        0.999
>            JAG      JPT        0.999
>           ACWX     VXUS        0.995
>           IXUS     ACWX        0.993
>           VONE      SPY        0.992
>           IXUS     VXUS        0.991
>           VTWO     VTWV        0.988
>            NTB      FBK        0.988
>           GOOG    GOOGL        0.987

### Run
cd Source/StockProcessing   
python Correlation_Stock_US.py   

## Reinforcement Learning:
This sector is directly clone from: [Link](https://github.com/shenyichen105/Deep-Reinforcement-Learning-in-Stock-Trading)

More in mind:
1. The approach use only "Adj Close" price as input, it's supposed more features combinations shall be joined to the party.
2. The Trading Strategy is a little mediocre and limited, better rewrite it.
3. At most only two tickers are allowed in the trading system, rewrite it.

testing output:
> init cash:  100000  
> Columns: [AMD, NVDA, SPY, ^VIX]  
> Index: []  
> Runner: Taking action 2016-03-16 00:00:00 buy  
> Runner: Taking action 2016-03-17 00:00:00 buy  
> Runner: Taking action 2016-03-18 00:00:00 hold  
> ......  
> Runner: Taking action 2017-06-12 00:00:00 buy  
> Runner: Taking action 2017-06-13 00:00:00 buy  
> Runner: Taking action 2017-06-14 00:00:00 buy  
> Final outcome: 121500.348294  

### Run
cd Source/ReinforcementLearning   
python runner.py   

## ToDo:
More AI approach will be arranged and upload ASAP
