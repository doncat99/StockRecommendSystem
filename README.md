# StockRecommendSystem


## Main Requirement:
Python 3.5.2
TensorFlow 1.0
nltk

## Data Fetching:
Cover Stock and News Fetching
1. Stock:（NSDQ, NYSE）-> US, (HKSE) -> HK, (SSE，SZSE) -> CHN
2. Earning: US stock market earning info.
3. Short: US stock market short squeeze info. (Require Multi IP Routing Support)
4. News: EventRegistry
5. Media: Twitter Data 


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

For the whole correlation list, run it and check "Result" folder for it.

## Reinforment Learning:
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


## ToDo:
More AI approach will be arranged and upload ASAP
