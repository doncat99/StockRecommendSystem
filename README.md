# StockRecommendSystem


## Requirement:
Test on Python 3.5.2

## Data Fetching:
Cover Stock and News Fetching
1. Stock:（NSDQ, NYSE）-> US, (HKSE) -> HK, (SSE，SZSE) -> CHN
2. Earning: US stock market earning info.
3. Short: US stock market short squeeze info. (Require Multi IP Routing Support)
4. News: EventRegistry

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
>          LBRDA    LBRDK        0.984
>          LBTYA    LBTYK        0.984
>            BHP      BBL        0.983
>             UN       UL        0.981
>            CCL      CUK        0.980
>          DISCA    DISCK        0.978
>            BIB      BBH        0.976
>            IBB      BBH        0.976
>            FYX     PRFZ        0.976
>            TLT     VGLT        0.974
>           IFGL     VNQI        0.974
>            VIA     VIAB        0.972
>            AIA     AAXJ        0.972
>           DGRW      SPY        0.972

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
> Runner: Taking action 2016-03-21 00:00:00 hold  
> Runner: Taking action 2016-03-22 00:00:00 sell  
> Runner: Taking action 2016-03-23 00:00:00 buy  
> Runner: Taking action 2016-03-24 00:00:00 hold  
> ......  
> Runner: Taking action 2017-06-06 00:00:00 buy  
> Runner: Taking action 2017-06-07 00:00:00 buy  
> Runner: Taking action 2017-06-08 00:00:00 buy  
> Runner: Taking action 2017-06-09 00:00:00 hold  
> Runner: Taking action 2017-06-12 00:00:00 buy  
> Runner: Taking action 2017-06-13 00:00:00 buy  
> Runner: Taking action 2017-06-14 00:00:00 buy  
> Final outcome: 121500.348294  


## ToDo:
More AI approach will be arranged and upload ASAP
