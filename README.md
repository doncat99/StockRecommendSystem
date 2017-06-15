# StockRecommendSystem


## Requirement
Test on Python 3.5.2

## Data Fetching:
Cover Stock and News Fetching
1. Stock:（NSDQ, NYSE）-> US, (HKSE) -> HK, (SSE，SZSE) -> CHN
2. Earning: US stock market earning info.
3. Short: US stock market short squeeze info. (Require Multi IP Routing Support)
4. News: EventRegistry

## Reinforment Learning:
The mainly implementation is directly clone from: [Link](https://github.com/shenyichen105/Deep-Reinforcement-Learning-in-Stock-Trading)

More in mind:
1. The approach use only "Adj Close" price as input, it's supposed more features combinations shall be joined to the party.
2. The Trading Strategy is a little mediocre and limited, better rewrite it.
3. At most only two tickers are allowed in the trading system, rewrite it.


> init cash:  100000  
> Columns: [AMD, NVDA, SPY, ^VIX]  
> Index: []  
Runner: Taking action 2016-03-16 00:00:00 buy  
Runner: Taking action 2016-03-17 00:00:00 buy  
Runner: Taking action 2016-03-18 00:00:00 hold  
Runner: Taking action 2016-03-21 00:00:00 hold  
Runner: Taking action 2016-03-22 00:00:00 sell  
Runner: Taking action 2016-03-23 00:00:00 buy  
Runner: Taking action 2016-03-24 00:00:00 hold  
......  
Runner: Taking action 2017-06-06 00:00:00 buy  
Runner: Taking action 2017-06-07 00:00:00 buy  
Runner: Taking action 2017-06-08 00:00:00 buy  
Runner: Taking action 2017-06-09 00:00:00 hold  
Runner: Taking action 2017-06-12 00:00:00 buy  
Runner: Taking action 2017-06-13 00:00:00 buy  
Runner: Taking action 2017-06-14 00:00:00 buy  
Final outcome: 121500.348294  


## ToDo:
More AI approach will be arranged and upload ASAP
