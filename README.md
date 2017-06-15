# StockRecommendSystem

Test on Python 3.5.2

## Data Fetching:
1. Cover Stock and News Fetching
2. Stock:（NSDQ, NYSE）-> US, (HKSE) -> HK, (SSE，SZSE) -> CHN
3. Short: US. market short stock info. (Require Multi Ip Support)
4. News: EventRegistry

## Reinforment Learning:
The mainly implementation is directly clone from:
https://github.com/shenyichen105/Deep-Reinforcement-Learning-in-Stock-Trading

The approach currently use one feature of the data, which is "close price" as input.
It's supposed there shall be many available combinations we can play and test with.

init cash:  100000
Columns: [AMD, NVDA, SPY, ^VIX]
Index: []
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


More AI approach will be arranged and upload ASAP
