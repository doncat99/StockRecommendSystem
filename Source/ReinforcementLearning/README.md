# Deep-Reinforcement-Learning-in-Stock-Trading
Using deep actor-critic model to learn best strategies in pair trading 

## Abstract
Partially observed Markov decision process problem of pairs trading is a challenging aspect in algorithmic trading. In this work, we tackle this by utilizing a deep reinforcement learning algorithm called advantage actor-critic by extending the policy network with a critic network, to incorporate both the stochastic policy gradient and value gradient. We have also used recurrent neural network coupled with long-short term memory to preserve information from time series data of stock market. A memory buffer for experience replay and a target network are also employed to reduce the variance from noisy and correlated environment. Our results demonstrate a success on learning a well-performing lucrative model by directly taking data from public available sources and present possibilities for extensions to other time-sensitive applications

## Source
https://github.com/shenyichen105/Deep-Reinforcement-Learning-in-Stock-Trading

## Installation
This project is implemented in python. In order to run below dependencies need to install first:

- theano
- lasagne
- numpy
- statsmodels
- pandas

All these python modules can be easily installed through Anaconda, a python package manage tool free available for [download](https://www.continuum.io/downloads).

```
conda install theano lasagne numpy statsmodels pandas
```

It's highly recommended to run the code on a GPU server as it's very time-consuming to run on CPU machine.

## Declaration
All the Source in Reinforcement Learning folder are originally clone from the 
"Source" sector above. Necessary modifications are made as to adapt to Data folder.