from environment import Simulator
from agent import PolicyGradientAgent, CriticsAgent
import datetime as dt
import numpy as np

def main():
    actions = ["buy", "sell", "hold"]

    n_iter = 5
    for i in range(n_iter):

        env_train = Simulator(['AMD', 'NVDA'], dt.datetime(2001, 1, 4), dt.datetime(2013, 12, 30))

        agent = PolicyGradientAgent(lookback=env_train.init_state())
        #critic_agent = CriticsAgent(lookback=env.init_state())
        action = agent.init_query()

        while env_train.has_more():
        	action = actions[action] # map action from id to name
        	#print("Runner: Taking action", env_train.date, action)
        	reward, state = env_train.step(action)
        	action = agent.query(state, reward)


    env_test = Simulator(['AMD', 'NVDA'], dt.datetime(2014, 1, 4), dt.datetime(2017, 6, 14))
    agent.reset(lookback=env_test.init_state())
    while env_test.has_more():
        action = actions[action] # map action from id to name
        print("Runner: Taking action", env_test.date, action)
        reward, state = env_test.step(action)
        action = agent.query(state, reward)

if __name__ == '__main__':
    main()
