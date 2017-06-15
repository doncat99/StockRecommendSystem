from environment import Simulator
from agent import PolicyGradientAgent, CriticsAgent
import datetime as dt
import numpy as np

def random_run():
    actions = ["buy", "sell", "hold"]
    env_train = Simulator(['scg', 'wec'], dt.datetime(2002, 1, 4), dt.datetime(2016, 12, 30))

    #agent = PolicyGradientAgent(lookback=env_train.init_state())
    #critic_agent = CriticsAgent(lookback=env.init_state())
    while env_train.has_more():
        action = np.random.randint(3)
        action = actions[action] # map action from id to name
        print("Runner: Taking action", env_train.date, action)
        reward, state = env_train.step(action)
        #action = agent.query(state, reward)

def rf_run():
    actions = ["buy", "sell", "hold", "gameover"]
    
    n_iter = 1
    for i in range(n_iter):
        env_train = Simulator(['AMD', 'NVDA'], dt.datetime(2006, 1, 4), dt.datetime(2016, 12, 30))
        agent = PolicyGradientAgent(lookback=env_train.init_state())
        #critic_agent = CriticsAgent(lookback=env.init_state())
        action = agent.init_query()

        while env_train.has_more():
        	action = actions[action] # map action from id to name
        	#print("Runner: Taking action", env_train.date, action)
        	reward, state = env_train.step(action)
        	action = agent.query(state, reward)

    env_test = Simulator(['AMD', 'NVDA'], dt.datetime(2016, 1, 4), dt.datetime(2017, 6, 14))
    agent.reset(lookback=env_test.init_state())
    while env_test.has_more():
        action = actions[action] # map action from id to name
        print("Runner: Taking action", env_test.date, action)
        reward, state = env_test.step(action)
        action = agent.query(state, reward)
        if action == actions[3]:
            print("Game over :p) ---> Run out of cash, wahaha...")
            break

if __name__ == '__main__':
    rf_run()
