import sys, os, time, configparser
from environment import Simulator
from agent import PolicyGradientAgent, CriticsAgent
import datetime
import numpy as np

cur_path = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    root_path = cur_path[0:cur_path.rfind('/', 0, len(cur_path))]
    cur_path = root_path
sys.path.append(root_path + "/" + 'Source/DataBase/')

# def random_run():
#     actions = ["buy", "sell", "hold"]
#     env_train = Simulator(['scg', 'wec'], dt.datetime(2002, 1, 4), dt.datetime(2016, 12, 30))

#     #agent = PolicyGradientAgent(lookback=env_train.init_state())
#     #critic_agent = CriticsAgent(lookback=env.init_state())
#     while env_train.has_more():
#         action = np.random.randint(3)
#         action = actions[action] # map action from id to name
#         print("Runner: Taking action", env_train.date, action)
#         reward, state = env_train.step(action)
#         #action = agent.query(state, reward)

def rf_run(start_date, end_date):
    actions = ["buy", "sell", "hold"]
    
    training_date = (end_date - datetime.timedelta(days=1 * 365))
    n_iter = 5
    for i in range(n_iter):
        env_train = Simulator(['AMD', 'NVDA'], start_date, training_date)
        agent = PolicyGradientAgent(lookback=env_train.init_state())
        #critic_agent = CriticsAgent(lookback=env.init_state())
        action = agent.init_query()

        while env_train.has_more():
            action = actions[action] # map action from id to name
            #print("Runner: Taking action", env_train.date, action)
            reward, state = env_train.step(action)
            action = agent.query(state, reward)

    env_test = Simulator(['AMD', 'NVDA'], training_date, end_date)
    agent.reset(lookback=env_test.init_state())
    while env_test.has_more():
        action = actions[action] # map action from id to name
        #print("Runner: Taking action", env_test.date, action)
        reward, state = env_test.step(action)
        action = agent.query(state, reward)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(root_path + "/" + "config.ini")
    storeType = int(config.get('Setting', 'StoreType'))

    # if storeType == 1:
    #     from Start_DB_Server import StartServer, ShutdownServer
    #     # start database server (async)
    #     thread = StartServer(root_path)
        
    #     # wait for db start, the standard procedure should listen to 
    #     # the completed event of function "StartServer"
    #     time.sleep(5)
    now_date = datetime.datetime.now().date()
    start_date = (now_date - datetime.timedelta(days=10 * 365))#.strftime("%Y-%m-%d")
    end_date = now_date#.strftime("%Y-%m-%d")
    rf_run(start_date, end_date)

    # if storeType == 1:
    #     # stop database server (sync)
    #     time.sleep(5)
    #     ShutdownServer()
    
