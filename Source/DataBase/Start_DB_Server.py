import os, time, threading, configparser
from subprocess import call

        
def DoStartServer(root_path):
    Config = configparser.ConfigParser()
    Config.read(root_path + "/" + "config.ini")
    dir = Config.get('Paths', 'DATABASE')
    database_path = root_path + "/" + dir

    if os.path.exists(database_path) == False: 
        os.makedirs(database_path)

    call(["mongod", "--quiet", "--repair", "--dbpath", database_path, "--storageEngine", "wiredTiger"])
    #call(["mongod", "--quiet", "--dbpath", database_path])

def StartServer(root_path):
    thread = threading.Thread(target=DoStartServer, args=(root_path, ))
    thread.daemon = True
    thread.start()
    return thread

def SetAdminUser(user, password, database):
    call(["mongo", "--port", "27017", "--host", "localhost", "-u", user, "-p", password, "--authenticationDatabase", database])

def ShutdownServer():
    call(["mongo", "admin", "--eval", "db.shutdownServer()"])
     
if __name__ == "__main__":
    cur_path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(2):
        root_path = cur_path[0:cur_path.rfind('/', 0, len(cur_path))]
        cur_path = root_path

    StartServer(root_path)
    time.sleep(5)
    
    #ShutdownServer()
