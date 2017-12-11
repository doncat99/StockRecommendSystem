import os, datetime, numpy, configparser

class SP_Global_Paras(object):
    
    def __init__(self, name, root_path, train_tickers, predict_tickers):
        self._name = name
        #self._identify = name + '_' + ''.join(tickers) + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S"))
        self._identify = name + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        self._config = configparser.ConfigParser()
        self._config.read(root_path + "/" + "config.ini")
        self._root_path = root_path
        self._save_folder = ''
        self._model_folder = ''
        self._stock_folder = ''
        self._model_name = name + '_model'
        self._save = False
        self._load = False
        self._run_hyperopt = False
        self._plot = False
        self._verbose = 0

        # ------------- INPUT -------------
        self._train_tickers = train_tickers
        self._predict_tickers = predict_tickers
        self._features = {'0_0':['ohlcv']}
        self._window_len = 120
        self._pred_len = 10
        self._valid_len = 20

        self._tree_min = [4]
        self._tree_max = [8]
        self._feature_min = [20]
        self._feature_max = [25]
        self._window_min = 1
        self._window_max = 5

        self._space = {}
        
        # ------------- OUTPUT -------------
        self._out_class_type = 'regression'
        self._n_out_class = 5
        
        self._start_date = '2010-01-01'
        self._end_date = 'current'
        
    def __str__(self):
        returnString = ('%%%%%%%%%% DUMP SP_Global_Paras %%%%%%%%%%\n' + 
                        'name \t' + str(self._name) + '\n' + 
                        'identify \t' + str(self._identify) + '\n' + 
                        'save \t' + str(self._save) + '\n' + 
                        'save_folder \t' + str(self._save_folder) + '\n' + 
                        #'ticker \t' + str(self._tickers) + '\n' +
                        'features \t' + str(self._features) + '\n' +
                        'window_len \t' + str(self._window_len) + '\n' +
                        'pred_len \t' + str(self._pred_len) + '\n' +
                        'valid_len \t' + str(self._valid_len) + '\n' +
                        'out_class_type \t' + str(self._out_class_type) + '\n' +
                        'n_out_class \t' + str(self._n_out_class) + '\n' +
                        'start_date \t' + str(self._start_date) + '\n')# +
                        #'end_date \t' + str(self._end_date) + '\n')
        if self._end_date == 'current':
            returnString = returnString + 'end_date \t' + str(datetime.date.today()) + '\n'
        else:
            returnString = returnString + 'end_date \t' + str(self._end_date) + '\n'
        return returnString
    
    @property
    def identify(self):
        return self._identify
    @identify.setter
    def identify(self, value):
        self._identify = value
        
    @property
    def config(self):
        return self._config
    @config.setter
    def config(self, value):
        self._config = value

    @property
    def root_path(self):
        return self._root_path
    @root_path.setter
    def root_path(self, value):
        self._root_path = value

    @property
    def save_folder(self):
        return self._save_folder
    @save_folder.setter
    def save_folder(self, value):
        self._save_folder = value

    @property
    def model_folder(self):
        return self._model_folder
    @model_folder.setter
    def model_folder(self, value):
        self._model_folder = value

    @property
    def stock_folder(self):
        return self._stock_folder
    @stock_folder.setter
    def stock_folder(self, value):
        self._stock_folder = value

    @property
    def model_name(self):
        return self._model_name
    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    @property
    def save(self):
        return self._save
    @save.setter
    def save(self, value):
        self._save = value

    @property
    def load(self):
        return self._load
    @load.setter
    def load(self, value):
        self._load = value
    
    @property
    def run_hyperopt(self):
        return self._run_hyperopt
    @run_hyperopt.setter
    def run_hyperopt(self, value):
        self._run_hyperopt = value

    @property
    def plot(self):
        return self._plot
    @plot.setter
    def plot(self, value):
        self._plot = value

    @property
    def verbose(self):
        return self._verbose
    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    @property
    def train_tickers(self):
        return self._train_tickers
#     @ticker.setter
#     def ticker(self, value):
#         self._ticker = value

    @property
    def predict_tickers(self):
        return self._predict_tickers

    @property
    def features(self):
        return self._features
    @features.setter
    def features(self, value):
        self._features = value
        
    @property
    def window_len(self):
        return self._window_len
    @window_len.setter
    def window_len(self, value):
        self._window_len = value
        
    @property
    def pred_len(self):
        return self._pred_len
    @pred_len.setter
    def pred_len(self, value):
        self._pred_len = value
        
    @property
    def valid_len(self):
        return self._valid_len
    @valid_len.setter
    def valid_len(self, value):
        self._valid_len = value

    @property
    def tree_min(self):
        return self._tree_min
    @tree_min.setter
    def tree_min(self, value):
        self._tree_min = value        

    @property
    def tree_max(self):
        return self._tree_max
    @tree_max.setter
    def tree_max(self, value):
        self._tree_max = value   

    @property
    def feature_min(self):
        return self._feature_min
    @feature_min.setter
    def feature_min(self, value):
        self._feature_min = value  

    @property
    def feature_max(self):
        return self._feature_max
    @feature_max.setter
    def feature_max(self, value):
        self._feature_max = value   

    @property
    def window_min(self):
        return self._window_min
    @window_min.setter
    def window_min(self, value):
        self._window_min = value  

    @property
    def window_max(self):
        return self._window_max
    @window_max.setter
    def window_max(self, value):
        self._window_max = value  

    @property
    def hyper_opt(self):
        return self._hyper_opt
    @hyper_opt.setter
    def hyper_opt(self, value):
        self._hyper_opt = value

    @property
    def out_class_type(self):
        return self._out_class_type
    @out_class_type.setter
    def out_class_type(self, value):
        self._out_class_type = value
        
    @property
    def n_out_class(self):
        return self._n_out_class
    @n_out_class.setter
    def n_out_class(self, value):
        self._n_out_class = value
        
    @property
    def start_date(self):
        return self._start_date
    @start_date.setter
    def start_date(self, value):
        self._start_date = value
        
    @property
    def end_date(self):
        if self._end_date == 'current':
            return str(datetime.date.today())
        else:
            return self._end_date
    @end_date.setter
    def end_date(self, value):
        self._end_date = value
        
    @property
    def n_features(self):
        return numpy.sum([len(group) for key, group in self.features.items()])

class SP_Paras(SP_Global_Paras):
    
    def __init__(self, name, root_path, train_tickers = [], predict_tickers = []):
        super(SP_Paras, self).__init__(name, root_path = root_path, train_tickers = train_tickers, predict_tickers = predict_tickers)

        # ------------- LSTM -------------
        self._batch_size = 32
        self._epoch = 10
        self._validation_split = .2
        self._model = {
            'hidden_layers' : [],
            'dropout' : [],
            'activation' : [],
            'out_layer' : 1,
            'out_activation' : 'linear',
            'loss' : 'mse',
            'optimizer' : 'rmsprop'
        }
    
    def __str__(self):
        returnString = (super(SP_Paras, self).__str__() + '\n' +
                        '%%%%%%%%%% DUMP SP_Paras %%%%%%%%%%\n' + 
                        'batch_size \t' + str(self._batch_size) + '\n' +
                        'epoch \t' + str(self._epoch) + '\n' +
                        'validation_split \t' + str(self._validation_split) + '\n' #+
                        # 'hidden_layers \t' + str(self._model['hidden_layers']) + '\n'
                        # 'dropout \t' + str(self._model['dropout']) + '\n'
                        # 'activation \t' + str(self._model['activation']) + '\n'
                        #'out_layer \t' + str(self._model['out_layer']) + '\n'
                        # 'out_activation \t' + str(self._model['out_activation']) + '\n'
                        # 'loss \t' + str(self._model['loss']) + '\n'
                        # 'optimizer \t' + str(self._model['optimizer']) + '\n'
                       )
        return returnString
        
    @property
    def batch_size(self):
        return self._batch_size
    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
        
    @property
    def epoch(self):
        return self._epoch
    @epoch.setter
    def epoch(self, value):
        self._epoch = value
    
    @property
    def validation_split(self):
        return self._validation_split
    @validation_split.setter
    def validation_split(self, value):
        self._validation_split = value
    
    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, value):
        self._model = value


class base_model(object):
    def __init__(self, paras):
        self.paras = paras
        self.df = []

    def get_file_id(self):
        return self.paras.identify

    def get_save_directory(self):
        history_folder = self.paras.config.get('Paths', 'ML_HISTORY')
        dir = self.paras.root_path + '/' + history_folder
        if os.path.exists(dir) == False: os.makedirs(dir)
        file_id = self.get_file_id()
        save_folder = str(dir) + str(file_id)
        os.makedirs(save_folder)
        return (save_folder + '/')

    def get_model_directory(self):
        model_folder = self.paras.config.get('Paths', 'ML_MODEL')
        dir = self.paras.root_path + '/' + model_folder
        if os.path.exists(dir) == False: os.makedirs(dir)
        return dir

    def get_model_name(self, window_len):
        return self.paras.model_name + '_' + str(self.paras.n_out_class) + '_' + str(window_len) + '_' + str(self.paras.pred_len)

    
