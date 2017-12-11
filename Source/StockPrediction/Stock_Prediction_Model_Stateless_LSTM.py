import os, csv
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, History
from keras import optimizers
import matplotlib.pyplot as plt  # http://matplotlib.org/examples/pylab_examples/subplots_demo.html
import pandas as pd
import numpy as np
import pickle

from hyperopt import fmin, tpe, partial

from Stock_Prediction_Base import base_model
from Stock_Prediction_Data_Processing import reshape_input, get_all_stocks_feature_data, preprocessing_data, kmeans_claasification, get_all_target_dict, preprocessing_train_data, kmeans_claasification


class lstm_model(base_model):
    train_x = None
    train_y = None
    test_x  = None
    test_y  = None
        
    def lstm_model(self):
        model = Sequential()
        first = True
        for idx in range(len(self.paras.model['hidden_layers'])):
            if idx == (len(self.paras.model['hidden_layers']) - 1):
                model.add(LSTM(int(self.paras.model['hidden_layers'][idx]), return_sequences=False))
                model.add(Activation(self.paras.model['activation']))
                model.add(Dropout(self.paras.model['dropout']))
            elif first == True:
                model.add(LSTM(input_shape=(None, int(self.paras.n_features)),
                               units=int(self.paras.model['hidden_layers'][idx]),
                               return_sequences=True))
                model.add(Activation(self.paras.model['activation']))
                model.add(Dropout(self.paras.model['dropout']))
                first = False
            else:
                model.add(LSTM(int(self.paras.model['hidden_layers'][idx]), return_sequences=True))
                model.add(Activation(self.paras.model['activation']))
                model.add(Dropout(self.paras.model['dropout']))

        if self.paras.model['optimizer'] == 'sgd':
            #optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            optimizer = optimizers.SGD(lr=self.paras.model['learning_rate'], decay=1e-6, momentum=0.9, nesterov=True)
        elif self.paras.model['optimizer'] == 'rmsprop':
            #optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
            optimizer = optimizers.RMSprop(lr=self.paras.model['learning_rate']/10, rho=0.9, epsilon=1e-08, decay=0.0)
        elif self.paras.model['optimizer'] == 'adagrad':
            #optimizer = optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
            optimizer = optimizers.Adagrad(lr=self.paras.model['learning_rate'], epsilon=1e-08, decay=0.0)
        elif self.paras.model['optimizer'] == 'adam':
            #optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            optimizer = optimizers.Adam(lr=self.paras.model['learning_rate']/10, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        elif self.paras.model['optimizer'] == 'adadelta':
            optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        elif self.paras.model['optimizer'] == 'adamax':
            optimizer = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        elif self.paras.model['optimizer'] == 'nadam':
            optimizer = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        else:
            optimizer = optimizers.Adam(lr=self.paras.model['learning_rate']/10, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        # output layer
        model.add(Dense(units=self.paras.model['out_layer']))
        model.add(Activation(self.paras.model['out_activation']))
        model.compile(loss=self.paras.model['loss'], optimizer=optimizer, metrics=['accuracy'])

        return model

    def LSTM(self, argsDict):
        self.paras.batch_size             = argsDict["batch_size"]
        self.paras.model['dropout']       = argsDict['dropout']
        self.paras.model['activation']    = argsDict["activation"]
        self.paras.model['optimizer']     = argsDict["optimizer"]
        self.paras.model['learning_rate'] = argsDict["learning_rate"]

        print(self.paras.batch_size, self.paras.model['dropout'], self.paras.model['activation'], self.paras.model['optimizer'], self.paras.model['learning_rate'])

        model = self.lstm_model()
        model.fit(self.train_x, self.train_y,
              batch_size=self.paras.batch_size,
              epochs=self.paras.epoch,
              verbose=0,
              callbacks=[EarlyStopping(monitor='loss', patience=5)])

        score, mse = model.evaluate(self.test_x, self.test_y, verbose=0)
        return -mse

    def best_model(self, X_train, y_train, X_test, y_test):
        self.train_x = X_train
        self.train_y = y_train
        self.test_x  = X_test
        self.test_y  = y_test

        algo = partial(tpe.suggest, n_startup_jobs=1)
        best = fmin(self.LSTM, space=self.paras.hyper_opt, algo=algo, max_evals=100)
        print("best", best)
        return best
        
    def build_model(self, window, X_train, y_train, X_test, y_test):
        if self.paras.load == True:
            model = self.load_training_model()
            if model != None:
                return model

        best = {}
        file_name = "hyper_parameter_lstm_" + str(window) + ".pkl"

        if self.paras.run_hyperopt == True:
            print('find hyper parameters...')
            best = self.best_model(X_train, y_train, X_test, y_test)
            pickle.dump(best, open(file_name, "wb"))
        else:
            if os.path.exists(file_name):
                best = pickle.load(open(file_name, "rb"))

        if len(best) != 0:
            self.paras.batch_size             = self.paras.hyper_opt['batch_size_opt'][best["batch_size"]]
            self.paras.model['dropout']       = best['dropout']
            self.paras.model['activation']    = self.paras.hyper_opt['activation_opt'][best["activation"]]
            self.paras.model['optimizer']     = self.paras.hyper_opt['optimizer_opt'][best["optimizer"]]
            self.paras.model['learning_rate'] = best["learning_rate"]

        print('build LSTM model...')
        return self.lstm_model()

    def save_training_model(self, model, window_len):
        if self.paras.save == True:
            print('save LSTM model...')
            # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
            model.save(self.paras.model_folder + self.get_model_name(window_len) + '.h5')  # creates a HDF5 file 'my_model.h5'

    def load_training_model(self, window_len):
        # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
        model_file = self.paras.model_folder + self.get_model_name(window_len) + '.h5'
        if os.path.exists(model_file):
            print('load LSTM model...')
            return load_model(model_file)  # creates a HDF5 file 'my_model.h5'
        return None

    def plot_training_curve(self, history):
        #         %matplotlib inline
        #         %pylab inline
        #         pylab.rcParams['figure.figsize'] = (15, 9)   # Change the size of plots

        # LSTM training
        f, ax = plt.subplots()
        ax.plot(history.history['loss'])
        #ax.plot(history.history['val_loss'])
        ax.set_title('loss function')
        ax.set_ylabel('mse')
        ax.set_xlabel('epoch')
        #ax.legend(['loss', 'val_loss'], loc='upper right')
        ax.legend(['loss'], loc='upper right')
        plt.show()
        if self.paras.save == True:
            w = csv.writer(open(self.paras.save_folder + 'training_curve_model.txt', 'w'))
            for key, val in history.history.items():
                w.writerow([key, val])
            for key, val in history.params.items():
                w.writerow([key, val])

# Classification
class rnn_lstm_classification(lstm_model):
    def __init__(self, paras):
        super(rnn_lstm_classification, self).__init__(paras=paras)

    def check_parameters(self):
        if (self.paras.out_class_type == 'classification' and self.paras.n_out_class > 1 and
                    self.paras.model['out_activation'] == 'softmax' and self.paras.model[
            'loss'] == 'categorical_crossentropy'):
            return True
        return False


    ###################################
    ###                             ###
    ###          Training           ###
    ###                             ###
    ###################################

    def prepare_train_test_data(self, data_feature, LabelColumnName):
        firstloop = 1
        for ticker, data in data_feature.items():
            #print(ticker, "n_feature", self.paras.n_features, len(data[0]))
            X, y = preprocessing_data(self.paras, data[0], LabelColumnName, one_hot_label_proc=True)
            X, y = reshape_input(self.paras.n_features, X, y)
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y, test_size=0.2)
            # print('Train shape X:', X_train_temp.shape, ',y:', y_train_temp.shape)
            # print('Test shape X:', X_test_temp.shape, ',y:', y_test_temp.shape)

            if firstloop == 1:
                firstloop = 0
                X_train = X_train_temp
                X_test = X_test_temp
                y_train = y_train_temp
                y_test = y_test_temp
            else:
                X_train = np.append(X_train, X_train_temp, 0)
                X_test = np.append(X_test, X_test_temp, 0)
                y_train = np.append(y_train, y_train_temp, 0)
                y_test = np.append(y_test, y_test_temp, 0)

        # print('Train shape X:', X_train.shape, ',y:', y_train.shape)
        # print('Test shape X:', X_test.shape, ',y:', y_test.shape)
        return X_train, y_train, X_test, y_test

    def prepare_train_data(self, data_feature, LabelColumnName):
        firstloop = 1
        print("get_data_feature")
        #print(data_feature.items())

        train_tickers_dict = get_all_target_dict()
        train_symbols = train_tickers_dict.keys()

        for ticker, data in data_feature.items():
            # print(ticker, "n_feature", self.paras.n_features, len(data[0]))
            # print("data[0]",data[0].head())
            #print("data[0]", data[0].index)
            if ticker not in train_symbols: continue

            X, y = preprocessing_train_data(self.paras, data[0], LabelColumnName, ticker, train_tickers_dict, one_hot_label_proc=True)

            if len(X) == 0 or len(y) == 0: continue
            # print(X.shape)
            X, y = reshape_input(self.paras.n_features, X, y)
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y, test_size=0.2)
            # print('Train shape X:', X_train_temp.shape, ',y:', y_train_temp.shape)
            # print('Test shape X:', X_test_temp.shape, ',y:', y_test_temp.shape)

            if firstloop == 1:
                firstloop = 0
                X_train = X_train_temp
                X_test = X_test_temp
                y_train = y_train_temp
                y_test = y_test_temp
            else:
                X_train = np.append(X_train, X_train_temp, 0)
                X_test = np.append(X_test, X_test_temp, 0)
                y_train = np.append(y_train, y_train_temp, 0)
                y_test = np.append(y_test, y_test_temp, 0)

        # print('Train shape X:', X_train.shape, ',y:', y_train.shape)
        # print('Test shape X:', X_test.shape, ',y:', y_test.shape)
        return X_train, y_train, X_test, y_test

    def train_data(self, data_feature, window, LabelColumnName):
        history = History()
        
        #X_train, y_train, X_test, y_test = self.prepare_train_test_data(data_feature, LabelColumnName)
        X_train, y_train, X_test, y_test = self.prepare_train_data(data_feature, LabelColumnName)
        model = self.build_model(window, X_train, y_train, X_test, y_test)

        model.fit(
            X_train,
            y_train,
            batch_size=self.paras.batch_size,
            epochs=self.paras.epoch,
            # validation_split=self.paras.validation_split,
            # validation_data = (X_known_lately, y_known_lately),
            callbacks=[history],
            # shuffle=True,
            verbose=self.paras.verbose
        )
        # save model
        self.save_training_model(model, window)

        # print(' ############## validation on test data ############## ')
        mse_test, tmp = self.predict(model, X_test, y_test)

        # plot training loss/ validation loss
        if self.paras.plot:
            self.plot_training_curve(history)

        return model


    ###################################
    ###                             ###
    ###         Predicting          ###
    ###                             ###
    ###################################

    def predict(self, model, X, y):
        predictions = model.predict(X)
        mse_scaled = np.mean((y - predictions) ** 2)
        # print('scaled data mse: ', mse_scaled)
        return mse_scaled, predictions


    def predict_data(self, model, data_feature, window, LabelColumnName):

        if model == None: model = self.load_training_model(window)

        if model == None:
            print('predict failed, model not exist')
            return

        for ticker in self.paras.predict_tickers:
            try:
                data = data_feature[ticker]
            except:
                # print('stock not preparee', ticker)
                continue

            X_train, y_train   = preprocessing_data(self.paras, data[0], LabelColumnName, one_hot_label_proc=True)
            X_valid, y_valid   = preprocessing_data(self.paras, data[1], LabelColumnName, one_hot_label_proc=True)
            X_lately, y_lately = preprocessing_data(self.paras, data[2], LabelColumnName, one_hot_label_proc=False)
            
            X_train, y_train   = reshape_input(self.paras.n_features, X_train, y_train)                                                                    
            X_valid, y_valid   = reshape_input(self.paras.n_features, X_valid, y_valid)
            X_lately, y_lately = reshape_input(self.paras.n_features, X_lately, y_lately)

            possibility_columns = [str(window) + '_' + str(idx) for idx in range(self.paras.n_out_class)]

            # print('\n ---------- ', ticker, ' ---------- \n')
            # print(' ############## validation on train data ############## ')
            mse_known_train, predictions_train = self.predict(model, X_train, y_train)
            data[3].loc[data[0].index, 'label'] = np.argmax(y_train, axis=1) #- int(self.paras.n_out_class/2)
            data[3].loc[data[0].index, 'pred'] = np.argmax(predictions_train, axis=1) #- int(self.paras.n_out_class/2)
            s = pd.DataFrame(predictions_train, index = data[0].index, columns=possibility_columns)

            # print(' ############## validation on valid data ############## ')
            mse_known_lately, predictions_valid = self.predict(model, X_valid, y_valid)
            data[3].loc[data[1].index, 'label'] = np.argmax(y_valid, axis=1) #- int(self.paras.n_out_class/2)
            data[3].loc[data[1].index, 'pred'] = np.argmax(predictions_valid, axis=1) #- int(self.paras.n_out_class/2)
            s = s.append(pd.DataFrame(predictions_valid, index = data[1].index, columns=possibility_columns))

            # print(' ############## validation on lately data ############## ')
            mse_lately, predictions_lately = self.predict(model, X_lately, y_lately)
            data[3].loc[data[2].index, 'label'] = np.nan#np.argmax(actual_lately, axis=1)
            data[3].loc[data[2].index, 'pred'] = np.argmax(predictions_lately, axis=1) #- int(self.paras.n_out_class/2)
            s = s.append(pd.DataFrame(predictions_lately, index = data[2].index, columns=possibility_columns))
            
            data[3] = pd.merge(data[3], s, how='outer', left_index=True, right_index=True)

            if data[3]['pred'][-1] == 3:
                continue

            actual_count = []
            predict_count = []
            for i in range(self.paras.n_out_class):
                actual_count.append(len(data[3][data[3]['label'] == i]))
                predict_count.append(len(data[3][(data[3]['label'] == i) & (data[3]['label'] == data[3]['pred'])]))

            valid_actual_count = []
            valid_predict_count = []
            data.append(data[3][-self.paras.valid_len:])
            for i in range(self.paras.n_out_class):
                valid_actual_count.append(len(data[4][data[4]['label'] == i]))
                valid_predict_count.append(len(data[4][(data[4]['label'] == i) & (data[4]['label'] == data[4]['pred'])]))

            # print('classification counter:\n', actual_count)
            # print('classification possibility:\n', 100*np.array(actual_count)/np.sum(actual_count))
            # print('classification train predict:\n', 100*np.array(predict_count)/np.array(actual_count))
            # print('classification valid predict:\n', 100*np.array(valid_predict_count)/np.array(valid_actual_count))
            
            # timePeriod = [22*24, 22*12, 22*6, 22*3, 22*2, 22]
            # pred_profit = data[3]["pred_profit"]
            # pred_profit_len = len(pred_profit)
            # centers_oris = []
            # index_oris = []
            # for time in timePeriod:
            #     if pred_profit_len < time: continue
            #     out_labels, counters, centers_ori = kmeans_claasification(pred_profit[pred_profit_len - time : pred_profit_len], self.paras.n_out_class)
            #     centers_oris.append(centers_ori)
            #     index_oris.append("Days: " + str(time))
            
            # df_ori = pd.DataFrame(centers_oris, index=index_oris, columns=[str(idx) for idx in range(self.paras.n_out_class)])
            # print('\nclassification centers:\n', df_ori)

            data[3]['label'] = data[3]['label'] - int(self.paras.n_out_class/2)
            data[3]['pred'] = data[3]['pred'] - int(self.paras.n_out_class/2)
            
            # rewrite data frame and save / update
            data[3] = self.save_data_frame_mse(ticker, data[3], window, possibility_columns, mses=[mse_known_train, mse_known_lately])
            self.df = data[3]

            pd.set_option('display.max_rows', None)
            print('\n ---------- ', ticker, ' ---------- \n')
            print(data[3][-(self.paras.pred_len + self.paras.valid_len):])


    ###################################
    ###                             ###
    ###       Save Data Output      ###
    ###                             ###
    ###################################
    def save_data_frame_mse(self, ticker, df, window, possibility_columns, mses):
        df['label'] = df['label']#.astype(int)
        df['pred'] = df['pred']#.astype(int)
        
        # df = df.rename(columns={'label': 'a_+' + str(self.paras.pred_len) + '_d',
        #                         'pred': 'p_+' + str(self.paras.pred_len) + '_d'})
        # new_list = ['a_+' + str(self.paras.pred_len) + '_d', 'p_+' + str(self.paras.pred_len) + '_d']

        #default_list = ['open', 'high', 'low', 'close', 'volume']
        #original_other_list = set(df.columns) - set(default_list) - set(new_list)
        #original_other_list = list(original_other_list)
        default_list = ['close', 'volume', 'pred_profit']
        original_other_list = []
        new_list = ['label', 'pred']
        df = df[default_list + original_other_list + new_list + possibility_columns]
        
        model_acc = mses[1] / mses[0]
        if self.paras.save == True:
            #df.to_csv(self.paras.save_folder + ticker + ('_%.2f' % model_acc) + '_data_frame.csv')
            df.to_csv(self.paras.save_folder + ticker + '_' + str(window) + ('_%.2f' % model_acc) + '.csv')
            with open(self.paras.save_folder + 'parameters.txt', 'w') as text_file:
                text_file.write(self.paras.__str__())
                text_file.write(str(mses[0]) + '\n')
                text_file.write(str(mses[1]) + '\n')
        return df


    ###################################
    ###                             ###
    ###        Main Enterance       ###
    ###                             ###
    ###################################

    def run(self, train, predict):
        if self.check_parameters() == False:
            raise IndexError('Parameters for LSTM is wrong, check out_class_type')

        ################################################################################
        self.paras.save_folder = self.get_save_directory()
        print(' Log  Directory: ', self.paras.save_folder)
        self.paras.model_folder = self.get_model_directory()
        print('Model Directory: ', self.paras.model_folder)
        ################################################################################

        for window in self.paras.window_len:
            self.do_run(train, predict, window)

    def do_run(self, train, predict, window):
        LabelColumnName = 'label'
        data_file = "data_file_lstm_" + str(window) + ".pkl"

        if os.path.exists(data_file):
            input = open(data_file, 'rb')
            data_feature = pickle.load(input)
        else:
            data_feature = get_all_stocks_feature_data(self.paras, window, LabelColumnName)
            #output = open(data_file, 'wb')
            #pickle.dump(data_feature, output)

        model = None
            
        if train: model = self.train_data(data_feature, window, LabelColumnName)
            
        if predict: self.predict_data(model, data_feature, window, LabelColumnName)

