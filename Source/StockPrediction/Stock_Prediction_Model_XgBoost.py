import os, csv
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from hyperopt import fmin, tpe, partial
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle

from Stock_Prediction_Base import base_model
from Stock_Prediction_Data_Processing import reshape_input, get_all_stocks_feature_data, preprocessing_data, kmeans_claasification, get_all_target_dict, preprocessing_train_data, kmeans_claasification
	
def S_score(y_true,y_pred):
    TP1=((y_true==y_pred)&(y_pred>3)).astype(int).sum()
    TP2=(y_true>y_pred).astype(int).sum()
    FP=((y_pred>y_true)&(y_true<4)).astype(int).sum()
    TP=TP1+TP2
    score=float(TP)/(TP+FP)
    return score

class xgboost_model(base_model):
    train_x = None
    train_y = None
    test_x  = None
    test_y  = None

    def GBM(self, argsDict):
        max_depth = argsDict["max_depth"] + 5
        n_estimators = argsDict['n_estimators'] * 5 + 50
        learning_rate = argsDict["learning_rate"] * 0.02 + 0.05
        gamma = argsDict["gamma"] * 0.1
        subsample = argsDict["subsample"] * 0.1 + 0.7
        min_child_weight = argsDict["min_child_weight"] + 1
        
        print("max_depth:" + str(max_depth), "n_estimator:" + str(n_estimators), "learning_rate:" + str(learning_rate), \
              "gamma:" + str(gamma), "subsample:" + str(subsample), "min_child_weight:" + str(min_child_weight))

        gbm = xgb.XGBClassifier(nthread=-1,    #进程数
                                max_depth=max_depth,  #最大深度
                                gamma=gamma,
                                n_estimators=n_estimators,   #树的数量
                                learning_rate=learning_rate, #学习率
                                subsample=subsample,      #采样数
                                min_child_weight=min_child_weight,   #孩子数
                                max_delta_step = 100,  #10步不降则停止
                                objective="multi:softmax")
        predicted=cross_val_predict(gbm, self.test_x, self.test_y,cv=5)
        scoring=recall_score(self.test_y, predicted, average='micro', labels=[4,5,6])
        #cro=cross_val_score(gbm, self.test_x, self.test_y, cv=5,scoring=scoring).mean()
        print('recall is ',scoring)
        return -scoring

    def best_model(self, X_train, y_train, X_test, y_test):
        self.train_x = X_train
        self.train_y = y_train
        self.test_x  = X_test
        self.test_y  = y_test

        algo = partial(tpe.suggest, n_startup_jobs=1)
        best = fmin(self.GBM, space=self.paras.hyper_opt, algo=algo, max_evals=100)
        print("best", best)
        return best
        
    def build_model(self, window, X_train, y_train, X_test, y_test):
        if self.paras.load == True:
            model = self.load_training_model()
            if model != None:
                return model

        best = {}
        #best {'gamma': 3, 'learning_rate': 5, 'max_depth': 1, 'min_child_weight': 1, 'n_estimators': 17, 'subsample': 0}
        file_name = "hyper_parameter_xgboost_" + str(window) + ".pkl"
        
        if self.paras.run_hyperopt == True:
            print('find hyper parameters...')
            best = self.best_model(X_train, y_train, X_test, y_test)
            pickle.dump(best, open(file_name, "wb"))
        else:
            if os.path.exists(file_name):
                best = pickle.load(open(file_name, "rb"))

        if len(best) == 0:
            max_depth = 10
            n_estimators = 100
            learning_rate = 0.09
            gamma = 0.2
            subsample = 0.9
            min_child_weight = 2
        else:
            max_depth = best["max_depth"] + 5
            n_estimators = best['n_estimators'] * 5 + 50
            learning_rate = best["learning_rate"] * 0.02 + 0.05
            gamma = best["gamma"] * 0.1
            subsample = best["subsample"] * 0.1 + 0.7
            min_child_weight = best["min_child_weight"] + 1

        print('Get XgBoost model parameter...')
        model = xgb.XGBClassifier(nthread=4,    #进程数
                                  max_depth=max_depth,  #最大深度
                                  gamma=gamma,
                                  n_estimators=n_estimators,   #树的数量
                                  learning_rate=learning_rate, #学习率
                                  subsample=subsample,      #采样数
                                  min_child_weight=min_child_weight,   #孩子数
                                  max_delta_step = 100,  #10步不降则停止
                                  objective="multi:softmax")
        return model

    def save_training_model(self, model, window_len):
        if self.paras.save == True:
            print('save XgBoost model...')
            # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
            #model.save(self.paras.model_folder + self.get_model_name(window_len) + '.h5')  # creates a HDF5 file 'my_model.h5'
            pickle.dump(model, open(self.paras.model_folder + self.get_model_name(window_len) + '.h5', "wb"))

    def load_training_model(self, window_len):
        # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
        model_file = self.paras.model_folder + self.get_model_name(window_len) + '.h5'
        if os.path.exists(model_file):
            print('load XgBoost model...')
            #return load_model(model_file)  # creates a HDF5 file 'my_model.h5'
            return pickle.load(open(model_file, "rb"))
        return None


# Classification
class xgboost_classification(xgboost_model):
    def __init__(self, paras):
        super(xgboost_classification, self).__init__(paras=paras)

    def check_parameters(self):
        if self.paras.out_class_type == 'classification' and self.paras.n_out_class > 1:
            return True
        return False


    ###################################
    ###                             ###
    ###          Training           ###
    ###                             ###
    ###################################

    def prepare_train_test_data(self, data_feature, LabelColumnName):
        firstloop = 1
        print("get_data_feature")
        #print(data_feature)
        for ticker, data in data_feature.items():
            #print(ticker, "n_feature", self.paras.n_features, len(data[0]))
            #print("data[0]",data[0].index)
            X, y = preprocessing_data(self.paras, data[0], LabelColumnName, one_hot_label_proc=False)
            #print(X.shape)
            #X, y = reshape_input(self.paras.n_features, X, y)
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

            X, y = preprocessing_train_data(self.paras, data[0].copy(), LabelColumnName, ticker, train_tickers_dict, one_hot_label_proc=False)
            # print(X.shape)
            # X, y = reshape_input(self.paras.n_features, X, y)
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
        print("Prepare Train data")
        X_train, y_train, X_test, y_test = self.prepare_train_data(data_feature, LabelColumnName)
        print("X_train",X_train.shape)

        model = self.build_model(window, X_train, y_train, X_test, y_test)
        print("build XgBoost model...")

        model.fit(
            X_train,
            y_train,
            verbose=self.paras.verbose
        )
        pred=model.predict(X_train)
        pred2 = model.predict(X_test)
        print("Filter train_data_recall is ",recall_score(y_train,pred,average='micro',labels=[4,5,6]))
        print("Filter test_data_recall is ", recall_score(y_test,pred2,average='micro',labels=[4,5,6]))

        # save model
        self.save_training_model(model, window)

        # print(' ############## validation on test data ############## ')
        #mse_test, tmp = self.predict(model, X_test, y_test)

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
        recall_p = S_score(y,predictions)
        return recall_p, predictions


    def predict_data(self, model, data_feature, window, LabelColumnName):

        if model == None: model = self.load_training_model(window)

        if model == None:
            print('predict failed, model not exist')
            return

        for ticker in self.paras.predict_tickers:
            try:
                data = data_feature[ticker]
            except:
                #print('stock not prepare:', ticker)
                continue

            # print(ticker, len(data[0]), len(data[1]), len(data[2]), len(data[3]))

            # print(data[3])

            # print("$"*40)

            # print(data[0])

            X_train, y_train   = preprocessing_data(self.paras, data[0], LabelColumnName, one_hot_label_proc=False)
            X_valid, y_valid   = preprocessing_data(self.paras, data[1], LabelColumnName, one_hot_label_proc=False)
            X_lately, y_lately = preprocessing_data(self.paras, data[2], LabelColumnName, one_hot_label_proc=False)
            
            # X_train, y_train   = reshape_input(self.paras.n_features, X_train, y_train)                                                                    
            # X_valid, y_valid   = reshape_input(self.paras.n_features, X_valid, y_valid)
            # X_lately, y_lately = reshape_input(self.paras.n_features, X_lately, y_lately)

            # possibility_columns = [str(self.paras.window_len) + '_' + str(idx) for idx in range(self.paras.n_out_class)]

            # print('\n ---------- ', ticker, ' ---------- \n')
            # print('############## validation on train data ##############')
            rec_known_train, predictions_train = self.predict(model, X_train, y_train)
            rec_known_valid, _ = self.predict(model, X_valid, y_valid)

            #print((data[3].index))
            # index_df = np.vectorize(lambda s: s.strftime('%Y-%m-%d'))(data[3].index.to_pydatetime())
            # print(data[3])
            # data[3].index=index_df
            data[3].loc[data[0].index, 'label'] = y_train #- int(self.paras.n_out_class/2)
            data[3].loc[data[0].index, 'pred'] = predictions_train #- int(self.paras.n_out_class/2)
            #s = pd.DataFrame(predictions_train, index = data[0].index, columns=possibility_columns)

            # print('############## validation on valid data ##############')
            mse_known_lately, predictions_valid = self.predict(model, X_valid, y_valid)
            # print('scaled data mse: ', mse_known_lately)
            # index_df = np.vectorize(lambda s: s.strftime('%Y-%m-%d'))(data[1].index.to_pydatetime())
            # data[1].index = index_df
            data[3].loc[data[1].index, 'label'] = y_valid #- int(self.paras.n_out_class/2)
            data[3].loc[data[1].index, 'pred'] = predictions_valid #- int(self.paras.n_out_class/2)
            #s = s.append(pd.DataFrame(predictions_valid, index = data[1].index, columns=possibility_columns))

            # print('############## validation on lately data ##############')
            mse_lately, predictions_lately = self.predict(model, X_lately, y_lately)
            # print('scaled data mse: ', mse_lately)
            # index_df = np.vectorize(lambda s: s.strftime('%Y-%m-%d'))(data[2].index.to_pydatetime())
            # data[2].index = index_df
            data[3].loc[data[2].index, 'label'] = np.nan#np.argmax(actual_lately, axis=1)
            data[3].loc[data[2].index, 'pred'] = predictions_lately #- int(self.paras.n_out_class/2)
            #s = s.append(pd.DataFrame(predictions_lately, index = data[2].index, columns=possibility_columns))
            
            #data[3] = pd.merge(data[3], s, how='outer', left_index=True, right_index=True)
            if predictions_lately[0] == 3:
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
            #data[3] = self.save_data_frame_mse(ticker, data[3], self.paras.window_len[index], possibility_columns, mses=[mse_known_train, mse_known_lately])
            self.df = data[3]

            pred_df = data[3]['pred'][-(self.paras.pred_len + self.paras.valid_len):]

            pd.set_option('display.max_rows', None)
            if (pred_df == 0).all() == False:
                print('\n ---------- ', ticker, ' ---------- \n')
                print('train data recall: ', rec_known_train)
                print('valid data recall: ', rec_known_valid)
                print(data[3][-(self.paras.pred_len + self.paras.valid_len):])


    ###################################
    ###                             ###
    ###       Save Data Output      ###
    ###                             ###
    ###################################

    def save_data_frame_mse(self, ticker, df, window_len, possibility_columns, mses,model):
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
            df.to_csv(self.paras.save_folder + ticker + '_' + str(window_len) + ('_%.2f' % model_acc) + '.csv')
            with open(self.paras.save_folder + 'parameters.txt', 'w') as text_file:
                text_file.write(model.get_xgb_params().__str__() + '\n')
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
            raise IndexError('Parameters for XgBoost is wrong, check out_class_type')

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
        data_file = "data_file_xgboost_" + str(window) + ".pkl"

        if os.path.exists(data_file):
            input = open(data_file, 'rb')
            data_feature = pickle.load(input)
            input.close()
        else:
            data_feature = get_all_stocks_feature_data(self.paras, window, LabelColumnName)
            output = open(data_file, 'wb')
            pickle.dump(data_feature, output)
            output.close()

        model = None

        train_feature = {}
            
        if train: model = self.train_data(data_feature, window, LabelColumnName)
            
        if predict: self.predict_data(model, data_feature, window, LabelColumnName)

