import os, csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.classification import accuracy_score
from sklearn.externals import joblib
from Stock_Prediction_Base import base_model
from Stock_Prediction_Data_Processing import get_all_stocks_feature_data, preprocessing_data, kmeans_claasification


from sklearn import preprocessing

# import matplotlib.pyplot as plt
# from matplotlib import style


class random_forrest_model(base_model):
    def build_model(self, X_train, y_train, index):
        if self.paras.load == True:
            model = self.load_training_model(self.paras.window_len[index])
            if model != None:
                return model

        print('build Random Forrest model...')
        #Define the prediction model
        model = RandomForestRegressor()
        return model

    def save_training_model(self, model, window_len):
        if self.paras.save == True:
            print('save Random Forrest model...')
            filename = self.paras.model_folder + self.get_model_name(window_len) + '.pkl'
            joblib.dump(model, filename) 

    def load_training_model(self, window_len):
        filename = self.paras.model_folder + self.get_model_name(window_len) + '.pkl'
        if os.path.exists(filename):
            print('load Random Forrest model...')
            return joblib.load(filename) 
        return None

# Classification
class random_forrest_regression(random_forrest_model):
    def __init__(self, paras):
        super(random_forrest_regression, self).__init__(paras=paras)

    ###################################
    ###                             ###
    ###          Training           ###
    ###                             ###
    ###################################

    def prepare_train_test_data(self, data_feature, LabelColumnName):
        firstloop = 1
        for ticker, data in data_feature.items():
            df = data[0][['hl_perc', 'co_perc']]
            X = np.array(df)
            X = preprocessing.scale(X)
            y = np.array(data[0]["price_next_month"])

            #X, y = preprocessing_data(self.paras, data[0], LabelColumnName, one_hot_label_proc=False)
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y, test_size=0.3)
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

        #print('Train shape X:', X_train.shape, ',y:', y_train.shape)
        #print('Test shape X:', X_test.shape, ',y:', y_test.shape)
        return X_train, y_train, X_test, y_test


    def train_data(self, data_feature, LabelColumnName, index):
        #history = History()

        X_train, y_train, X_test, y_test = self.prepare_train_test_data(data_feature, LabelColumnName)

        model = self.build_model(X_train, y_train, index)

        model.fit(X_train, y_train)

        # save model
        self.save_training_model(model, self.paras.window_len[index])

        print(' ############## validation on test data ############## ')
        self.predict(model, X_test, y_test)

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
        conf = model.score(X, y)
        print('Confidence: ', conf)

        predictions = model.predict(X)
        print(predictions)
        # if np.isfinite(y).all():
        #     print('Accuracy: ', accuracy_score(y, np.argmax(predictions, axis=1)))
        return predictions


    def predict_data(self, model, data_feature, LabelColumnName, index):

        if model == None: model = self.load_training_model(self.paras.window_len[index])

        if model == None:
            print('predict failed, model not exist')
            return

        for ticker in self.paras.predict_tickers:
            try:
                data = data_feature[ticker]
            except:
                print('stock not preparee', ticker)
                continue

            X_train, y_train   = preprocessing_data(self.paras, data[0], LabelColumnName, one_hot_label_proc=False)
            X_valid, y_valid   = preprocessing_data(self.paras, data[1], LabelColumnName, one_hot_label_proc=False)
            X_lately, y_lately = preprocessing_data(self.paras, data[2], LabelColumnName, one_hot_label_proc=False)

            possibility_columns = [str(self.paras.window_len[index]) + '_' + str(idx) for idx in range(self.paras.n_out_class)]

            print('\n ---------- ', ticker, ' ---------- \n')
            print(' ############## validation on train data ############## ')
            predictions_train = self.predict(model, X_train, y_train)
            data[3].loc[data[0].index, 'label'] = y_train#np.argmax(y, axis=1) #- int(self.paras.n_out_class/2)
            data[3].loc[data[0].index, 'pred'] = np.argmax(predictions_train, axis=1) #- int(self.paras.n_out_class/2)
            s = pd.DataFrame(predictions_train, index = data[0].index, columns=possibility_columns)

            print(' ############## validation on valid data ############## ')
            predictions_valid = self.predict(model, X_valid, y_valid)
            data[3].loc[data[1].index, 'label'] = y_valid#np.argmax(y_valid, axis=1) #- int(self.paras.n_out_class/2)
            data[3].loc[data[1].index, 'pred'] = np.argmax(predictions_valid, axis=1) #- int(self.paras.n_out_class/2)
            s = s.append(pd.DataFrame(predictions_valid, index = data[1].index, columns=possibility_columns))

            print(' ############## validation on lately data ############## ')
            predictions_lately = self.predict(model, X_lately, y_lately)
            data[3].loc[data[2].index, 'label'] = np.nan#np.argmax(actual_lately, axis=1)
            data[3].loc[data[2].index, 'pred'] = np.argmax(predictions_lately, axis=1) #- int(self.paras.n_out_class/2)
            s = s.append(pd.DataFrame(predictions_lately, index = data[2].index, columns=possibility_columns))
            
            data[3] = pd.merge(data[3], s, how='outer', left_index=True, right_index=True)

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

            print('\nclassification counter:\n', actual_count)
            print('\nclassification possibility:\n', 100*np.array(actual_count)/np.sum(actual_count))
            print('\nclassification train predict:\n', 100*np.array(predict_count)/np.array(actual_count))
            print('\nclassification valid predict:\n', 100*np.array(valid_predict_count)/np.array(valid_actual_count))

            timePeriod = [22*24, 22*12, 22*6, 22*3, 22*2, 22]
            pred_profit = data[3]["pred_profit"]
            pred_profit_len = len(pred_profit)
            centers_oris = []
            index_oris = []
            for time in timePeriod:
                if pred_profit_len < time: continue
                out_labels, counters, centers_ori = kmeans_claasification(pred_profit[pred_profit_len - time : pred_profit_len], self.paras.n_out_class)
                centers_oris.append(np.sort(centers_ori))
                index_oris.append(time)
            
            df_ori = pd.DataFrame(centers_oris, index=index_oris)
            df_ori.index.name = 'Days'
            print('\nclassification centers:\n', df_ori)
            
            # rewrite data frame and save / update
            data[3] = self.save_data_frame_mse(ticker, data[3], self.paras.window_len[index], possibility_columns)
            self.df = data[3]

            pd.set_option('display.max_rows', None)
            print('\n -------------------- \n')
            data[3]['label'] = data[3]['label'] - int(self.paras.n_out_class/2)
            data[3]['pred'] = data[3]['pred'] - int(self.paras.n_out_class/2)
            print(data[3][-(self.paras.pred_len + self.paras.valid_len):])
            


    ###################################
    ###                             ###
    ###       Save Data Output      ###
    ###                             ###
    ###################################

    def save_data_frame_mse(self, ticker, df, window_len, possibility_columns):
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
        
        if self.paras.save == True:
            #df.to_csv(self.paras.save_folder + ticker + ('_%.2f' % model_acc) + '_data_frame.csv')
            df.to_csv(self.paras.save_folder + ticker + '_' + str(window_len) + '.csv')
            with open(self.paras.save_folder + 'parameters.txt', 'w') as text_file:
                text_file.write(self.paras.__str__())
                #text_file.write(str(mses[0]) + '\n')
                #text_file.write(str(mses[1]) + '\n')
        return df


    ###################################
    ###                             ###
    ###        Main Enterance       ###
    ###                             ###
    ###################################

    def run(self, train, predict):
        ################################################################################
        self.paras.save_folder = self.get_save_directory()
        print('Save Directory: ', self.paras.save_folder)
        self.paras.model_folder = self.get_model_directory()
        print('Model Directory: ', self.paras.model_folder)
        ################################################################################

        LabelColumnName = 'label'

        for index in range(len(self.paras.window_len)):
            
            data_feature = get_all_stocks_feature_data(self.paras, self.paras.window_len[index], LabelColumnName)

            model = None
            
            if train: model = self.train_data(data_feature, LabelColumnName, index)
            
            if predict: self.predict_data(model, data_feature, LabelColumnName, index)







# style.use("fivethirtyeight")
# pd.set_option('precision', 3)
# pd.set_option('display.width',1000)

#Read the csv file into a DataFrame
# df = pd.read_csv("Tesla_stocks.csv")

#Make two new columns which will be used for making predictions.
# df["HL_Perc"] = (df["High"]-df["Low"]) / df["Low"] * 100
# df["CO_Perc"] = (df["Close"] - df["Open"]) / df["Open"] * 100

#Make array of dates
#Last 30 dates will be used for forecasting.
# dates = np.array(df["Date"])
# dates_check = dates[-30:]
# dates = dates[:-30]

# df = df[["HL_Perc", "CO_Perc", "Adj Close", "Volume"]]

#Define the label column


#Make fetaure and label arrays


#Divide the data set into training data and testing data
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)

# #Define the prediction model
# model = RandomForestRegressor()

# #Fit the model using training data
# model.fit(X_train, y_train)

# #Calculate the confidence value by applying the model to testing data
# conf = model.score(X_test, y_test)
# print(conf)

# #Fit the model again using the whole data set
# model.fit(X,y)

# #Make predictions
# predictions = model.predict(X_Check)

#Make the final DataFrame containing Dates, ClosePrices, and Forecast values
# actual = pd.DataFrame(dates, columns = ["Date"])
# actual["ClosePrice"] = df["Adj Close"]
# actual["Forecast"] = np.nan
# actual.set_index("Date", inplace = True)
# forecast = pd.DataFrame(dates_check, columns=["Date"])
# forecast["Forecast"] = predictions
# forecast.set_index("Date", inplace = True)
# var = [actual, forecast]
# result = pd.concat(var)  #This is the final DataFrame


# #Plot the results
# result.plot(figsize=(20,10), linewidth=1.5)
# plt.legend(loc=2, prop={'size':20})
# plt.xlabel('Date')
# plt.ylabel('Price')