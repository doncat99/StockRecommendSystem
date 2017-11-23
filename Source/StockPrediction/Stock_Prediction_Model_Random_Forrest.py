import os, csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.classification import accuracy_score
from sklearn.externals import joblib
from Stock_Prediction_Base import base_model
from Stock_Prediction_Data_Processing import get_all_stocks_feature_data, preprocessing_data, kmeans_claasification

class random_forrest_model(base_model):
    # CROSS VALIDATION : Compute accuracy of a model ##############################
    ## Inputs : X_train, y_train, number of folds, number of trees, max of features
    ## Output : Accuracy of classifier
    def perform_CV(self, X_train, y_train, number_folds, n, m):
        model = RandomForestClassifier(n_estimators=n, max_features=m, n_jobs=8, verbose=self.paras.verbose)
        acc = np.mean(cross_val_score(model, X_train, y_train, cv=number_folds))
        #print 'Size of Forrest : number of trees : ' + str(n) + ', maximum of features : ' + str(m) + '. Accuracy : ' + str(acc)
        return acc

    # MODEL SELECTION : Find best parameters ######################################
    ## Inputs :  X_train, y_train, number of folds, range of number of trees, range of max of features
    ## Outputs : optimal number of trees, optimal max of features, accuracy  
    def best_forrest(self, X_train, y_train, number_folds, t1, t2, f1, f2):  
        # Initialize parameters        
        t_opt = t1
        f_opt = f1
        accur_opt = 0.
        #x_n = []; y_m= []; z_accu = []
        
        # Find best forest 
        for t in range(t1,t2+1):
            for f in range(f1,f2+1):
                t_ = 16 * t
                accur = self.perform_CV(X_train, y_train, number_folds, t_, f)
                if (accur > accur_opt) : t_opt, f_opt, accur_opt = t_, f, accur
                #x_n.append(n), y_m.append(m), z_accu.append(accur)
    
        #my_df = pd.DataFrame([x_n,y_m,z_accu])
        #my_df.to_csv('n_'+str(n)+'_m_'+str(m)+'_.csv', index=False, header=False)
        #fig = pylab.figure()
        #ax = Axes3D(fig)
        #ax.plot_trisurf(x_n,y_m,z_accu)
        #ax.set_xlabel('Number of Trees')
        #ax.set_ylabel('Number of Features')
        #ax.set_zlabel('Accuracy')    
        #plt.show()
    
        #print('Best Forrest : number of trees : ' + str(n_opt) + ', maximum of features : ' + str(m_opt) + ', with accuracy :' + str(accur_opt))
        return t_opt,f_opt,accur_opt


    # BEST WINDOW : Find best window ##############################################    
    def best_window(self, X_train, y_train, w_min, w_max, t_min,t_max,f_min,f_max):
        w_opt = 0
        t_opt = 0
        f_opt = 0
        accur_opt = 0.
        
        x_w = []
        y_accu= []
        
        # range of window : w_min --> w_max     
        for w in range(w_min,w_max+1):
            #X,y = preprocess_data(w)
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
            t, f, accur = self.best_forrest(X_train,y_train,10,t_min,t_max,f_min,f_max)
            print('Window = '+str(w)+' days --> Best Forrest : number of trees : ' + str(t) + ', maximum of features : ' + str(f) + ', with accuracy :' + str(accur))
            
            if (accur > accur_opt) : w_opt, t_opt, f_opt, accur_opt = w, t, f, accur
            x_w.append(w), y_accu.append(accur)
            
        print('Best window : w = '+str(w_opt)+'. Best Forrest : number of trees : ' + str(t_opt) + ', maximum of features : ' + str(f_opt) + ', with accuracy :' + str(accur_opt))
        return w_opt, t_opt, f_opt


    def build_model(self, X_train, y_train, index):
        if self.paras.load == True:
            model = self.load_training_model(self.paras.window_len[index])
            if model != None:
                return model

        print('build Random Forrest model...')

        # range of number of trees : 5*(1 -> 10) = 5,10,...,50 trees
        t_min = self.paras.tree_min[index]
        t_max = self.paras.tree_max[index]
        # range of max of features : 1 -> 10 features
        f_min = self.paras.feature_min[index]
        f_max = self.paras.feature_max[index]
        # range of window : 1 -> 70 days 
        w_min = self.paras.window_min
        w_max = self.paras.window_max
        
        w_opt, n_opt, m_opt = self.best_window(X_train, y_train, w_min,w_max,t_min,t_max,f_min,f_max)
        model = RandomForestClassifier(n_estimators=n_opt,max_features=m_opt, n_jobs=8, verbose=self.paras.verbose)
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
class random_forrest_classification(random_forrest_model):
    def __init__(self, paras):
        super(random_forrest_classification, self).__init__(paras=paras)


    ###################################
    ###                             ###
    ###          Training           ###
    ###                             ###
    ###################################

    def prepare_train_test_data(self, data_feature, LabelColumnName):
        firstloop = 1
        for ticker, data in data_feature.items():
            X, y = preprocessing_data(self.paras, data[0], LabelColumnName, one_hot_label_proc=False)
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
        predictions = model.predict_proba(X)
        if np.isfinite(y).all():
            print('Accuracy: ', accuracy_score(y, np.argmax(predictions, axis=1)))
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
        print(' Log  Directory: ', self.paras.save_folder)
        self.paras.model_folder = self.get_model_directory()
        print('Model Directory: ', self.paras.model_folder)
        ################################################################################

        LabelColumnName = 'label'

        for index in range(len(self.paras.window_len)):
            
            data_feature = get_all_stocks_feature_data(self.paras, self.paras.window_len[index], LabelColumnName)

            model = None
            
            if train: model = self.train_data(data_feature, LabelColumnName, index)
            
            if predict: self.predict_data(model, data_feature, LabelColumnName, index)

