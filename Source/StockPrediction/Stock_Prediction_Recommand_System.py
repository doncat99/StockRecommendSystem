import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.classification import accuracy_score
from Stock_Prediction_Base import base_model
from Stock_Prediction_Data_Processing import get_all_stocks_label_possibility_data, preprocessing_data


class recommand_system_model(base_model):
    # build model
    def build_model(self, CategoricalColumnName, ContinuousColumnName, flags):
        self.load_training_model()

        print('build Recommand System model...')
        
        # Continuous base columns.
        deep_columns = []
        for column in ContinuousColumnName:
            deep_columns.append(tf.contrib.layers.real_valued_column(column))

        # 离散分类别的
        wide_columns = []
        for column in CategoricalColumnName:
            temp = tf.contrib.layers.sparse_column_with_hash_bucket(column, hash_bucket_size=100)
            wide_columns.append(temp)
            deep_columns.append(tf.contrib.layers.embedding_column(temp, dimension=8))

        # open         = tf.contrib.layers.real_valued_column("Open")
        # high         = tf.contrib.layers.real_valued_column("High")
        # low          = tf.contrib.layers.real_valued_column("Low")
        # close        = tf.contrib.layers.real_valued_column("Close")
        # volume       = tf.contrib.layers.real_valued_column("Volume")
        # top_line     = tf.contrib.layers.real_valued_column("Top_line")
        # middle_block = tf.contrib.layers.real_valued_column("Middle_block")
        # bottom_line  = tf.contrib.layers.real_valued_column("Bottom_line")
        # profit       = tf.contrib.layers.real_valued_column("Profit")

        #类别转换
        #age_buckets = tf.contrib.layers.bucketized_column(age, boundaries= [18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

        # wide_columns = [analyist, estimate, actual, surprise, earningDay, status,
        #                 tf.contrib.layers.crossed_column([estimate, actual], hash_bucket_size=int(1e4)),
        #                 tf.contrib.layers.crossed_column([analyist, surprise], hash_bucket_size=int(1e6)),
        #                 tf.contrib.layers.crossed_column([surprise, earningDay, status],hash_bucket_size=int(1e4))]

        #embedding_column用来表示类别型的变量
        # deep_columns = [tf.contrib.layers.embedding_column(analyist  , dimension=8),
        #                 tf.contrib.layers.embedding_column(estimate  , dimension=8),
        #                 tf.contrib.layers.embedding_column(actual    , dimension=8),
        #                 tf.contrib.layers.embedding_column(surprise  , dimension=8),
        #                 tf.contrib.layers.embedding_column(earningDay, dimension=8),
        #                 tf.contrib.layers.embedding_column(status    , dimension=8),
        #                 open, high, low, close, volume, top_line, middle_block, bottom_line, profit]

        if flags.model_type =="wide":
            model = tf.contrib.learn.LinearClassifier(model_dir=self.paras.model_folder,feature_columns=wide_columns)
        elif flags.model_type == "deep":
            model = tf.contrib.learn.DNNClassifier(model_dir=self.paras.model_folder, feature_columns=deep_columns, hidden_units=[100,50])
        else:
            model = tf.contrib.learn.DNNLinearCombinedClassifier(model_dir=self.paras.model_folder, linear_feature_columns=wide_columns, dnn_feature_columns=deep_columns, dnn_hidden_units=[100,50])
        
        self.save_training_model(model)
        
        return model 

    def save_training_model(self, model):
        # Do nothing, by setting model_dir will save the model automatically
        return

    def load_training_model(self):
        if self.paras.load == False:
            # Todo: clear the model file
            model_file = self.paras.model_folder
        #else:
            # Do nothing, by setting model_dir will load the model automatically


class recommand_system(recommand_system_model):
    def __init__(self, paras):
        super(recommand_system, self).__init__(paras=paras)

    def input_fn(self, df, y, CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS):
        continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
        # 原文例子为dense_shape
        categorical_cols = {k: tf.SparseTensor(indices=[[i,0] for i in range(df[k].size)], values = df[k].values, dense_shape=[df[k].size,1]) for k in CATEGORICAL_COLUMNS}
        feature_cols = dict(continuous_cols)
        feature_cols.update(categorical_cols)
        label = tf.constant(y)
        return feature_cols, label

    ###################################
    ###                             ###
    ###          Training           ###
    ###                             ###
    ###################################

    def prepare_train_test_data(self, data_feature, LabelColumnName):

        firstloop = 1
        for ticker, data in data_feature.items():
            X, y = preprocessing_data(self.paras, data[0], LabelColumnName, one_hot_label_proc=False, array_format=False)
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
                X_train.append(X_train_temp, ignore_index=True)
                X_test.append(X_test_temp, ignore_index=True)
                y_train = np.append(y_train, y_train_temp, 0)
                y_test = np.append(y_test, y_test_temp, 0)

        # print('Train shape X:', X_train.shape, ',y:', y_train.shape)
        # print('Test shape X:', X_test.shape, ',y:', y_test.shape)
        return X_train, y_train, X_test, y_test


    def train_data(self, data_feature, LabelColumnName, CategoricalColumnName, ContinuousColumnName, flags):
        model = self.build_model(CategoricalColumnName, ContinuousColumnName, flags)

        X_train, y_train, X_test, y_test = self.prepare_train_test_data(data_feature, LabelColumnName)

        model.fit(input_fn=lambda: self.input_fn(X_train, y_train, ContinuousColumnName, CategoricalColumnName), steps=flags.train_steps)

        print(' ############## validation on test data ############## ')
        self.predict(model, X_test, y_test, ContinuousColumnName, CategoricalColumnName)
        return model


    ###################################
    ###                             ###
    ###         Predicting          ###
    ###                             ###
    ###################################

    def predict(self, model, X, y, ContinuousColumnName, CategoricalColumnName):
        predictions = np.array(list(model.predict_proba(input_fn=lambda: self.input_fn(X, y, ContinuousColumnName, CategoricalColumnName))))
        
        results = model.evaluate(input_fn=lambda: self.input_fn(X, y, ContinuousColumnName, CategoricalColumnName), steps=1)

        for key in sorted(results):
            print("%s: %s"%(key, results[key]))
        print('Accuracy: ', accuracy_score(y, tf.argmax(predictions, axis=1)))
        return predictions


    def predict_data(self, model, data_feature, LabelColumnName):

        if model == None: model = self.build_model()

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

            possibility_columns = ['outClass_' + str(idx) for idx in range(self.paras.n_out_class)]

            print('\n ---------- ', ticker, ' ---------- \n')
            print(' ############## validation on train data ############## ')
            mse_known_train, predictions_train = self.predict(model, X_train, y_train)
            data[3].loc[data[0].index, 'label'] = np.argmax(y, axis=1) #- int(self.paras.n_out_class/2)
            data[3].loc[data[0].index, 'pred'] = np.argmax(predictions_train, axis=1) #- int(self.paras.n_out_class/2)
            s = pd.DataFrame(predictions_train, index = data[0].index, columns=possibility_columns)

            print(' ############## validation on valid data ############## ')
            mse_known_lately, predictions_valid = self.predict(model, X_valid, y_valid)
            data[3].loc[data[1].index, 'label'] = np.argmax(y_valid, axis=1) #- int(self.paras.n_out_class/2)
            data[3].loc[data[1].index, 'pred'] = np.argmax(predictions_valid, axis=1) #- int(self.paras.n_out_class/2)
            s = s.append(pd.DataFrame(predictions_valid, index = data[1].index, columns=possibility_columns))

            print(' ############## validation on lately data ############## ')
            mse_lately, predictions_lately = self.predict(model, X_lately, y_lately)
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

            print('classification counter:\n', actual_count)
            print('classification possibility:\n', 100*np.array(actual_count)/np.sum(actual_count))
            print('classification train predict:\n', 100*np.array(predict_count)/np.array(actual_count))
            print('classification valid predict:\n', 100*np.array(valid_predict_count)/np.array(valid_actual_count))
            #print('\nclassification centers:\n', np.round(np.sort(data[5], axis=1), decimals=3))

            data[3]['label'] = data[3]['label'] - int(self.paras.n_out_class/2)
            data[3]['pred'] = data[3]['pred'] - int(self.paras.n_out_class/2)
            
            # rewrite data frame and save / update
            data[3] = self.save_data_frame_mse(ticker, data[3], self.paras.window_len[index], possibility_columns, mses=[mse_known_train, mse_known_lately])
            self.df = data[3]

            pd.set_option('display.max_rows', None)
            print('\n -------------------- \n')
            print(data[3][-(self.paras.pred_len + self.paras.valid_len):])


    ###################################
    ###                             ###
    ###       Save Data Output      ###
    ###                             ###
    ###################################

    def save_data_frame_mse(self, ticker, df, window_len, possibility_columns, mses):
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
        flags = tf.app.flags
        FLAGS = flags.FLAGS
        flags.DEFINE_string("model_dir",    "",               "Base directory for output models.")
        flags.DEFINE_string("model_type",   "wide_n_deep",    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
        flags.DEFINE_integer("train_steps", self.paras.epoch, "Number of training steps.")
        flags.DEFINE_string("train_data",   "",               "Path to the training data.")
        flags.DEFINE_string("test_data",    "",               "path to the test data")
        
        ################################################################################
        self.paras.save_folder  =  self.get_save_directory()
        print('Save Directory: ',  self.paras.save_folder)
        self.paras.model_folder =  self.get_model_directory()
        print('Model Directory: ', self.paras.model_folder)
        ################################################################################
        
        DropColumnName        = []
        LabelColumnName       = 'label'
        CategoricalColumnName = ['WeekDay']
        ContinuousColumnName  = []
        for window in self.paras.window_len:
            for i in range(self.paras.n_out_class):
                ContinuousColumnName.append(str(window) + '_' + str(i))

        data_possibility = get_all_stocks_label_possibility_data(self.paras, DropColumnName)

        model = None
        
        # Fixme: "train" must be True by far, since save/load model functions are not yet accomplished
        if train: model = self.train_data(data_possibility, LabelColumnName, CategoricalColumnName, ContinuousColumnName, FLAGS)
        
        if predict: self.predict_data(model, data_possibility, LabelColumnName)
