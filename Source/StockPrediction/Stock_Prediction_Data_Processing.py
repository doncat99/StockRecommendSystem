import sys, os, datetime
import numpy as np
import pandas as pd
from stockstats import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore
from sklearn.cluster import KMeans
import pickle

cur_path = os.path.dirname(os.path.abspath(__file__))
for _ in range(2):
    root_path = cur_path[0:cur_path.rfind('/', 0, len(cur_path))]
    cur_path = root_path
sys.path.append(root_path + "/" + 'Source/DataBase/')
from DB_API import queryStock


###################################
###                             ###
###        Data Utility         ###
###                             ###
###################################

def reshape_input(n_features, X, y):
    '''
    X.shape = [n_sample, window_len*n_features]
    X_reshaped = [n_sample, window_len, n_features]
    '''
    n_sample = X.shape[0]
    n_channel = n_features
    n_features_per_channel = int(X.shape[1] / n_channel)
    X_reshaped = np.reshape(X, (n_sample, n_features_per_channel, n_channel))
    y_reshaped = np.reshape(y, (n_sample, -1))
    return X_reshaped, y_reshaped

# def quantalize(df, columns):
#     normalize(df, columns)
#     scale = list(range(1, 256))
#     bins =[x / 256 for x in scale]
#     for feature_name in columns:
#         df[feature_name] = pd.np.digitize(df[feature_name], bins)

def normalization_scaler(norm, data, row_processing):
    '''
    data: N*M np.array
    N: sample
    M: features
    data_T: M*N
    data_T_scale: scaler for column by column, M*N
    data_T_scale_T: N*M
    '''
    if data.size == 0:
        return data 

    if '1' in norm:
        if row_processing:
            data_T = data.transpose()
            scaler = MinMaxScaler().fit_transform(StandardScaler().fit_transform(np.log(data+1)))
            #data_T_scale = scaler.transform(data_T)
            return scaler.transpose()
        else:
            #scaler = StandardScaler().fit(data)
            scaler =MinMaxScaler().fit_transform(StandardScaler().fit_transform(np.log(data+1)))
            return scaler

    elif '2' in norm:
        if row_processing:
            data_T = data.transpose()
            scaler = MinMaxScaler().fit(data_T)
            data_T_scale = scaler.transform(data_T)
            return data_T_scale.transpose()
        else:
            scaler = MinMaxScaler().fit(data)
            return scaler.transform(data)

    elif '3' in norm:
        if row_processing:
            data_T = data.transpose()
            data_T.apply(zscore)
            return data_T.transpose()
        else:
            data.apply(zscore)
            return data

    else: 
        return data

def one_hot_processing(data, class_count):
    data = data.astype(int)
    one_hot = np.zeros((len(data), class_count))
    one_hot[np.arange(len(data)), data] = 1
    return one_hot

def simple_means(value):
    if   value < -10:    return 0
    elif value < -5:     return 1
    elif value < -1.75:  return 2
    elif value < 1.75:   return 3
    elif value < 5:      return 4
    elif value < 10:     return 5
    else:                return 6

def simple_claasification(df, n_cluster=5):
    len_total = len(df)
    df.dropna(inplace=True)
    labels = [simple_means(df[i]) for i in range(len(df))]
    
    # check how many for each class
    counters = np.repeat(0, n_cluster)
    for i in labels:
        counters[i] += 1

    out_labels = np.append(labels, np.repeat(np.nan, len_total - len(df)))
    return out_labels, counters, None


def kmeans_claasification(df, n_cluster=5):
    '''
    Use KMeans algorithm to get the classification output
    '''
    len_total = len(df)
    df.dropna(inplace=True)
    X = np.array(df)
    X = X.reshape(-1, 1)
        
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(X)

    # resort KMeans label
    centers_ori = np.reshape(kmeans.cluster_centers_,(1, -1))  # [[ 0.16464226  2.03577568 -0.55692057  0.89430484 -1.52722935]]
        
    centers_ori_argsort = np.argsort(centers_ori, axis=1)  # [[4 2 0 3 1]]
    centers_new = np.argsort(centers_ori_argsort, axis=1)  # [[2 4 1 3 0]]
    centers_new = np.reshape(centers_new, (-1, 1))
    labels = kmeans.labels_
        
    # replace label value form centers_ori to centers_new
    labels = [centers_new[labels[i]] for i in range(len(labels))]

    # check how many for each class
    counters = np.repeat(0, n_cluster)
    for i in labels:
        counters[i] += 1

    out_labels = np.append(labels, np.repeat(np.nan, len_total - len(df)))
    # print('\n ------------------- \n')
    # print('classification counter:\n', counters)
    # print('\nclassification centers:\n', np.round(np.sort(centers_ori, axis=1), decimals=3))
    return out_labels, counters, centers_ori[0]


###################################
###                             ###
###       Read Stock Data       ###
###                             ###
###################################
def get_single_stock_data(root_path, symbol):
    '''
    All data is from quandl wiki dataset
    Feature set: [Open  High    Low  Close    Volume  Ex-Dividend  Split Ratio Adj. Open  Adj. High  Adj. Low
    Adj. Close  Adj. Volume]
    '''
    # file_name = stock_folder + ticker + '.csv'
    # COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    # RENAME_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume']

    COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume']

    # if os.path.exists(file_name) == False: 
    #     print("get stock: " + ticker + " failed")
    #     return pd.DataFrame()

    file_name = root_path + "/Data/CSV/symbols/" + symbol + ".csv"

    if os.path.exists(file_name) == False: return pd.DataFrame()
    
    df = pd.read_csv(
        file_name,
        #names=COLUMNS,
        skipinitialspace=True,
        engine='python',
        index_col=['date'],
        usecols=COLUMNS,
        parse_dates=['date'],
        #skiprows=1,
        #memory_map=True,
        #chunksize=300,
    ).sort_index()
    
    #df, lastUpdateTime = queryStock(root_path, "DB_STOCK", "SHEET_US", "_DAILY", symbol, "daily_update")
    #df.index = pd.to_datetime(df.index)

    if df.empty: 
        print("empty df", symbol)
        return df

    # if 'Adj Close' in df:
    #     close = 'Adj Close'
    # else:
    #     close = 'Close'
    #df=df.rename(columns = {'Date':'date', 'Open':'open', 'High':'high', 'Low':'low', close:'close', 'Volume':'volume'})

    return df


def get_all_stocks_data(root_path, train_tickers):
    data_original = {}
    # get data
    for ticker in train_tickers:
        data = get_single_stock_data(root_path, ticker) #[df, df_valid, df_lately, df_all, counters, centers_ori]
        if data.empty: continue
        data_original[ticker] = data
    return data_original


###################################
###                             ###
###       Prepareing Data       ###
###                             ###
###################################

def group_by_features(features, df):
    '''
    df.columns = [..., o_-10_d,h_-10_d,l_-10_d,c_-10_d,v_-10_d,...]
    return [...,o_-10_d,h_-10_d,l_-10_d,c_-10_d,...], [...,v_-10_d,...]
    '''
    data_group_features = {}
    data_group_columns = []

    for key, group in features.items():
        df_feature_col = []
        for i in group:
            df_feature_col.extend([col for col in df.columns if i in col])
        for feature in df_feature_col:
            data_group_columns.append(feature)
        data_group_features[key] = df[df_feature_col]

    # print("group by features")
    # print(data_group_features)
    # print("-"*20)
    # print(data_group_columns)
    return data_group_features, data_group_columns


###################################
###                             ###
###       Preprocess Data       ###
###                             ###
###################################
def preprocessing_train_data(paras, df, LabelColumnName, ticker, train_tickers_dict, one_hot_label_proc, array_format=True):
    day_list=train_tickers_dict[ticker]
    index_df=np.vectorize(lambda s: s.strftime('%Y-%m-%d'))(df.index.to_pydatetime())
    df.index=index_df
    common_day=list(set(day_list).intersection(set(index_df)))
    df=df.loc[common_day]
    X = df.drop(LabelColumnName, 1)
    y = np.array(df[LabelColumnName])
    #print(X.head())

    # print("ticker", ticker)
    # print(X)

    if one_hot_label_proc == True:
        # generate one hot output
        y_normalized_T = one_hot_processing(y, paras.n_out_class)
    else:
        y_normalized_T = y.astype(int)  # np.repeat(float('nan'), len(y))

    if array_format: return X.values, y_normalized_T

    return X, y_normalized_T
	
def preprocessing_data(paras, df, LabelColumnName, one_hot_label_proc, array_format=True):
    '''
    df: pd.DataFrame
    X: np.array
    y: np.array
    convert df into X,y
    '''
    X = df.drop(LabelColumnName, 1)
    y = np.array(df[LabelColumnName])

    # data_group_features, data_group_columns = group_by_features(paras.features, X)

    # X_normalized_T = pd.DataFrame(index=X.index, columns=data_group_columns)

    # for key_norm, df_feature in data_group_features.items():
    #     df_feature_norm = normalization_scaler(key_norm, df_feature, False)
    #     X_normalized_T.loc[df_feature.index, df_feature.columns] = df_feature_norm

    if one_hot_label_proc == True:
        # generate one hot output
        y_normalized_T = one_hot_processing(y, paras.n_out_class)
    else:
        y_normalized_T = y.astype(int) #np.repeat(float('nan'), len(y))

    # print("X_normalized_T", X_normalized_T.columns)
    # print(y_normalized_T)

    # if array_format: return X_normalized_T.values, y_normalized_T 

    # return X_normalized_T, y_normalized_T
    if array_format: return X.values, y_normalized_T

    return X, y_normalized_T


###################################
###                             ###
### Pack Data into window size  ###
###                             ###
###################################

def generate_time_series_data(paras, df, window_len):
    df_origin = df[['close', 'volume', 'pred_profit']]

    if window_len > 0:
        # Generate input features for time series data
        featureset = ['label']
        #featuresDict = {'c': 'close', 'h': 'high', 'l': 'low', 'o': 'open', 'v': 'volume', 't': 'top_pct', 'm': 'middle_pct', 'b': 'bottom_pct', 'p': 'close_pct', 'u': 'volume_pct'}
        
        for key, group in paras.features.items():
            sub_key = key.split('_')
            if sub_key[0] == '0': 
                featureset += group
                continue
            for i in range(window_len-1, -1, -1):
                for j in group:
                    df[j + '_-' + str(i) + '_d'] = df[j].shift(1 * i)
                    featureset.append(j + '_-' + str(i) + '_d')
        df = df[featureset]

    df_lately = df[-paras.pred_len:]
    df_valid = df[-paras.valid_len-paras.pred_len : -paras.pred_len]
    df_train = df[0:len(df) - paras.valid_len - paras.pred_len].dropna()

    # print('df_train len:', (df_train))
    # print('df_valid len:', (df_valid))
    # print('df_lately len:', (df_lately))
    # print('df_origin len:', len(df_origin))
    return [df_train, df_valid, df_lately, df_origin]


###################################
###                             ###
###      Get Feature Data       ###
###                             ###
###################################
def get_all_target_dict():
    # symbool_path = os.path.join(cur_path, 'Data')
    # symbool_path = os.path.join(symbool_path, 'CSV')
    # symbool_path = os.path.join(symbool_path, 'target')
    symbol_path = root_path + "/Data/CSV/target/"
    date_files = os.listdir(symbol_path)
    target_dict = {}
    for day_items in date_files:
        filename = os.path.join(symbol_path, day_items)
        target_df = pd.read_csv(filename, index_col='symbol')
        target_symbol_list = (target_df.index.astype(str).str.zfill(6)).tolist()
        for symbol in target_symbol_list:
            if symbol in target_dict:
                target_dict[symbol].append(day_items.split('.')[0])
            else:
                target_dict[symbol] = day_items.split('.')[:1]
    return target_dict

def get_single_stock_feature_data(ticker, paras, window_len, input_data, LabelColumnName):
    cashflow_file = root_path + "/Data/CSV/cashflow/" + ticker + ".csv"

    if os.path.exists(cashflow_file) == False: return pd.DataFrame()

    df = pd.read_csv(cashflow_file, index_col=["index"])
    df.set_index('date', inplace=True)

    input_data = input_data.add(df, fill_value=0)

    start_date = pd.Timestamp(paras.start_date)
    end_date   = pd.Timestamp(paras.end_date)
    input_data = input_data.loc[(input_data.index >= start_date) & (input_data.index <= end_date)]
    input_data = input_data[input_data['volume'] > 0]

    if len(input_data) < window_len + 3 * (paras.pred_len + paras.valid_len) : return pd.DataFrame()

    #print(ticker, input_data)#len(input_data))

    dataset = StockDataFrame.retype(input_data)
    
    # dataset.get('rsi_7')
    # dataset.get('rsi_14')
    # dataset.get('rsi_21')
    # dataset.get('kdjk_9')
    # dataset.get('kdjk_14')
    # dataset.get('wr_9')
    # dataset.get('wr_14')
    # dataset.get('close_-5_r')
    # dataset.get('close_-10_r')
    # dataset.get('close_-20_r')
    # dataset.get('close_-60_r')
    # dataset = dataset[np.isfinite(dataset['close_-20_r'])]
    
    #dataset['frac_change'] = (dataset['close'] - dataset['open']) / dataset['open']
    #dataset['frac_high']   = (dataset['high'] - dataset['open'])  / dataset['open']
    #dataset['frac_low']    = (dataset['open'] - dataset['low'])   / dataset['open']

    ret = lambda x,y: np.log(y/x) #Log return 
    # zscore = lambda x:(x -x.mean())/x.std() # zscore

    dataset['c_2_o'] = ret(dataset['open'], dataset['close'])
    dataset['h_2_o'] = ret(dataset['open'], dataset['high'])
    dataset['l_2_o'] = ret(dataset['open'], dataset['low'])
    dataset['c_2_h'] = ret(dataset['high'], dataset['close'])
    dataset['h_2_l'] = ret(dataset['high'], dataset['low'])
    dataset['vol_p'] = dataset['volume']

    dataset.loc[dataset.index, 'week_day'] = [pd.Timestamp(day).weekday() for day in dataset.index.values]
    #dataset['week_day'] = one_hot_processing(dataset['week_day'], 5)
    # print(dataset['week_day'])

    #dataset["hl_perc"] = (dataset["high"]-dataset["low"]) / dataset["low"] * 100
    #dataset["co_perc"] = (dataset["close"] - dataset["open"]) / dataset["open"] * 100
    #dataset["price_next_month"] = dataset["adj_close"].shift(-30)

    dataset['last_close']  = dataset['close'].shift(1 * (paras.pred_len))
    dataset['pred_profit'] = ((dataset['close'] - dataset['last_close']) / dataset['last_close'] * 100).shift(-1 * (paras.pred_len))
 
    df = dataset[['open', 'high', 'low', 'close', 'volume', 'pred_profit', 
                  'week_day',
                  #'frac_change', 'frac_high', 'frac_low', 
                  #'top',  'bottom', 'middle', 'vol_stat', 'pred_profit',
                  #'close_-5_r', 'close_-10_r', 'close_-20_r', 'close_-60_r'
                  #'open_pct', 'high_pct', 'low_pct', 'close_pct', 'volume_pct',
                  #'top_pct', 'middle_pct', 'bottom_pct', 'stock_stat'
                  #'rsi_7', 'rsi_14', 'rsi_21', 'kdjk_9', 'kdjk_14', 'wr_9', 
                  #'wr_14', 'close_-5_r', 'close_-10_r', 'close_-20_r',
                  'c_2_o', 'h_2_o', 'l_2_o', 'c_2_h', 'h_2_l', 'vol_p', 
                  'buy_amount', 'sell_amount', 'even_amount', 'buy_volume', 'sell_volume', 'even_volume', 'buy_max', 'buy_min', 'buy_average', 'sell_max', 'sell_min', 'sell_average', 'even_max', 'even_min', 'even_average',
                  #'hl_perc', 'co_perc'
                ]]    

    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

    data_group_features, data_group_columns = group_by_features(paras.features, df)

    for key, df_feature in data_group_features.items():
        sub_key = key.split('_')
        df_feature_norm = normalization_scaler(sub_key[1], df_feature, False)
        df[df_feature.columns.values] = df_feature_norm
          
    # Data frame output
    df[LabelColumnName], counter, center = simple_claasification(df['pred_profit'], paras.n_out_class)
    return df


def get_all_stocks_feature_data(paras, window_len, LabelColumnName):
    ori_file = "ori_file.pkl"
    if os.path.exists(ori_file):
        input = open(ori_file, 'rb')
        data_original = pickle.load(input)
        input.close()
    else:
        data_original = get_all_stocks_data(paras.root_path, paras.train_tickers)
        #output = open(ori_file, 'wb')
        #pickle.dump(data_original, output)
        #output.close()

    #data_original = get_all_stocks_data(paras.root_path, paras.train_tickers)
    data_feature = {}
    # get data
    for ticker, single_data in data_original.items():
        df_feature = get_single_stock_feature_data(ticker, paras, window_len, single_data, LabelColumnName)
        if df_feature.empty: continue
        df_feature = df_feature.fillna(0.0)
        data_feature[ticker] = generate_time_series_data(paras, df_feature, window_len)
    return data_feature


def get_train_stocks_feature_data(para, data_feature):
    date_files = os.listdir(root_path + "/Data/CSV/target/")

    train_feature = {}

    for date_file in date_files:
        date = date_file.split('.')[0]
        df = pd.read_csv(date_file)
        stocks = df['symbol'].values.tolist()

        
            

    # for ticker, data in data_feature.items():

###################################
###                             ###
### Get Label Possibility Data  ###
###                             ###
###################################

def get_single_stock_label_possibility_data(paras, ticker):
    first = True
    df_data = pd.DataFrame()
        
    #df_read_col = ['Date', 'label']
    df_possbibilty_col = [str(paras.window_len) + '_' + str(i) for i in range(paras.n_out_class)]
    #df_read_col.extend(df_possbibilty_col)

    df = get_single_stock_data(paras.root_path, ticker)

    if first:
        df_data = df.rename(columns={'label': 'label'})
        first = False
    else:
        for col in df_possbibilty_col: df_data.loc[df.index, col] = df[col]

    return df_data

def get_all_stocks_label_possibility_data(paras, DropColumnName):
    data_original = {}
    # get data
    for ticker in paras.train_tickers:
        data = get_single_stock_label_possibility_data(paras, ticker)
        if data.empty: continue

        data.dropna(inplace=True)
        data.loc[data.index, 'WeekDay'] = [str(pd.Timestamp(day).weekday()) for day in data.index.values]

        for dropColumn in DropColumnName:
            data = data.drop(dropColumn, 1)

        # first = True
        # for window in paras.window_len:
        #     df_possbibilty_col = [str(window) + '_' + str(i) for i in range(paras.n_out_class)]

        #     if first: 
        #         data_count = data[df_possbibilty_col].values
        #         first = False
        #     else: data_count = data_count + data[df_possbibilty_col].values

        # data_count = data_count / 3
        # new_df = data[['pred_profit', 'label']]
        # new_df.loc[new_df.index, 'pred'] = np.argmax(data_count, axis=1)
        # for col in range(paras.n_out_class): new_df.loc[new_df.index, str(col)] = data_count[:,col]

        # print(new_df[len(new_df) - paras.valid_len: len(new_df)])

        data_original[ticker] = generate_time_series_data(paras, data, 0)
    return data_original
 