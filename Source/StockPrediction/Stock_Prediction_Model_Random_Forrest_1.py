import pandas as pd
import numpy as np
from sklearn import cross_validation, preprocessing
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib import style

style.use("fivethirtyeight")
pd.set_option('precision', 3)
pd.set_option('display.width',1000)

#Read the csv file into a DataFrame
df = pd.read_csv("Tesla_stocks.csv")

#Make two new columns which will be used for making predictions.
df["HL_Perc"] = (df["High"]-df["Low"]) / df["Low"] * 100
df["CO_Perc"] = (df["Close"] - df["Open"]) / df["Open"] * 100

#Make array of dates
#Last 30 dates will be used for forecasting.
dates = np.array(df["Date"])
dates_check = dates[-30:]
dates = dates[:-30]

df = df[["HL_Perc", "CO_Perc", "Adj Close", "Volume"]]

#Define the label column
df["PriceNextMonth"] = df["Adj Close"].shift(-30)

#Make fetaure and label arrays
X = np.array(df.drop(["PriceNextMonth"], 1))
X = preprocessing.scale(X)
X_Check = X[-30:]
X = X[:-30]
df.dropna(inplace = True)
y = np.array(df["PriceNextMonth"])

#Divide the data set into training data and testing data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size = 0.3)

#Define the prediction model
model = RandomForestRegressor()

#Fit the model using training data
model.fit(X_train, y_train)

#Calculate the confidence value by applying the model to testing data
conf = model.score(X_test, y_test)
print(conf)

#Fit the model again using the whole data set
model.fit(X,y)

#Make predictions
predictions = model.predict(X_Check)

#Make the final DataFrame containing Dates, ClosePrices, and Forecast values
actual = pd.DataFrame(dates, columns = ["Date"])
actual["ClosePrice"] = df["Adj Close"]
actual["Forecast"] = np.nan
actual.set_index("Date", inplace = True)
forecast = pd.DataFrame(dates_check, columns=["Date"])
forecast["Forecast"] = predictions
forecast.set_index("Date", inplace = True)
var = [actual, forecast]
result = pd.concat(var)  #This is the final DataFrame


print(original_df)


#Plot the results
result.plot(figsize=(20,10), linewidth=1.5)
plt.legend(loc=2, prop={'size':20})
plt.xlabel('Date')
plt.ylabel('Price')