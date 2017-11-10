import pandas as pd
from pandas.tseries.offsets import CustomBusinessMonthBegin
from pandas.tseries.holiday import USFederalHolidayCalendar


def convert_week_based_data(df):
    weekly_data = df.resample('W').agg({
                         'open': 'first', 
                         'high': 'max',
                         'low': 'min',
                         'close': 'last',
                         'volume': 'sum'})
    return weekly_data.dropna(inplace = True)

def convert_month_based_data(df):
    month_index =df.index.to_period('M')
    min_day_in_month_index = pd.to_datetime(df.set_index(month_index, append=True).reset_index(level=0).groupby(level=0)['open'].min())
    custom_month_starts = CustomBusinessMonthBegin(calendar = USFederalHolidayCalendar())
    ohlc_dict = {'open':'first','high':'max','low':'min','close': 'last','volume': 'sum'}
    mthly_data = df.resample(custom_month_starts).agg(ohlc_dict)
    return mthly_data.dropna(inplace = True)

