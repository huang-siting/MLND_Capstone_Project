"""
Utility code

Part of the utility funtions are in the template from the following project (c) 2015 Tucker Balch
http://quantsoftware.gatech.edu/Summer_2016_Project_5
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy

def symbol_to_path(symbol, base_dir=os.path.join("..", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_all_data(symbols, colname = 'Adj Close'):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', colname], na_values=['nan'])
        df_temp = df_temp.rename(columns={colname: symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

def get_data(symbols, dates, addSPY=True, colname = 'Adj Close'):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', colname], na_values=['nan'])
        df_temp = df_temp.rename(columns={colname: symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def get_rolling_mean(values, window=15):
    """Return rolling mean of given values, using specified window size."""
    return pd.rolling_mean(values, window=window)


def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return pd.rolling_std(values, window=window)


def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands.""" 
    upper_band = rm + rstd*2
    lower_band = rm - rstd*2
    return upper_band, lower_band


def plot_bollinger_bands(df, window):
    """Plot stock prices with Bollinger bands."""

    # Compute rolling mean
    rm = get_rolling_mean(df, window=window)

    # Compute rolling standard deviation
    rstd = get_rolling_std(df, window=window)

    # Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm, rstd)
    
    # Plot raw SPY values, rolling mean and Bollinger Bands
    # TODO: fix the label on plot
    ax = df.plot(title="Bollinger Bands")
    rm.plot(label='Rolling mean', ax=ax)
    upper_band.plot(label='upper band', ax=ax)
    lower_band.plot(label='lower band', ax=ax)

    # Add axis labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()

def compute_percent_b(df, window=15):
    """Return percent b """ 

    # Compute rolling mean
    rm = get_rolling_mean(df, window=window)

    # Compute rolling standard deviation
    rstd = get_rolling_std(df, window=window)

    # Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm, rstd)

    bandwidth = upper_band - lower_band 
    diff = df - lower_band

    return diff/bandwidth 

def bollinger_band_indicator(price, upperband, lowerband):
    if price > upperband:
        return 2
    elif price < lowerband:
        return 0
    else:
        return 1

def discretize_bollinger_band_state(prices, window=15):
    """discretize stock price relative to bollingerband """ 
    # Compute rolling mean
    rm = get_rolling_mean(prices, window=window)

    # Compute rolling standard deviation
    rstd = get_rolling_std(prices, window=window)

    # Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm, rstd)

    upper_band.dropna(inplace=True)

    data = upper_band.index.map(lambda x: bollinger_band_indicator(prices.ix[x].values[0], upper_band.ix[x].values[0], lower_band.ix[x].values[0]))
        
    bollinger_band_state = pd.DataFrame(data, index = upper_band.index)
    return bollinger_band_state 


def compute_daily_returns(df):
    """Return daily returns.""" 
    daily_returns = (df/df.shift(1)) - 1
    daily_returns.ix[0,:] = 0 #set daily returns for row 0 to 0
    return daily_returns


def compute_daily_changes(df):
    """Return daily changes.""" 
    daily_changes = df-df.shift(1)
    daily_changes.ix[0,:] = 0 
    return daily_changes


def compute_cumulative_returns(df):
    """Return cumulative returns.""" 
    cumulative_returns = df/df.ix[0,:]-1
    return cumulative_returns


def compute_ratio_close_SMA(df, window=15):
    """Return ratio_close_SMA for df with only one column""" 
    rolling_mean = get_rolling_mean(df,window)
    ratio_close_SMA = df.div(rolling_mean)
    return ratio_close_SMA


def compute_discretizing_thresholds(data, steps=10):
    """Return threshold after discretizing""" 
    stepsize = len(data) / steps
    copy_list = copy.copy(data)
    copy_list.sort()
    
    thresholds = np.zeros(steps)
    for i in range(0, steps):
        thresholds[i] = copy_list[(i+1)*stepsize-1]
    if len(copy_list) % steps != 0:
        thresholds[i] = copy_list[-1]

    return thresholds


def discretize_list(data, thresholds):
    """Return discretized data basd on threshold"""

    return map(lambda x: next(index for index, value in enumerate(thresholds) if value >= x),
               data)


def discretize(real_value, thresholds):
    """Return discretized real_value basd on threshold"""

    return next(index for index, value in enumerate(thresholds) if value >= real_value)
               
               
def discretize_df(df, sym, steps=10):
    """Return discretized dataframe (only one column) basd on threshold which is calcualted inside this function"""
    thresholds = compute_discretizing_thresholds(df[sym], steps=steps)
    discrete_df = df[sym].map(lambda x: next(index for index, value in enumerate(thresholds) if value >= x))

    return discrete_df 

def discretize_df_thresholds(df, thresholds):
    """Return discretized dataframe (only one column) basd on threshold"""
    def discretize(time_series_value):
        try:
            category = next(index for index, value in enumerate(thresholds)
                            if value >= time_series_value)
        except StopIteration:
            return len(thresholds) - 1
        return category

    discrete_df = df.map(discretize)

    return discrete_df 


def stack_digits(data):
    """stack digits to interger """
    
    num = 0
    lenth = len(data)
    
    for i in range(0,lenth):
        num += data[i]* (10**(lenth-i-1))
    
    return num
    

def discretize_return_since_entry(ev, sv, threshold=0.1):
    # 10% exit point
    return_since_entry = (ev - sv)/sv - 1 
    if return_since_entry > threshold:
        return 1
    else:
        return 0


def plot_parameters(alpha = 0.2, \
        gamma = 0.9,\
        rar = 0.98,\
        radr = 0.9999,\
        num_trials = 500):
    """
    plot three Q-Learning parameters over trials
    """

    learning_factor = list()
    exploration_factor = list()
    discount_factor= list()
    
    for trial in range(0,num_trials): 
        learning_factor.append(alpha)  
        discount_factor.append(gamma)
        exploration_factor.append(rar)
        rar = rar * radr 
     
    plt.plot(discount_factor, color='blue', label='Discount factor')
    plt.plot(learning_factor, color='red', label='Learning factor')
    plt.plot(exploration_factor, color='green', label='Exploration factor')
    
    plt.title("Parametes of Q-Learner")
    plt.ylabel("Parameter Value")
    plt.xlabel("Trial Number")
    plt.legend()
    
    plt.show()

