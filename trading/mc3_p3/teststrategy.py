"""
Test a Strategy Learner
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import datetime as dt
import util as ut
import StrategyLearner as sl
import matplotlib.pyplot as plt
import numpy as np

def run_simulations_b(symbol = "IBM", \
        sd_train = dt.datetime(2007,12,31), \
        ed_train = dt.datetime(2009,12,31),\
        sd_test = dt.datetime(2009,12,31),\
        ed_test = dt.datetime(2011,12,31), \
        sv =10000, \
        alpha=0.1, \
        rar=0.98, \
        radr=0.9999, \
        window=15, \
        num_simulation = 10, \
        plot_results = False, \
        verbose = False):

    syms=[symbol]

    # read in training data
    train_dates = pd.date_range(sd_train, ed_train)
    train_prices_all = ut.get_data(syms, train_dates)  
    train_prices = train_prices_all[syms]  
    
    # read in testing data
    dates = pd.date_range(sd_test, ed_test)
    prices_all = ut.get_data(syms, dates)  
    prices = prices_all[syms]  
    
    # compute benchmark
    train_cumulative_return_buy_hold = ((train_prices.ix[-1,:][0]- train_prices.ix[0,:][0])*100+sv)/sv - 1
    if verbose:
        print "cumulative return of buy-and-hold strategy on training:",\
                train_cumulative_return_buy_hold 
    
    test_cumulative_return_buy_hold = ((prices.ix[-1,:][0]- prices.ix[0,:][0])*100+sv)/sv - 1
    if verbose:
        print "cumulative return of buy-and-hold strategy on testing:",\
                test_cumulative_return_buy_hold 

    cumulative_returns_train = np.zeros(num_simulation)
    cumulative_returns_test = np.zeros(num_simulation)
    
    for i in range(0,num_simulation): 
        # instantiate the strategy learner
        learner = sl.StrategyLearner(alpha=alpha,\
                rar = rar,\
                radr = radr, \
                verbose = False)
    
        # learning
        cumulative_return = learner.addEvidence(symbol = symbol,\
                    sd = sd_train, \
                    ed = ed_train, sv = 10000) 

        if plot_results: 
            plt.plot(cumulative_return)
    
        #save the final result
        cumulative_returns_train[i] = cumulative_return[-1]
        if verbose:
            print "cumulative_return of training:",cumulative_returns_train[i]
    
        # test the learner
        df_trades, cumulative_returns_test[i] = learner.testPolicy(symbol = symbol, sd = sd_test, \
        ed = ed_test, sv = 10000)
    
        if plot_results: 
            plt.title("Cumulative return on training set of ten simulations")
            plt.ylabel("Cumulative return")
            plt.xlabel("Trials")
            plt.show()
    
    if verbose:
        print "cumulative_returns_train", cumulative_returns_train 
        print "cumulative_returns_test", cumulative_returns_test

    avg_cumulative_returns_train = np.mean(cumulative_returns_train)
    avg_cumulative_returns_test = np.mean(cumulative_returns_test)

    return (avg_cumulative_returns_train,\
            train_cumulative_return_buy_hold, \
            avg_cumulative_returns_test,\
            test_cumulative_return_buy_hold)


def test_strategy():
    """
    Run simulaions on different inputs
    """

    log_simulations = dict()

    symbol_set = ["IBM", \
            "IBM",\
            "IBM",\
            "IBM",\
            "IBM",\
            "SPY",\
            "SPY",\
            "SPY",\
            "SPY",\
            "SPY",\
            "GOOG",\
            "GOOG",\
            "GOOG"]
    
    sd_train_set = [dt.datetime(2003,12,31), \
            dt.datetime(2005,12,31), \
            dt.datetime(2007,12,31), \
            dt.datetime(2009,12,31), \
            dt.datetime(2011,12,31), \
            dt.datetime(2003,12,31), \
            dt.datetime(2005,12,31), \
            dt.datetime(2007,12,31), \
            dt.datetime(2009,12,31), \
            dt.datetime(2011,12,31), \
            dt.datetime(2005,12,31), \
            dt.datetime(2007,12,31), \
            dt.datetime(2009,12,31)]
    
    ed_train_set = [dt.datetime(2005,12,31), \
            dt.datetime(2007,12,31), \
            dt.datetime(2009,12,31), \
            dt.datetime(2011,12,31), \
            dt.datetime(2013,12,31), \
            dt.datetime(2005,12,31), \
            dt.datetime(2007,12,31), \
            dt.datetime(2009,12,31), \
            dt.datetime(2011,12,31), \
            dt.datetime(2013,12,31), \
            dt.datetime(2007,12,31), \
            dt.datetime(2009,12,31),\
            dt.datetime(2011,12,31)]

    sd_test_set = [dt.datetime(2005,12,31), \
            dt.datetime(2007,12,31), \
            dt.datetime(2009,12,31), \
            dt.datetime(2011,12,31), \
            dt.datetime(2013,12,31), \
            dt.datetime(2005,12,31), \
            dt.datetime(2007,12,31), \
            dt.datetime(2009,12,31), \
            dt.datetime(2011,12,31), \
            dt.datetime(2013,12,31), \
            dt.datetime(2007,12,31), \
            dt.datetime(2009,12,31),\
            dt.datetime(2011,12,31)]

    ed_test_set = [dt.datetime(2007,12,31), \
            dt.datetime(2009,12,31), \
            dt.datetime(2011,12,31), \
            dt.datetime(2013,12,31), \
            dt.datetime(2015,12,31), \
            dt.datetime(2007,12,31), \
            dt.datetime(2009,12,31), \
            dt.datetime(2011,12,31), \
            dt.datetime(2013,12,31), \
            dt.datetime(2015,12,31), \
            dt.datetime(2009,12,31), \
            dt.datetime(2011,12,31),\
            dt.datetime(2013,12,31)]


    for i in range(12,13): 

        sd_train = sd_train_set[i]
        ed_train = ed_train_set[i]
        sd_test = sd_test_set[i]
        ed_test = ed_test_set[i]
        symbol = symbol_set[i]

        log_simulations[(symbol, sd_train, ed_train, sd_test, ed_test)] = run_simulations_b(symbol = symbol,
                sd_train = sd_train,\
                ed_train = ed_train,\
                sd_test = sd_test,\
                ed_test = ed_test,\
                sv =10000, \
                num_simulation = 10,\
                plot_results = False, \
                verbose = False)
        print "simulation:",i 
        print "Inputs:", symbol, sd_train, ed_train, sd_test, ed_test
        print "result (train_return, train_benchmark, test_return, test_benchmark):", \
                log_simulations[(symbol, sd_train, ed_train, sd_test, ed_test)] 

    return log_simulations


if __name__=="__main__":

       #print run_simulations()
       print test_strategy()

