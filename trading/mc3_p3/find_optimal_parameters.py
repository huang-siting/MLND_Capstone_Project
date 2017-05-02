"""
find optimal hyperparameters for Strategy learner
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import datetime as dt
import util as ut
import StrategyLearner as sl
import matplotlib.pyplot as plt
import numpy as np

def run_simulations(symbol = "IBM", \
        sd_train = dt.datetime(2007,12,31), \
        ed_train = dt.datetime(2009,12,31),\
        sd_test = dt.datetime(2009,12,31),\
        ed_test = dt.datetime(2011,12,31), \
        sv =10000, \
        alpha=0.2, \
        rar=0.98, \
        radr=0.99, \
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
    train_cumulative_return_buy_hold_strategy = ((train_prices.ix[-1,:][0]- train_prices.ix[0,:][0])*100+sv)/sv - 1
    if verbose:
        print "cumulative return of buy-and-hold strategy on training:",\
                train_cumulative_return_buy_hold_strategy 
    
    cumulative_return_buy_hold_strategy = ((prices.ix[-1,:][0]- prices.ix[0,:][0])*100+sv)/sv - 1
    if verbose:
        print "cumulative return of buy-and-hold strategy on testing:",\
                cumulative_return_buy_hold_strategy 

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

    return (avg_cumulative_returns_train, avg_cumulative_returns_test)


def find_optimal_parameters():
    """
    Run simulaions with combinations of parameters 
    """

    log_simulations = dict()

    for alpha in [0.1, 0.2]:
        for rar in [0.9, 0.98]:
            for radr in [0.9999, 0.999]:
                for window in [10,15,20]:
                    log_simulations[(alpha, rar,radr,window)] = run_simulations(symbol = "IBM", \
                            sd_train = dt.datetime(2007,12,31), \
                            ed_train = dt.datetime(2009,12,31),\
                            sd_test = dt.datetime(2009,12,31),\
                            ed_test = dt.datetime(2011,12,31), \
                            sv =10000, \
                            alpha=alpha, \
                            rar=rar, \
                            radr=radr, \
                            window=window, \
                            num_simulation = 10,\
                            plot_results = False, \
                            verbose = False)
                    print "Parameters (alpha, rar,radr,window):", (alpha, rar,radr,window)
                    print "cumulative_returns_train, cumulative_returns_test",\
                            log_simulations[(alpha, rar,radr,window)]
    return log_simulations


if __name__=="__main__":

       #print run_simulations()
       print find_optimal_parameters()
