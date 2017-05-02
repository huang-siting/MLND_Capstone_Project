import datetime as dt
import QLearner as ql
import pandas as pd
import util as ut
import numpy as np

class StrategyLearner(object):

    # constructor
    def __init__(self, window=15,\
            alpha = 0.1, \
            gamma = 0.9, \
            rar = 0.98, \
            radr = 0.9999, \
            learning = True, verbose = False):

        self.alpha = alpha
        self.gamma = gamma 
        self.rar = rar  
        self.radr = radr  
        self.verbose = verbose
        self.window = window
        self.learning = learning 

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "GOOG", \
        sd=dt.datetime(2004,8,19), \
        ed=dt.datetime(2005,8,19), \
        sv = 10000): 

        syms=[symbol]
        dates = pd.date_range(sd, ed)

        # read price data
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print prices.head()

        # discretize ratio close/SMA
        ratio_close_SMA = ut.compute_ratio_close_SMA(prices, window=self.window)
        ratio_close_SMA.dropna(inplace=True)
        self.ratio_close_SMA_thresholds = ut. compute_discretizing_thresholds(ratio_close_SMA[symbol], steps=10)
        discrete_close_SMA = ut.discretize_df_thresholds(ratio_close_SMA[symbol], \
                self.ratio_close_SMA_thresholds)

        # discretize percent_b
        percent_b = ut.compute_percent_b(prices, window=self.window)
        percent_b.dropna(inplace=True)
        self.percent_b_thresholds = ut.compute_discretizing_thresholds(
        percent_b[symbol], steps=10)
        discrete_percent_b = ut.discretize_df_thresholds(percent_b[symbol],\
                self.percent_b_thresholds)

        # compute daily returns of the stock
        stock_daily_returns = ut.compute_daily_returns(prices)

        # compute daily absolute changes of the stock
        stock_daily_changes = ut.compute_daily_changes(prices)

        # initiate a Q-learner
        self.learner = ql.QLearner(num_states=3000,\
            num_actions = 3, \
            alpha = self.alpha, \
            gamma = self.gamma, \
            rar = self.rar, \
            radr = self.radr, \
            dyna = 0, \
            learning = self.learning,\
            verbose=False) 


        # initiate cumulative return
        cumulative_return = list() 

        num_trials = 300
        for trial in range(0,num_trials): 
            
            # initiate holding (holding takes values -1, 0 , 1)
            holding = 0
            # initiate portfolio value
            portfolio = sv 

            ## for each day in the training data with features 
            # set the state and get the first action
            state_since_entry = ut.discretize_return_since_entry(portfolio, sv, threshold=0.1)
            state = ut.stack_digits([holding+1,\
                    state_since_entry, \
                    discrete_close_SMA[0],\
                    discrete_percent_b[0]]) 

            action = self.learner.querysetstate(state) 
            holding = action - 1 

            ##move to new state according to action and then get a new action
            for t in discrete_close_SMA[1:].index.tolist(): 

                # daily return as rewards
                r = stock_daily_returns[symbol][t] * holding

                state_since_entry = ut.discretize_return_since_entry(portfolio, sv, threshold=0.1)

                state = ut.stack_digits([holding+1, \
                        state_since_entry, \
                        discrete_close_SMA[t], \
                        discrete_percent_b[t]]) 

                action = self.learner.query(state,r)

                #update portfolio value
                portfolio += stock_daily_changes[symbol][t] * holding * 100
                holding = action - 1

            cumulative_return.append(portfolio/sv - 1)

        #print "cumulative return of training:",cumulative_return[-1]

        return cumulative_return


    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,12,31), \
        ed=dt.datetime(2011,12,31), \
        sv = 10000):

        syms=[symbol]
        dates = pd.date_range(sd, ed)

        # read price data
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later

        # discretize ratio close/SMA
        ratio_close_SMA = ut.compute_ratio_close_SMA(prices, window=self.window)
        ratio_close_SMA.dropna(inplace=True)
        discrete_close_SMA = ut.discretize_df_thresholds(ratio_close_SMA[symbol], \
                self.ratio_close_SMA_thresholds)

        # discretize percent_b
        percent_b = ut.compute_percent_b(prices, window=self.window)
        percent_b.dropna(inplace=True)
        discrete_percent_b = ut.discretize_df_thresholds(percent_b[symbol],\
                self.percent_b_thresholds)

        # compute daily returns of the stock
        stock_daily_returns = ut.compute_daily_returns(prices)

        # compute daily absolute changes of the stock
        stock_daily_changes = ut.compute_daily_changes(prices)

        holding = 0
        portfolio = sv 

        trades = prices_all[[symbol,]]  # only portfolio symbols
	trades.values[:,:] = 0 # set them all to nothing

        for t in discrete_close_SMA.index.tolist(): 

            # compute current state
            state_since_entry = ut.discretize_return_since_entry(portfolio, sv, threshold=0.1)
            state = ut.stack_digits([holding+1, \
                    state_since_entry, \
                    discrete_close_SMA[t], \
                    discrete_percent_b[t]]) 
            action = self.learner.querysetstate(state) 

            #update portfolio value
            portfolio += stock_daily_changes[symbol][t] * holding * 100
            old_holding = holding
            holding = action - 1
	    trades[symbol][t] = (holding - old_holding) * 100

        cumulative_return = portfolio/sv - 1

        #print "cumulative return of testing:",cumulative_return

        return trades, cumulative_return

if __name__=="__main__":
    print "One does not simply think up a strategy"

