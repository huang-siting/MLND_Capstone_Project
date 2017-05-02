"""
Implementing QLearner  

The code follows the basic architecture outlined in the template from the following project (c) 2015 Tucker Balch
http://quantsoftware.gatech.edu/Summer_2016_Project_5
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        learning = True,\
        verbose = False):

        self.verbose = verbose
        self.learning = learning
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma 
        self.rar = rar  
        self.radr = radr  
        self.dyna = dyna  

        self.s = 0
        self.a = 0

        self.Q = np.zeros((num_states, num_actions)) 

        if self.dyna > 1: 

            self.R = np.zeros((num_states, num_actions)) 

            # initialize Tc with small non-zero values 
            self.Tc = 0.00001 * np.ones((num_states, num_actions, num_states)) 
            self.T = 1.0 / num_states * np.ones((num_states, num_actions, num_states)) 


    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s

        if self.learning: 
            action = np.argmax(self.Q[s,:])
        else: 
            action = rand.randint(0, self.num_actions-1)

        if self.verbose: print "s =", s,"a =",action, self.Q

        return action
    

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
   
        if self.learning:
            # update Q-table
            self.Q[self.s, self.a] = (1-self.alpha) * self.Q[self.s, self.a] + \
                    self.alpha * (r + self.gamma * np.max(self.Q[s_prime,:]))

            # TODO: speed up Dyna Q
            if self.dyna > 0: 

                # learning T
                self.Tc[self.s, self.a, s_prime] += 1 
                self.T[self.s, self.a, :] = self.Tc[self.s, self.a, :]/ np.sum(self.Tc[self.s, self.a, :])
            
                # learning R
                self.R[self.s, self.a] = (1-self.alpha) * self.R[self.s, self.a] + self.alpha * r

                # repeat dyna times: halucinate experience
                s_samples = np.random.randint(0, self.num_states-1, self.dyna)
                a_samples = np.random.randint(0, self.num_actions-1, self.dyna)


                for i in range(self.dyna):

                    #random previously observed state and action
                    #d_s = np.random.choice(np.nonzero(self.Q)[0])
                    #d_a = np.random.choice(np.nonzero(self.Q[d_s,:])[0])

                    d_s = s_samples[i]
                    d_a = a_samples[i]

                    d_s_prime = np.argmax(np.random.multinomial(1, self.T[d_s, d_a, :]))
                    #d_s_prime = np.random.choice(self.num_states, p=self.T[d_s, d_a, :])
                    d_R = self.R[d_s, d_a]
                
                    #one step Q-learning 
                    self.Q[d_s, d_a] = (1-self.alpha) * self.Q[d_s, d_a] + \
                            self.alpha * (d_R + self.gamma * np.max(self.Q[d_s_prime,:]))

            # return an action
            prand = np.random.random()
            if prand < self.rar: 
                action = rand.randint(0, self.num_actions-1)
            else: 
                # return list of actions with same max Q and randomly choose one action from the list
                mylist = np.argwhere(self.Q[s_prime,:] == np.amax(self.Q[s_prime,:]))
                action = rand.choice(mylist)[0]

            self.s = s_prime
            self.a = action
            self.rar = self.rar * self.radr
        else: 
            action = rand.randint(0, self.num_actions-1) #return random action if not learning

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r, "Q=", self.Q

        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
