import numpy as np
from scipy import optimize
import sympy as sm
from types import SimpleNamespace
import math

class examclass2_2():

    def __init__(self,do_print=True):
        """ create the model """
        if do_print: print('initializing the model:')

        # a. Create namespaces 
        self.par = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()


    def setup(self):
        """ baseline parameters """

        par = self.par

        # a. Parameter values  
        par.eta_val = 0.5
        par.w_val = 1.0 
        par.rho_val = 0.90
        par.iota_val = 0.01
        par.sigma_eps = 0.10
        par.R_val = (1+0.01)**(1/12)
        
        # b. Planning horizon 
        par.T = 10*12 # months 


    def ex_post_value(self,shocks):

        par = self.par

        # a. Defining shocks and optimal l 
        kappa = np.exp(shocks) # exponential due to log 
        opt_l = ((1-par.eta_val)*kappa/par.w_val)**(1/par.eta_val) # optimal l from Q1

        # b. Creating an empty list to store the solution
        value = []

        # c. For loop 
        for t in range(par.T):
            profit = kappa[t]*opt_l[t]**(1-par.eta_val)-par.w_val*opt_l[t] # defining profit from optimal l 
            adjustment_cost = (opt_l[t] != opt_l[t-1]) * par.iota_val # defining adjustment cost for hiring/firing 
            value.append(par.R_val**(-t)*(profit-adjustment_cost)) 
    
        return np.sum(value)

    def ex_ante_value(self,K):

        par = self.par

        # a. Defining the distibution for the shocks 
        shocks = np.random.normal(loc=-0.5 * par.sigma_eps**2, scale=par.sigma_eps, size=(K, par.T))
        values = np.zeros(K) 

        # b. For loop 
        for k in range(K):
            values[k] = self.ex_post_value(shocks[k])

        return np.mean(values)
    






#### BRUGES DET HER? ####
class examclass2_3():
    
    def __init__(self,do_print=True):
        """ create the model """
        if do_print: print('initializing the model:')

        # a. Create namespaces 
        self.par = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()


    def setup(self):
        """ baseline parameters """

        par = self.par

        # a. Preferences 
        par.eta_val = 0.5
        par.w_val = 1.0 
        par.rho_val = 0.90
        par.iota_val = 0.01
        par.sigma_eps = 0.10
        par.R_val = (1+0.01)**(1/12)
        par.Delta = 0.05
        
        # b. Planning horizon 
        par.T = 10*12 # months 

    def ex_post_value(self,shocks):

        par = self.par

        # a. Defining shocks and optimal l 
        kappa = np.exp(shocks) # exponential due to log 
        opt_l = ((1-par.eta_val)*kappa/par.w_val)**(1/par.eta_val) # optimal l from Q1
        l = l[t-1]

        for t in range(par.T):
            employee = (l[t-1] - opt_l[t])  

        #if employee>par.Delta:
        #    l = opt_l
        #else: 
        #    l = l[t-1]

        # b. Creating an empty list to store the solution
        value = []

        # c. For loop 
        for t in range(par.T):
            if employee > par.Delta:
                profit = kappa[t]*opt_l[t]**(1-par.eta_val)-par.w_val*opt_l[t] # defining profit from optimal l 
                adjustment_cost = (opt_l[t] != opt_l[t-1]) * par.iota_val # defining adjustment cost for hiring/firing 
                value.append(par.R_val**(-t)*(profit-adjustment_cost))
            else:
                profit = kappa[t]*l[t]**(1-par.eta_val)-par.w_val*l[t] # defining profit from optimal l 
                adjustment_cost = (l[t] != l[t-1]) * par.iota_val # defining adjustment cost for hiring/firing 
                value.append(par.R_val**(-t)*(profit-adjustment_cost))
    
        return np.sum(value)

    def ex_ante_value(self,K):

        par = self.par 

        # a. Defining the distibution for the shocks 
        shocks = np.random.normal(loc=-0.5 * par.sigma_eps**2, scale=par.sigma_eps, size=(K, par.T))
        values = np.zeros(K) 

        # b. For loop 
        for k in range(K):
            values[k] = self.ex_post_value(shocks[k])

        return np.mean(values)