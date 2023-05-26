import numpy as np
from scipy import optimize
import sympy as sm
from types import SimpleNamespace
import math

class examclass1():

    def __init__(self,do_print=True):
        """ create the model """
        if do_print: print('initializing the model:')

        # a. Create namespaces 
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()


    def setup(self):
        """ baseline parameters """

        par = self.par
        sol = self.sol

        # a. Parameter values  
        par.alpha = 0.5
        par.kappa = 1.0 
        par.nu = 1/(2*16**2)
        par.w = 1.0
        par.tau = 0.30
        opt_tau = None

        sol.L = []

    def calc_utility(self):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. Consumption 
        C = par.kappa+(1-par.tau)*par.w*sol.L

        # b. Government consumption 
        G = par.tau*par.w*sol.L

        # c. Utility 
        utility = np.fmax(np.log(C**par.alpha * G**(1-par.alpha)))

        # d. Disutlity of work
        disutility = par.nu*((sol.L**2))/2
        
        return utility - disutility

    def opt_tau(self):

        par = self.par
        sol = self.sol
        opt = SimpleNamespace() 

        bounds = [(0,1)]
        guess = 0.7

        solution = optimize.minimize(self.expressions,
                                   guess,
                                   method='Nelder-Mead',
                                   bounds=bounds)

        opt_tau = solution.x[0]
        
        print(f'Optimal tau = {opt_tau}')

        return solution
    
    def expressions(self,tau):

        par = self.par
        sol = self.sol

        L = ((-par.kappa*par.nu+np.sqrt(par.nu*(4*par.alpha*((1-tau)*par.w)**2+par.kappa**2*par.nu))))/(2*par.nu*(1-tau)*par.w)
        G = tau*par.w*L

        V = np.log(((par.kappa+(1-tau)*par.w*L)**par.alpha)*G**(1-par.alpha))-par.nu*(L**2)/2

        return -V

class examclass2():

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