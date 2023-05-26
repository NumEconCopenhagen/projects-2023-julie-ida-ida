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

        # Baseline parameters 
        par.alpha = 0.5
        par.kappa = 1.0 
        par.nu = 1/(2*16**2)
        par.w = 1.0
        par.tau = 0.30

        # Set 1 parameters 
        par.rho1 = 1.001
        par.sigma1 = 1.001 
        par.epsilon1 = 1.0

        # Set 2 parameters 
        par.rho2 = 1.5
        par.sigma2 = 1.5 
        par.epsilon2 = 1.0

    def optimal_tau(self):

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

        L = ((-par.kappa*par.nu+np.sqrt(par.nu*(4*par.alpha*((1-tau)*par.w)**2+par.kappa**2*par.nu))))/(2*par.nu*(1-tau)*par.w)
        G = tau*par.w*L

        V = np.log(((par.kappa+(1-tau)*par.w*L)**par.alpha)*G**(1-par.alpha))-par.nu*(L**2)/2

        return -V
    
    def calc_utility1(self,x):
        """ calculates utility """

        L = x[0]
        tau = x[1]

        par = self.par

        G1 = tau * par.w * L

        utility = ((((par.alpha*(par.kappa+(1-tau)*par.w*L)**((par.sigma1-1)/par.sigma1) + (1-par.alpha)*G1**((par.sigma1-1)/par.sigma1)))**(par.sigma1/(par.sigma1-1)))**(1-par.rho1)-1)/(1-par.rho1)

        disutility = par.nu*(L**(1+par.epsilon1)/(1+par.epsilon1))

        return -utility + disutility
    
    def optimal_L1(self,tau):

        par = self.par 
        
        bounds = [(1e-8,24)]
        x0 = [12, 0.5]

        solution1 = optimize.minimize(self.calc_utility1,
                                   x0,
                                   method='Nelder-Mead',
                                   bounds=bounds)

        opt_L = solution1.x[0]
        optimal_tau = tau

        G_opt = optimal_tau*par.w*opt_L

        print(f'Optimal L = {opt_L}')
        print(f'Optimal G = {G_opt}')

        return opt_L, optimal_tau


    def calc_utility2(self,x):
        """ calculates utility """

        L = x[0]
        tau = x[1]

        par = self.par

        G2 = tau * par.w * L

        utility2 = ((((par.alpha*(par.kappa+(1-tau)*par.w*L)**((par.sigma2-1)/par.sigma2) + (1-par.alpha)*G2**((par.sigma2-1)/par.sigma2)))**(par.sigma2/(par.sigma2-1)))**(1-par.rho2)-1)/(1-par.rho2)

        disutility2 = par.nu*(L**(1+par.epsilon2)/(1+par.epsilon2))

        return -utility2 + disutility2
    
    def optimal_L2(self,tau):

        par = self.par 
        
        bounds = [(1e-8,24)]
        x0 = [12, 0.5]

        solution1 = optimize.minimize(self.calc_utility2,
                                   x0,
                                   method='Nelder-Mead',
                                   bounds=bounds)

        opt_L2 = solution1.x[0]
        optimal_tau2 = tau

        G_opt2 = optimal_tau2*par.w*opt_L2

        print(f'Optimal L = {opt_L2}')
        print(f'Optimal G = {G_opt2}')

        return opt_L2, optimal_tau2

class examclass2():

    def __init__(self,do_print=True):
        """ create the model """
        if do_print: print('initializing the model:')

        # Create namespaces 
        self.par = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()


    def setup(self):
        """ baseline parameters """

        par = self.par

        # Parameter values  
        par.eta_val = 0.5
        par.w_val = 1.0 
        par.rho_val = 0.90
        par.iota_val = 0.01
        par.sigma_eps = 0.10
        par.R_val = (1+0.01)**(1/12)
        
        # Planning horizon 
        par.T = 10*12 # months 


    def ex_post_value(self,shocks):

        par = self.par

        # Defining shocks and optimal l 
        kappa = np.exp(shocks) # exponential due to log 
        opt_l = ((1-par.eta_val)*kappa/par.w_val)**(1/par.eta_val) # optimal l 

        # Creating an empty list to store the solution
        value = []

        # For loop 
        for t in range(par.T):
            profit = kappa[t]*opt_l[t]**(1-par.eta_val)-par.w_val*opt_l[t] # defining profit from optimal l 
            adjustment_cost = (opt_l[t] != opt_l[t-1]) * par.iota_val # defining adjustment cost for hiring/firing 
            value.append(par.R_val**(-t)*(profit-adjustment_cost)) 
    
        return np.sum(value)

    def ex_ante_value(self,K):

        par = self.par

        # Defining the distibution for the shocks 
        shocks = np.random.normal(loc=-0.5 * par.sigma_eps**2, scale=par.sigma_eps, size=(K, par.T))
        values = np.zeros(K) 

        # For loop 
        for k in range(K):
            values[k] = self.ex_post_value(shocks[k])

        return np.mean(values)