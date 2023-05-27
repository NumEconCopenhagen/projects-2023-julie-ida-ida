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
        """ baseline parameters and set 1 + set 2 parameters """

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

    def expressions(self,tau):
        """ Calculating utility for baseline parameters """

        par = self.par

        # Defining L, G and V
        L = ((-par.kappa*par.nu+np.sqrt(par.nu*(4*par.alpha*((1-tau)*par.w)**2+par.kappa**2*par.nu))))/(2*par.nu*(1-tau)*par.w)

        G = tau*par.w*L

        V = np.log(((par.kappa+(1-tau)*par.w*L)**par.alpha)*G**(1-par.alpha))-par.nu*(L**2)/2

        # Setting -V since we want to maximize
        return -V
    
    def optimal_tau(self):
        """ Calculating optimal tau for baseline parameters """

        # Setting bounds and guess
        bounds = [(0,1)]
        guess = 0.7

        # Optimize
        solution = optimize.minimize(self.expressions,
                                   guess,
                                   method='Nelder-Mead',
                                   bounds=bounds)

        opt_tau = solution.x[0]
        
        # Printing the result
        print(f'Optimal tau = {opt_tau:.4}')

        return solution
    
    def calc_utility1(self,x):
        """ Calculating utility for set 1 parameters """

        L = x[0]
        tau = x[1]

        par = self.par

        # Defining G and utility for set 1 parameters
        G1 = tau * par.w * L

        utility = ((((par.alpha*(par.kappa+(1-tau)*par.w*L)**((par.sigma1-1)/par.sigma1) + (1-par.alpha)*G1**((par.sigma1-1)/par.sigma1)))**(par.sigma1/(par.sigma1-1)))**(1-par.rho1)-1)/(1-par.rho1)

        disutility = par.nu*(L**(1+par.epsilon1)/(1+par.epsilon1))

        # Setting - in front since we want to maximize
        return -utility + disutility
    
    def optimal_L1(self,tau):
        """ Calculating optimal L and G for set 1 parameters """

        par = self.par 
        
        # Setting bounds and guess
        bounds = [(1e-8,24)]
        x0 = [12, 0.5]

        # Optimize
        solution1 = optimize.minimize(self.calc_utility1,
                                   x0,
                                   method='Nelder-Mead',
                                   bounds=bounds)

        opt_L = solution1.x[0]
        opt_tau_new = solution1.x[1]

        G_opt = tau*par.w*opt_L

        # Print results
        print(f'Optimal L = {opt_L:.6}')
        print(f'Optimal G = {G_opt:.6}')

        return opt_tau_new

    def calc_utility2(self,x):
        """ Calculating utility for set 2 parameters """

        L = x[0]
        tau = x[1]

        par = self.par

        # Defining G and utility for set 2 parameters
        G2 = tau * par.w * L

        utility2 = ((((par.alpha*(par.kappa+(1-tau)*par.w*L)**((par.sigma2-1)/par.sigma2) + (1-par.alpha)*G2**((par.sigma2-1)/par.sigma2)))**(par.sigma2/(par.sigma2-1)))**(1-par.rho2)-1)/(1-par.rho2)

        disutility2 = par.nu*(L**(1+par.epsilon2)/(1+par.epsilon2))

        # Setting - in front since we want to maximize
        return -utility2 + disutility2
    
    def optimal_L2(self,tau):
        """ Calculating optimal L and G for set 2 parameters """

        par = self.par 
        
        # Setting bounds and guess
        bounds = [(1e-8,24)]
        x0 = [12, 0.5]

        # Optimize
        solution2 = optimize.minimize(self.calc_utility2,
                                   x0,
                                   method='Nelder-Mead',
                                   bounds=bounds)

        opt_L2 = solution2.x[0]
        opt_tau_new2 = solution2.x[1]

        G_opt2 = tau*par.w*opt_L2

        # Print results
        print(f'Optimal L = {opt_L2:.6}')
        print(f'Optimal G = {G_opt2:.6}')

        return opt_tau_new2


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
    
    def ex_post_value_delta(self, shocks, delta=0):

        par = self.par

        # Defining shocks and optimal l
        kappa1 = np.exp(shocks)  # exponential due to log
        opt_l1 = ((1 - par.eta_val) * kappa1 / par.w_val) ** (1 / par.eta_val)  # optimal l

        # Creating an empty list to store the solution
        value1 = []

        # For loop
        previous_l = opt_l1[0]  # Set initial value of previous_l

        for t in range(par.T):
            if abs(opt_l1[0] - previous_l) > delta:
                current_l = opt_l1[0]
            else:
                current_l = previous_l
            
            profit = kappa1[t] * current_l ** (1 - par.eta_val) - par.w_val * current_l  # defining profit from optimal l 
            adjustment_cost = (current_l != previous_l) * par.iota_val if t > 0 else 0  # defining adjustment cost for hiring/firing 
            value1.append(par.R_val ** (-t) * (profit - adjustment_cost))

            previous_l = current_l

        return np.sum(value1)

    def optimal_delta(self):
        """ Calculating optimal Delta """
        
        # Setting bounds and guess
        bounds = [(1e-8,24)]
        x0 = [12, 0.5]

        # Optimize
        solution4 = optimize.minimize(self.ex_post_value_delta,
                                   x0,
                                   method='Nelder-Mead',
                                   bounds=bounds)

        opt_delta = solution4.x[0]

        # Print results
        print(f'Optimal Delta = {opt_delta}')

        return solution4