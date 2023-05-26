# Importing modules 
from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


class examclass1():

    def __init__(self,do_print=True):
        """ create the model """
        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()
        self.sol = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()

    def setup(self):
        """ baseline parameters """

        par = self.par 
        sol = self.sol

        par.alpha = 0.50 
        par.kappa = 1.00
        par.nu = 1/(2*16**2)
        par.w = 1.0
        par.tau = 0.30
        par.G = 1.0

        sol.L = np.zeros(par.w)

    def calc_utility(self):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. Private consumption
        C = par.kappa + (1-par.tau)*par.w*sol.L

        # b. Utility
        utility = np.fmax(np.log(C**par.alpha * par.G**(1-par.alpha)))

        # c. Disutlity
        disutility = par.nu*((sol.L**2)/2)
        
        return utility - disutility
    
    def solve(self,do_print=False):
        """ solve model """

        opt = SimpleNamespace()

        x = np.linspace(1.0,2.0)
        L = np.meshgrid(x) # all combinations 

        u = self.calc_utility(L)

        I = (L > 24)
        u[I] = -np.inf

        j = np.argmax(u)

        opt.L = L[j]

        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt
    
    def solve1_4(self, tau=None):

        opt_as = SimpleNamespace() 
        
        bounds = [(1e-8,24-1e-8)]*2
        guess = [0.5, 0.8]

        result1_4 = optimize.minimize(self.calc_utility,
                                   guess,
                                   method='Nelder-Mead',
                                   bounds=bounds,
                                   tol=1e-10)
        
        opt_as.tau = result1_4.x[0]
        
        print(f' optimal tau = {opt_as.tau}')

        return result1_4