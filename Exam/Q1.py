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
