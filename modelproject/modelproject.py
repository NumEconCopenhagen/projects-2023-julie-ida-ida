from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize

class modelclass():

    def __init__(self,do_print=True):
        """ create the model """

        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

        par = self.par 

        # Values of parameters
        par.alpha = 0.5
        par.tau = 0.4
        par.rho = 0.2
        par.A = 1
        par.k = 0.0125

        # Length of simulations
        par.simT = 10_000

    def allocate(self):
        """ allocate arrays for simulation """
        par = self.par
        sim = self.sim

        household = ['C1','C2']
        prices = ['w', 'r', 'rt']
        firm = ['K']

        allvarnames = household + firm + prices 
        for varname in allvarnames:
            sim.__dict__[varname] = np.nan*np.ones(par.simT)

    def simulate(self):
        """ simulate the full model """
        
        par = self.par 
        sim = self.sim

        self.__init__()
        self.allocate()

        # values
        sim.k = par.k

        for t in range(par.simT):
            # i. simulate before s
        
            self.simulate_before_s(par,sim,t,s)

            if t == par.simT: continue          

            # ii. find optimal s
            obj = lambda s: self.calc_euler_error(par,sim,t=t)
            result = optimize.root_scalar(obj,bracket=[1E-10,10],method='bisect')
            s = result.root

            # iii. simulate after s
            self. simulate_after_s(par,sim,t,s)


    def simulate_before_s(self,par,sim,t,s):
        """ simulate forward"""

        if t >= 0:
            sim.k[t] = sim.k[t+1]
        else: 
            raise NotImplementedError('')

        sim.C2[t] = s*(1+sim.r) + par.tau * sim.w


    def simulate_after_s(self, par,sim,t,s):
        sim.C1[t] = sim.w[t]*(1-par.tau) - s


    def calc_euler_error(self,s,par,sim,t):


        self.simulate_after_s(par,sim,t,s)
        self.simulate_before_s(par,sim,t+1)

        LHS = sim.C2[t+1]
        RHS = (1+sim.r[t+1])/(1+par.rho) * sim.C1

        return LHS-RHS