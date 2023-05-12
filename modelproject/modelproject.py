from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize

class modelclass():

    def __init__(self,do_print=True):
        """ create the model """
        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()

        if do_print: print('calling .allocate()')
        self.allocate()

    def setup(self):
        """ baseline parameters """
        
        par = self.par 
    
        par.alpha = 0.30 # capital weight 
        par.tau = 0.10 # contribution from young to old in same period 
        par.rho = 0.05 
        par.A = 1.0 # total factor productivity 
        par.n = 0.0 # population growth is 0.0

        par.simT = 50 # length of simulation

    def allocate(self):
        """ allocate arrays for simulation """

        par = self.par 
        sim = self.sim

        # List of variables 
        household = ['C1','C2']
        prices = ['w', 'r']
        firm = ['K', 'A']

        # Allocate 
        allvarnames = household + firm + prices 
        for varname in allvarnames:
            sim.__dict__[varname] = np.nan*np.ones(par.simT)


    def simulate(self,do_print=True):
        
        t0 = time.time() 
        
        par = self.par 
        sim = self.sim

        # Iterate values 
        for t in range(par.simT):
            
            # simulate before s
            self.simulate_before_s(par,t)  

            if t == par.simT: continue   

            s_min,s_max = self.find_s_bracket(par,sim,t)        

            # ii. find optimal s
            obj = lambda s: self.calc_euler_error(s,par,sim,t=t) 
            result = optimize.root_scalar(obj,bracket=(s_min,s_max),method='bisect')
            s = result.root

            # iii. simulate after s
            self.simulate_after_s(par,sim,t,s)
        
        if do_print: print(f'simulation done in {time.time()-t0:.2f} secs') 

    def find_s_bracket(self,par,sim,t,maxiter=500,do_print=False):
        """ find bracket for s to search in """

        s_min = 0.0 + 1e-8 # saving alsmot nothing
        s_max = 1.0 - 1e-8 # saving almost everything 

        value = self.calc_euler_error(s_max,par,sim,t)
        sign_max = np.sign(value)
        if do_print: print(f'euler-error for s = {s_max:12.8f} = {value:12.8f}')

        # c. find bracket      
        lower = s_min
        upper = s_max

        it = 0
        while it < maxiter: 
                
            # i. midpoint and value
            s = (lower+upper)/2 # midpoint
            value = self.calc_euler_error(s,par,sim,t)

            if do_print: print(f'euler-error for s = {s:12.8f} = {value:12.8f}')

            # ii. check conditions
            valid = not np.isnan(value)
            correct_sign = np.sign(value)*sign_max < 0
        
            # iii. next step
            if valid and correct_sign: # found!
                s_min = s
                s_max = upper
                if do_print: 
                    print(f'bracket to search in with opposite signed errors:')
                    print(f'[{s_min:12.8f}-{s_max:12.8f}]')
                return s_min,s_max
            elif not valid: # too low s -> increase lower bound
                lower = s
            else: # too high s -> increase upper bound
                upper = s

            # iv. increment
            it += 1

        raise Exception('cannot find bracket for s')
    
    def calc_euler_error(self,s,par,sim,t):
        """ target function for finding s with bisection """
        
        # a. simulate forward
        self.simulate_after_s(par,sim,t,s)
        self.simulate_before_s(par,sim,t+1) # next period

        # c. Euler equation
        LHS = sim.C2[t+1]
        RHS = ((1+sim.r[t+1])/(1+par.rho))*sim.C1[t]

        return LHS-RHS

    def simulate_before_s(par,sim,t):
        """ simulate forward """

        for t in range(par.simT):

            # periode t
            if t == 1:
                r = sim.r[t]
                k = sim.k[t]
                w = sim.w[t]
            else:
                r = sim.r[t+1]
                k = sim.k[t+1]
                w = sim.w[t+1]

        sim.k[t] = (1/(2+par.rho))*(1-par.alpha)*par.A*sim.k[t]**par.alpha*(1-par.tau)-par.tau*((1+par.rho)/(2+par.rho))*((1-par.alpha)/par.alpha)*sim.k[t]

        sim.s[t] = (1 + par.n)*sim.k[t+1]

        sim.C2[t+1] = sim.s[t] * (1 + sim.r[t+1]) + par.tau * sim.w[t+1]
        
        
    def simulate_after_s(par,sim,t):
        """ simulate forward """

        sim.C1[t] = sim.w[t]*(1-par.tau)-sim.s[t]

        sim.k[t+1] = sim.s[t]/(1 + par.n)