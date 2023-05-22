# Importing modules 
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

        # a. Household parameters 
        par.rho = 0.03 
        par.n = 0.05 # population growth 

        # b. Firms parameters 
        par.p_f = 'cobb-douglas' # OBS! Hvorfor skriver man denne ind? 
        par.alpha = 0.3

        # c. Government parameters 
        par.tau = 0.0 # OBS! Wage tax? Troede det var bidraget fra de unge til de gamle i samme periode? 

        # d. Start values and length of simulation
        par.K_ini = 0.1 # initial capital stock 
        par.L_ini = 1.0 # OBS! Hvad er dette? De skriver initial population 
        par.simT = 20 # length og simulation 

    def allocate(self):
        """ allocate arrays for simulation """

        par = self.par 
        sim = self.sim 

        # a. List of variables
        household = ['C1', 'C2', 's']
        firm = ['K', 'Y', 'L', 'k']
        prices = ['w', 'r']

        # b. Allocates
        allvarnames = household + firm + prices 
        for varname in allvarnames:
            sim.__dict__[varname] = np.nan*np.ones(par.simT)
        # OBS! Forklar denne kode! 

    def simulate(self,do_print=True):
        """ simulate model """
        
        t0 = time.time() 

        par = self.par 
        sim = self.sim

        # a. Initial values for simulation 
        sim.K[0] = par.K_ini
        sim.L[0] = par.L_ini

        # b. Simulate the model 
        for t in range(par.simT):

            # i. Simulate before s 
            self.simulate_before_s(par,sim,t)
            if t == par.simT-1: continue 

            # ii. Find bracket to search 
            s_min,s_max = self.find_s_bracket(par,sim,t)

            # iii. Find optimal s 
            obj = lambda s: self.calc_euler_error(s,par,sim,t=t)
            result = optimize.root_scalar(obj,bracket=(s_min,s_max),method='bisect') 
            s = result.root
            # OBS! Forklar denne kode! 

            # iv. Log optimal savings rate 
            sim.s[t] = s

            # v. Simulate after s 
            self.simulate_after_s(par,sim,t,s)
        
        if do_print: print(f'simulation done in {time.time()-t0:.2f} secs')

    def find_s_bracket(self,par,sim,t,maxiter=500,do_print=False):
        """ find bracket for s to search in """

        # a. Minimum/maximum bracket 
        s_min = 0.0 + 1e-8 # save almost nothing 
        s_max = 1.0 - 1e-8 # save almost everything 

        # b. It is always possible to save a lot 
        value = self.calc_euler_error(s_max,par,sim,t)
        sign_max = np.sign(value)
        if do_print: print(f'euler-error for s = {s_max:12.8f} = {value:12.8f}')

        # c. Find brackets 
        lower = s_min 
        upper = s_max 

        it = 0 # OBS! Hvad gør denne? 
        while it < maxiter: 

            # i. Midpoint and value 
            s = (lower+upper)/2 # midpoint
            value = self.calc_euler_error(s,par,sim,t)

            if do_print: print(f'euler-error for s = {s:12.8f} = {value:12.8f}')

            # ii. Check conditions
            valid = not np.isnan(value)
            correct_sign = np.sign(value)*sign_max < 0
        
            # iii. Next step
            if valid and correct_sign:
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
            
            # iv. Increment 
            it += 1 
    
    def calc_euler_error(self,s,par,sim,t):
        """ target function for finding s with bisection """

        # a. Simulate forward 
        self.simulate_after_s(par,sim,t,s)
        self.simulate_before_s(par,sim,t+1)

        par.beta = 1/(1+par.rho)

        # b. Euler equation 
        LHS = sim.C1[t]**(-1) 
        RHS = (1+sim.r[t+1])*par.beta * sim.C2[t+1]**(-1)

        return LHS-RHS 
    
    def simulate_before_s(self,par,sim,t):
        """ simulate forward """

        if t == 0: 
            sim.K[t] = par.K_ini
            sim.L[t] = par.L_ini
        if t > 0:
            sim.L[t] = sim.L[t-1]*(1+par.n)
        
        # a. Production 
        sim.Y[t] = sim.K[t]**par.alpha * (sim.L[t])**(1-par.alpha)

        # b. Factor prices 
        sim.r[t] = par.alpha * sim.K[t]**(par.alpha-1) * (sim.L[t])**(1-par.alpha)
        sim.w[t] = (1-par.alpha) * sim.K[t]**(par.alpha) * (sim.L[t])**(-par.alpha)

        # c. Consumption 
        sim.C2[t] = (1+sim.r[t])*(sim.K[t]) # Hvorfor K og ikke s?
    
    def simulate_after_s(self,par,sim,t,s):
        """ simulate forward """

        sim.k[t] = sim.K[t]/sim.L[t]

        # a. Consumption of young 
        sim.C1[t] = (1-par.tau)*sim.w[t]*(1.0-s) * sim.L[t] # Skal denne ændres?

        # b. End-of-periods stock 
        I = sim.Y[t] - sim.C1[t] - sim.C2[t]
        sim.K[t+1] = sim.K[t]+I

    def sim_results(self):
        # a. Importing the model 
        model = modelclass()

        par = model.par 
        sim = model.sim 

        # b. Guess for savings 
        s_guess = 0.4

        par.beta = 1/(1+par.rho)

        #sim.K[0] = par.K_ini
        #sim.L[0] = par.L_ini

        # c. Simulating t = 0 and t 
        self.simulate_before_s(par,sim,t=0)
        print('Consumption by old people in period t = 0',f'{sim.C2[0] = : .4f}')

        self.simulate_after_s(par,sim,s=s_guess,t=0)
        print('Consumption by young in period t = 0',f'{sim.C1[0] = : .4f}')

        self.simulate_before_s(par,sim,t=1)
        print('Consumption by old people in period t',f'{sim.C2[1] = : .4f}')

        self.simulate_after_s(par,sim,s=s_guess,t=1)
        print('Consumption by young people in period t',f'{sim.C1[1] = : .4f}')

        # d. Calculating the Euler error
        LHS_Euler = sim.C1[0]**(-1)
        RHS_Euler = (1 + sim.r[1])*par.beta * sim.C2[1]**(-1)
        print(f'euler-error = {LHS_Euler-RHS_Euler:.8f}')

        # e. Check if Euler error converges towards 0:
        model.simulate()
        LHS_Euler = sim.C1[18]**(-1)
        RHS_Euler = (1+sim.r[19])*par.beta * sim.C2[19]**(-1)
        print("euler error after model has been simulated", LHS_Euler-RHS_Euler)

        # d. Save steady state for this version
        sim.k_orig=sim.k.copy()