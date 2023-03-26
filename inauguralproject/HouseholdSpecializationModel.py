
from types import SimpleNamespace

import numpy as np
from scipy import optimize
import scipy.stats as stats

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 0:
            H = np.min(HM,HF)
        elif par.sigma == 1:
            H =  HM**(1-par.alpha)*HF**par.alpha
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def ratio(self):

        sol = self.sol
        par = self.par

        sol.HM_wage_vec = []
        sol.HF_wage_vec = []
        wF = (0.8, 0.9, 1.0, 1.1, 1.2)


        # b. for loop
        for wages in wF:
            par.wF = wages
            _, _, HM, HF = self.solve_con()
            sol.HM_wage_vec.append(HM)
            sol.HF_wage_vec.append(HF)
            
        #c. extracting results
        #HF_wage_vec = [ns[3] for ns in sol.solution_wage]
        #HM_wage_vec = [ns[2] for ns in sol.solution_wage]

        H_ratio = [np.log(HF_ny/HM_ny) for HF_ny, HM_ny in zip(sol.HF_wage_vec, sol.HM_wage_vec)]
        w_ratio = np.log(wF)  

        return H_ratio, w_ratio


    def solve_con(self,do_print=False):
        """ solve model continously """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()  

        def objective(x):
            return -self.calc_utility(*x)

        constraints = [{'type':'ineq','fun': lambda x: 24 - x[0] - x[1]},
                       {'type':'ineq','fun': lambda x: 24 - x[2] - x[2]}]
        bounds = [(1e-8,24-1e-8)]*4
        guess = [2*12/2]*4

        solution = optimize.minimize(objective,
                                   guess,
                                   method='SLSQP',
                                   bounds=bounds,
                                   constraints=constraints)

            
        LM = solution.x[0]
        HM = solution.x[1]
        LF = solution.x[2]
        HF = solution.x[3]
        
        return LM, LF, HM, HF 


    def solve_wF_vec(self,discrete=False):
        func = self.func

        par = self.par
        sol = self.sol

        wM = 1.0
        wF = (0.8, 0.9, 1.0, 1.1, 1.2)

        par.beta0_target = 0.4
        par.beta1_target = -0.1

        H_ratio, w_ratio = self.ratio()

        slope, intercept = stats.linregress(self.ratio.H_ratio,self.ratio.w_ratio)

        beta0 = intercept
        beta1 = slope

        reg = (par.beta0_target - beta0)**2 + (par.beta1_target - beta1)**2
        return reg
    
    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self):
        """ estimate alpha and sigma """
        par = self.par
        sol = self.sol

        def estimate_alpha_sigma(x):
            return -self.calc_utility(*x)
        
        bounds_beta = [(1e-8,24-1e-8)]*2
        guess_beta = (0.1, 0.2)

        result_beta = optimize.minimize(estimate_alpha_sigma,
                                        guess_beta,
                                        method='SLSQP',
                                        bounds=bounds_beta,)
    
        opt_alpha, opt_sigma = result_beta.x
        return opt_alpha, opt_sigma
    
