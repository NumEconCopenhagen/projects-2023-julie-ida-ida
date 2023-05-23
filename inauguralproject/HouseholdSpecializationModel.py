
from types import SimpleNamespace

import numpy as np
from scipy import optimize
import scipy.stats as stats

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. Create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. Preferences
        par.rho = 2.0 
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. Household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. Wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. Targets 
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. Solution
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

        # a. Consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. Consumption of home production
        if par.sigma == 0:
            H = np.min(HM,HF)
        elif par.sigma == 1:
            H =  HM**(1-par.alpha)*HF**par.alpha
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # c. Total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho) #OBS! Hvorfor skriver man Q,1e-8 her? 

        # d. Disutlity of work
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
        
        # a. All possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. Calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. Set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) 
        u[I] = -np.inf
    
        # d. Find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. Print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def ratio(self):

        sol = self.sol
        par = self.par

        # a. Creating an empty list to store the values
        sol.HM_wage_vec = []
        sol.HF_wage_vec = []
        wF = (0.8, 0.9, 1.0, 1.1, 1.2)


        # b. For loop
        for wages in wF:
            par.wF = wages
            _, _, HM, HF = self.solve_con()
            sol.HM_wage_vec.append(HM)
            sol.HF_wage_vec.append(HF)

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
                                   method='Nelder-Mead',
                                   bounds=bounds,
                                   constraints=constraints,
                                   tol=1e-10)

        LM = solution.x[0]
        HM = solution.x[1]
        LF = solution.x[2]
        HF = solution.x[3]
        
        return LM, LF, HM, HF 

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """
       
        par = self.par
        sol = self.sol
        opt = SimpleNamespace() 

        for i,wF in enumerate(par.wF_vec):
            par.wF = wF

            def objective4(x):
                return -self.calc_utility(*x)

            bounds = [(1e-8,24-1e-8)]*4
            guess = [2*12/2]*4

            solution4 = optimize.minimize(objective4,
                                   guess,
                                   method='Nelder-Mead',
                                   bounds=bounds,
                                   tol=1e-10)

            maxHM = solution4.x[1]
            maxHF = solution4.x[3]

            sol.HM_vec[i] = solution4.x[1]
            sol.HF_vec[i] = solution4.x[3]

        return sol.HM_vec, sol.HF_vec
    
    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec) 
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

    def loss(self, x):
        """ estimate alpha and sigma """
         
        par = self.par
        sol = self.sol
        
        par.alpha = x[0]
        par.sigma = x[1]

        self.solve_wF_vec()
        self.run_regression()

        loss = (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2

        return loss

    def estimate(self, alpha=None, sigma=None):
        
        par = self.par
        sol = self.sol
        opt_as = SimpleNamespace() 
        
        bounds = [(1e-8,24-1e-8)]*2
        guess = [0.5, 1]

        solution4a = optimize.minimize(self.loss,
                                   guess,
                                   method='Nelder-Mead',
                                   bounds=bounds,
                                   tol=1e-10)
        
        opt_as.alpha4 = solution4a.x[0]
        opt_as.sigma4 = solution4a.x[1]
        
        print(f' optimal alpha = {opt_as.alpha4}')
        print(f' optimal sigma = {opt_as.sigma4}')

        return solution4a


###### QUESTION 5 ###### 
# We are making a copy of the code above, but now we have kappa = 0.8 and alpha fixed at 0.5 
class HouseholdSpecializationModelClass_5:
    
    def __init__(self):
        """ setup model """

        # a. Create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. Preferences
        par.rho = 2.0 
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 
        par.kappa = 0.8 

        # c. Household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. Wages
        par.wM = 1.0
        par.wF = 0.8
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. Targets 
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. Solution
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

        # a. Consumption of market goods including wage gap 
        C = par.wM*LM + par.kappa*par.wF*LF 

        # b. Consumption of home production
        if par.sigma == 0:
            H = np.min(HM,HF)
        elif par.sigma == 1:
            H =  HM**(1-par.alpha)*HF**par.alpha
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # c. Total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho) 

        # d. Disutlity of work
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
        
        # a. All possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. Calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. Set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) 
        u[I] = -np.inf
    
        # d. Find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. Print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def ratio(self):

        sol = self.sol
        par = self.par

        # a. Creating an empty list to store the values
        sol.HM_wage_vec = []
        sol.HF_wage_vec = []
        wF = (0.8, 0.9, 1.0, 1.1, 1.2)


        # b. For loop
        for wages in wF:
            par.wF = wages
            _, _, HM, HF = self.solve_con()
            sol.HM_wage_vec.append(HM)
            sol.HF_wage_vec.append(HF)

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
                                   method='Nelder-Mead',
                                   bounds=bounds,
                                   constraints=constraints,
                                   tol=1e-10)

        LM = solution.x[0]
        HM = solution.x[1]
        LF = solution.x[2]
        HF = solution.x[3]
        
        return LM, LF, HM, HF 

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """
       
        par = self.par
        sol = self.sol
        opt = SimpleNamespace() 

        for i,wF in enumerate(par.wF_vec):
            par.wF = wF

            def objective4(x):
                return -self.calc_utility(*x)

            bounds = [(1e-8,24-1e-8)]*4
            guess = [2*12/2]*4

            solution4 = optimize.minimize(objective4,
                                   guess,
                                   method='Nelder-Mead',
                                   bounds=bounds,
                                   tol=1e-10)

            maxHM = solution4.x[1]
            maxHF = solution4.x[3]

            sol.HM_vec[i] = solution4.x[1]
            sol.HF_vec[i] = solution4.x[3]

        return sol.HM_vec, sol.HF_vec
    
    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec) 
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

    def loss(self, x):
        """ estimate alpha and kappa """ 
         
        par = self.par
        sol = self.sol
        
        par.alpha = 0.5
        par.kappa = 0.8 

        self.solve_wF_vec()
        self.run_regression()

        loss = (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2

        return loss

    def estimate(self, alpha=None, kappa=None):
        
        par = self.par
        sol = self.sol
        opt_as = SimpleNamespace() 
        
        bounds = [(1e-8,24-1e-8)]*2
        guess = [0.5, 0.8]

        solution5 = optimize.minimize(self.loss,
                                   guess,
                                   method='Nelder-Mead',
                                   bounds=bounds,
                                   tol=1e-10)
        
        opt_as.alpha4 = solution5.x[0]
        opt_as.kappa4 = solution5.x[1]
        
        print(f' optimal alpha = {opt_as.alpha4}')
        print(f' optimal kappa = {opt_as.kappa4}')

        return solution5
