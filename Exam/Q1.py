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

        if do_print: print('calling .setup()')
        self.setup()


    def setup(self):
        """ baseline parameters """

        par = self.par

        # a. Parameter values  
        par.alpha = 0.5
        par.kappa = 1.0 
        par.nu = 1/(2*16**2)
        par.w = 1.0
        par.tau = 0.30