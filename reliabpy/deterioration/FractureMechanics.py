"""
DETERIORATION MODEL
===================

Script organized in class with the deterioration models to be used on the 
Kalman Filters.
"""
import numpy as np
from reliabpy.commons.normal_relations import logN2N

class GeometricFactor:
    '''
    Geometric Factor
    ================
    '''
    def __init__(self):
        pass
    
    def lognormal(mean=1.0, std=0.1, n_samples=None):
        '''
        TODO: describe the source of it

        Params:
        -------
        mean : float
            mean of lognormal distribution
        std : float
            standart deviation of lognormal distribution
        n_samples : int
            number of samples to be generated
        
        Return:
        =======
        Y_g : float or array of floats
            geometric factor
        '''

        var=std**2
        muYg = np.log((mean**2)/np.sqrt(var+mean**2))
        sigmaYg = np.sqrt(np.log(var/(mean**2) + 1))

        Y_g = np.random.lognormal(muYg, sigmaYg, size=n_samples)

        return Y_g
    

class Paris_Erdogan:
    '''
    Paris-Erdogan law
    =================
    
    Linear elastic fracture mechanics (LEFM).
    
    Parameters
    ----------
    a_0 : float or array of floats 
        initial crack size
    m : float or array of floats
        crack growth parameter
    n : float or array of floats
        number of fatigue cycles
    C : float or array of floats
        crack growth parameter
    S : float or array of floats
        stress range
    Y_g : float or array of floats
        geometric factor
    '''
    def __init__(self):
        pass

    def initialize(self, a_0, m, n, C, S, Y_g):
        self.a_0, self.a, self.m, self.n, self.C, self.S, self.Y_g = a_0, a_0, m, n, C, S, Y_g

        self.bad_values_counts = {}

    def _filter_values(self, a):
        bad_values = np.full_like(a, False, dtype=bool)
        isnan = np.isnan(a)
        decrasing = (a - self.a) < 0 
        bad_values[isnan | decrasing] = True

        # TODO: count and register bad values count. Must deal with arrays and floats

        if type(a) is np.ndarray:
            a[bad_values] = np.inf
            Warning("Unstable values")
        elif bad_values:
            a = np.inf
            Warning("Unstable values")
        
        self.a = a


    def propagate(self):
        '''
        Propagate
        =========

        Progagate one time step.

        Return
        ------
        a : float or array of floats
            crack size
        '''
        a, m, n, C, S, Y_g = self.a, self.m, self.n, self.C, self.S, self.Y_g

        a = (a**((2-m)/2) + ((2-m)/2)*C*(Y_g*np.pi**0.5*S)**m*n)**(2./(2-m))
        self._filter_values(a)
        return self.a

    @staticmethod
    def run_example():
        '''
        Run example
        ===========

        A simple example.
        '''
        a_0, m, n_cycles, C, S, Y_g = 2.0, 3.0, 5049216, 1.00e-12, 12.0, 1.0
        print('=== PARIS ERDOGAN LAW EXAMPLE ===')
        print('- Input variables')
        print('   a_0      :', a_0)
        print('   m        :', m)
        print('   n_cycles :', n_cycles)
        print('   C        :', C)
        print('   S        :', S)
        print('   Y_g      :', Y_g)

        fm = Paris_Erdogan()
        fm.initialize(a_0, m, n_cycles, C, S, Y_g)
        
        print('- Crack propagation')
        print('   time step:', 0, '| creck depth:', a_0)
        for t in range(1, 6):
            print('   time step:', t, '| creck depth:', fm.propagate())
        



            