import numpy as np
from scipy.stats import norm
from reliabpy.models.inference._common import Base
from reliabpy.models.observation import Probability_of_Detection as PoD
# TODO: documentation

def _discretize(discretization, dist_params, function):
    dist_params['a'] = discretization
    return np.diff(function(**dist_params))/np.diff(discretization)

# TODO: Transition matrix class
# write a class to built the transiton matrix for DBN from a given function. 
class TransitionMatrix:
    def __init__(self):
        pass

    def initialize(self, discretization, function):
        pass

    def built_T(self):
        pass

class DeteriorationRate(Base):
    '''
    Component level dynamic Bayesian network
    ========================================
    '''

    def _reorder(self):
        return self.s.reshape(list(self.n_states.values()))
    
    
    def initialize(self, T, discretizations, s0):
        self.T = T
        self.s = s0
        self.discretizations = discretizations

        self.states_values = np.diff(discretizations['a']/2) + discretizations['a'][:-1]

        self.n_states = {}
        self.total_nstates = 1
        for var_name in discretizations:
            quant = int(len(discretizations[var_name]) - 1)
            self.n_states[var_name] = quant
            self.total_nstates *= quant
        
        if self.store_results:
            self.pfs = [[
                self.t, 
                self.get_prob_fail()]]

    def predict(self):
        self.t += 1
        self.s = np.dot(self.s, self.T)
        
        if self.store_results: self._store_results()
    
    def update(self, model, parameters):
        
        if model == 'normal':
            parameters, function = parameters, norm.cdf
            obs_pmf = _discretize(self.discretizations['a'], parameters, function)
        elif model == 'PoD':
            parameters, function = PoD.get_settings(parameters['quality'])
            obs_pmf = function(self.states_values, **parameters)

        obs_state = np.tile(obs_pmf, int(self.total_nstates/len(obs_pmf)))
        detected_pmf = self.s*obs_state

        self.crack_detected = bool(np.random.binomial(1, detected_pmf.sum()))
        if any([self.force_detection, self.force_notdetection]):
            if self.force_detection:
                self.s = detected_pmf
            if self.force_notdetection:
                self.s = self.s*(1 - obs_state)
        else:
            if self.crack_detected:
                self.s = detected_pmf
            else:
                self.s = self.s*(1 - obs_state)

        self.s /= self.s.sum()
        
        if self.store_results: self._store_results()
        
    def get_prob_fail(self):
        s = self._reorder()
        return s.sum(axis=0)[-1]