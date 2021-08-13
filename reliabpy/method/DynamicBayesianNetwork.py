import numpy as np
from scipy.stats import norm
from reliabpy.method._common import Base

# TODO: documentation

def discretize(discretization, dist_params, scipy_stats_dist=norm):
    dist_params['x'] = discretization
    return np.diff(scipy_stats_dist.cdf(**dist_params))/np.diff(discretization)

class TransitionMatrix:
    def __init__(self):
        # TODO: function to build transition matrix
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

        self.n_states = {}
        for var_name in discretizations:
            self.n_states[var_name] = int(len(discretizations[var_name]) - 1)
        
        if self.store_pfs:
            self.pfs = [[
                self.t, 
                self.get_prob_fail()]]

    def predict(self):
        self.t += 1
        self.s = np.dot(self.s, self.T)
        
        if self.store_pfs: self._store_pfs()
    
    def update(self, obs=None, std=None):
        
        dist_params = {'loc':obs, 'scale':std}

        z = np.zeros_like(self.s)
        z_t = discretize(self.discretizations['a'], dist_params, scipy_stats_dist=norm)
        z[0, self.n_states['a']*self.t:self.n_states['a']*(self.t+1)] = z_t
        self.s = self.s*z
        self.s /= self.s.sum()
        
        if self.store_pfs: self._store_pfs()
        
    def get_prob_fail(self):
        s = self._reorder()
        return s.sum(axis=0)[-1]