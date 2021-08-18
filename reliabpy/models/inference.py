import numpy as np
from scipy.stats import norm
from reliabpy.models.observation import Probability_of_Detection as PoD

# TODO: documentation

class _Base(object):
    def __init__(self, store_results=True):
        self.store_results = store_results
        self.t, self.obs, self.action = 0, None, None

        self.force_detection =  False
        self.force_notdetection = False

    def _store_results(self):
        if self.store_results:
            self.results.append([
                self.t, 
                self.get_prob_fail(),
                self.obs,
                self.action
                ])


    def get_pf(self):
        
        if not self.store_results:
            raise Warning("No stored probability of failure: store_results is", self.store_results)
        
        results = np.array(self.results)
        time, pfs = results[:,0], results[:,1]

        return time, pfs
    
    def get_results(self):

        if not self.store_results:
            raise Warning("No stored probability of failure: store_results is", self.store_results)

        results = np.array(self.results)

        return {"year" : results[:,0], "pf" : results[:,1], "obs" : results[:,2], "action" : results[:,3]}

        
        

class MonteCarloSimulation(_Base):
    def __init__(self, a_0, function, a_crit):
        self.a_0 = a_0 
        self.a = a_0.copy()
        self.f = function
        self.a_crit = a_crit

        self.num_samples = len(self.a_0)

        self.t = 0
        self.PoD = np.ones_like(a_0)

        if self.store_results:
            self.results = [[
                self.t, 
                self.get_prob_fail(),
                None,
                None]]

    def predict(self):
        self.a = self.f()
        self.t += 1

        if self.store_results: self._store_results()

    def update(self, parameters):
        self.obs = 'PoD'
        parameters, invPoD_func = PoD.get_settings(parameters['quality'], inverse=True)

        uniform_dist = np.random.uniform(size=self.num_samples)
        a_detected = invPoD_func(uniform_dist, **parameters)
        samples_detected = self.a > a_detected
        self.crack_detected = bool(np.random.binomial(1, samples_detected.sum()/self.num_samples))
        if any([self.force_detection, self.force_notdetection]):
            if self.force_detection:
                gH = samples_detected
            if self.force_notdetection:
                gH = self.a <= a_detected
        else:
            if self.crack_detected:
                gH = samples_detected
            else:
                gH = self.a <= a_detected
        self.PoD = self.PoD*gH

        if self.store_results: self._store_results()

    def get_prob_fail(self):
        failed_samples = self.a > self.a_crit

        total = self.PoD.sum()
        failed_detected = np.sum(failed_samples*self.PoD)

        return failed_detected/total

# TODO: Transition matrix class
# write a class to built the transiton matrix for DBN from a given function. 
class TransitionMatrix:
    def __init__(self):
        pass

    def initialize(self, discretization, function):
        pass

    def built_T(self):
        pass

class DynamicBayesianNetwork(_Base):
    '''
    Component level dynamic Bayesian network
    ========================================
    '''
    
    def __init__(self, T, discretizations, s0):
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
            self.results = [[
                self.t, 
                self.get_prob_fail(),
                None,
                None]]

    def predict(self):
        self.t += 1
        self.s = np.dot(self.s, self.T)
        
        if self.store_results: self._store_results()
    
    def update(self, parameters):
        self.model = 'PoD'

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

    def _reorder(self):
        return self.s.reshape(list(self.n_states.values()))
    
    def _discretize(discretization, dist_params, function):
        dist_params['a'] = discretization
        return np.diff(function(**dist_params))/np.diff(discretization)

class metrics:
    def pf_rmse(model_1, model_2):
        pf1 = model_1.get_pf()[1]
        pf2 = model_2.get_pf()[1]

        return np.sqrt(((pf1 - pf2)**2.0).mean())