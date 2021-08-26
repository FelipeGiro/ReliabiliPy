import numpy as np
from scipy.stats import norm
from reliabpy.models.observation import Probability_of_Detection as PoD

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class _Base(object):
    def _global_init(self):
        self.store_results = True
        self.t, self.obs, self.action = 0, None, None

        self.force_detection =  False
        self.force_notdetection = False


    def _store_results(self):
        if self.store_results:
            self.results.append([
                self.t, 
                self.pf,
                self.obs,
                self.action
                ])

    def get_pf(self):
        """
        Get P_f
        =======

        Get the probability of failure and its timestep.
        Needs model with <store_results = True>.

        Return:
        -------
        time : array
            time for the pf
        pf : array
            probability of failure
        """
        # Get the probability of failure. Needs <store_results = True>
        if not self.store_results:
            raise Warning("No stored probability of failure: store_results is", self.store_results)
        
        results = np.array(self.results)
        time, pf = results[:,0], results[:,1]

        return time, pf
    
    def get_results(self):
        """
        Get results
        ===========
        
        Return a dictionary with the results for the entire episode: 
        - year   : year
        - pf     : probability of failure
        - obs    : observation
        - action : action on a component

        Return:
        -------
        results : dict
            year, pf, obs, action
        """

        if not self.store_results:
            raise Warning("No stored probability of failure: store_results is", self.store_results)

        results = np.array(self.results)

        return {"year" : results[:,0], "pf" : results[:,1], "obs" : results[:,2], "action" : results[:,3]}

class MonteCarloSimulation(_Base):
    """
    Monte Carlo Simulation
    ======================

    Parameters:
    -----------
    a_0 : array
        initial crack size of the samples
    function : function
        funtion to propagate over time with <a> only as an input 
    a_crit : float
        critical crack depth
    """
    def __init__(self, a_0, function, a_crit):
        self._global_init()

        self.a_0 = a_0 
        self.a = a_0.copy()
        self.f = function
        self.a_crit = a_crit

        self.num_samples = len(self.a_0)

        self.pf = self.get_prob_fail()

        self.t = 0
        self.PoD = np.ones_like(a_0)

        if self.store_results:
            self.results = [[
                self.t, 
                self.pf,
                None,
                None]]

    def predict(self):
        """
        Predict
        =======
        
        Propagate one time step.
        """
        self.a = self.f()
        self.t += 1

        if self.store_results: self._store_results()

    def update(self, parameters):
        """
        Update
        ======

        Update the current state with (so far) Probability of Detection inspection model.

        TODO: check this restriction for only PoD

        Parameters:
        -----------
        parameters : dict
            inspection parameters. so far, with only "quality" key (good, normal and bad)
        """
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
        """
        Get probability of failure
        ==========================

        Get the probability of failure for the current timestep

        Returns
        -------
        pf : float
            current probability of failure
        """
        failed_samples = self.a > self.a_crit

        total = self.PoD.sum()
        failed_detected = np.sum(failed_samples*self.PoD)

        return failed_detected/total

class TransitionMatrix:
    # TODO: Transition matrix class
    """
    Transtion Matrix
    ================
    
    Build the transtion matrix for a given function and discretization scheme.

    Parameters:
    -----------
    discretization : array
        initial state vector (& x n)
    function : function
        propagation function over time: f(a)
    """
    def __init__(self, discretization, function):
        self.discretization = discretization
        self.function = function

    def build_T(self):
        """
        Build T
        =======

        Build transtiion matrix.

        Returns:
        --------
        T : matrix
            transtion matrix
        """
        T = None
        return T

class DynamicBayesianNetwork(_Base):
    '''
    Component level dynamic Bayesian network
    ========================================

    Dynamic Bayesian nework for one component.

    Parameters:
    -----------
    T : matrix
        (n x n) transtiion matrix
    s0 : array
        initial state vector (1 x n)
    discretization : dict
        discretization of all variables (e.g.: crack depth and time) 
    '''
    
    def __init__(self, T, s0, discretizations):
        self._global_init()

        self.T = tf.convert_to_tensor(T)
        self.s = tf.convert_to_tensor(s0)
        self.s0 = tf.convert_to_tensor(s0.copy())
        self.discretizations = discretizations

        self.states_values = np.diff(discretizations['a']/2) + discretizations['a'][:-1]

        self.n_states = {}
        self.total_nstates = 1
        for var_name in discretizations:
            quant = int(len(discretizations[var_name]) - 1)
            self.n_states[var_name] = quant
            self.total_nstates *= quant
        
        self.pf = self.get_prob_fail()

        if self.store_results:
            self.results = [[
                self.t, 
                self.pf,
                self.obs,
                self.action]]

    def predict(self):
        """
        Predict
        =======
        
        Propagate one time step.

        TODO: implement for benchmark:
        - Numba
        - PyTorch
        - TensorFlow
        """
        self.t += 1
        self.s = tf.matmul(self.s, self.T)
        self.obs, self.action = None, None
        self.pf = self.get_prob_fail()
        
        if self.store_results: self._store_results()
    
    def update(self, insp_quality):
        """
        Update
        ======

        Update the current state with (so far) Probability of Detection inspection model.

        TODO: check this restriction for only PoD

        Parameters:
        -----------
        parameters : dict
            inspection parameters. so far, with only "quality" key (good, normal and bad)
        """
        self.model = 'PoD'

        parameters, function = PoD.get_settings(insp_quality)
        obs_pmf = function(self.states_values, **parameters)

        obs_state = np.tile(obs_pmf, int(self.total_nstates/len(obs_pmf)))
        obs_state = tf.convert_to_tensor(obs_state)
        detected_pmf = self.s*obs_state

        self.crack_detected = bool(np.random.binomial(1, tf.reduce_sum(detected_pmf).numpy()))
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

        self.s /= tf.reduce_sum(self.s)
        self.obs, self.action = None, None
        self.pf = self.get_prob_fail()
        
        if self.store_results: self._store_results()
    
    def perform_action(self):
        """
        Perform action
        ==============

        Perform perfect repair action.
        """
        if self.crack_detected:
            self.action = 'perfect_repair'
            self.s = self.s0
            self.pf = self.get_prob_fail()
        if self.store_results: self._store_results()
        
    def get_prob_fail(self):
        """
        Get probability of failure
        ==========================

        Get the probability of failure for the current timestep

        Returns
        -------
        pf : float
            current probability of failure
        """
        s = self._reorder()
        return tf.reduce_sum(s, 0).numpy()[-1]

    def _reorder(self):
        return tf.reshape(self.s, list(self.n_states.values()))
    
    def _discretize(discretization, dist_params, function):
        dist_params['a'] = discretization
        return np.diff(function(**dist_params))/np.diff(discretization)

class metrics:
    """
    Metrics
    =======
    
    Metric functions for performance comparisons.
    """
    def pf_rmse(model_1, model_2):
        """
        RMSE for P_f
        ============

        Root mean square error for the probability of failure of two models

        Parameters:
        -----------
        model_1 : inference model
            inference model
        model_2 : inference model
            inference model

        Returns:
        --------
        rmse : float
            Root mean square
        """
        pf1 = model_1.get_pf()[1]
        pf2 = model_2.get_pf()[1]

        return np.sqrt(((pf1 - pf2)**2.0).mean())