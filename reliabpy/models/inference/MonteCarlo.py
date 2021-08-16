import numpy as np 
from reliabpy.models.inference._common import Base
from reliabpy.models.observation import Probability_of_Detection as PoD

# TODO: documentation

class foward_propagation(Base):
    def initialize(self, a_0, function, a_crit):
        self.a_0 = a_0 
        self.a = a_0.copy()
        self.f = function
        self.a_crit = a_crit

        self.num_samples = len(self.a_0)

        self.t = 0
        self.PoD = np.ones_like(a_0)

        if self.store_results:
            self.pfs = [[
                self.t, 
                self.get_prob_fail()]]

    def predict(self):
        self.a = self.f()
        self.t += 1

        if self.store_results: self._store_results()

    def update(self, parameters):
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
     

