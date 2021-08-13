import numpy as np 
from reliabpy.method._common import Base

# TODO: documentation

class foward_propagation(Base):
    def initialize(self, a_0, function, a_crit):
        self.a_0 = a_0 
        self.a = a_0.copy()
        self.f = function
        self.a_crit = a_crit

        self.t = 0

        if self.store_pfs:
            self.pfs = [[
                self.t, 
                self.get_prob_fail()]]

    def predict(self):
        self.a = self.f()
        self.t += 1

        if self.store_pfs: self._store_pfs()

    def update(self):
        pass

        if self.store_pfs: self._store_pfs()

    def get_prob_fail(self):
        mask = self.a > self.a_crit

        total = len(self.a)
        failed = np.sum(mask)

        return failed/total
     

