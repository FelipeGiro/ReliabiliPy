import numpy as np

# TODO: documentation

class Base(object):
    def __init__(self, store_pfs=True):
        self.store_pfs = store_pfs
        self.t = 0

    def _store_pfs(self):
        if self.store_pfs:
            self.pfs.append([
                self.t, 
                self.get_prob_fail()])
    
    def get_pfs(self):
        if self.store_pfs:
            time_pfs = np.array(self.pfs)
            time, pfs = time_pfs[:,0], time_pfs[:,1]
        else:
            print("No stored probability of failure: store_pfs is", self.store_pfs)
        
        return time, pfs
