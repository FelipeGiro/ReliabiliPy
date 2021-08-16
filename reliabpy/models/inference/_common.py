import numpy as np

# TODO: documentation

class Base(object):
    def __init__(self, store_results=True):
        self.store_results = store_results
        self.t = 0

        self.force_detection =  False
        self.force_notdetection = False

    def _store_results(self):
        if self.store_results:
            self.pfs.append([
                self.t, 
                self.get_prob_fail()])


    def get_pfs(self):
        if self.store_results:
            time_pfs = np.array(self.pfs)
            time, pfs = time_pfs[:,0], time_pfs[:,1]
        else:
            print("No stored probability of failure: store_results is", self.store_results)
        
        return time, pfs
