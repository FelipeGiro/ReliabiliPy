import numpy as np

def select_highest_VoI(pf_array, n):
    return pf_array.argsort()[-n:][::-1]

# TODO: implemnet user defined policy
class UserDefined:
    def __init__(self, system_model):
        pass

    def to_observe(self):
        pass
    
    def to_repair(self):
        pass

class HeuristicRules:
    def __init__(self, system_model, delta_t, nI, to_avoid=[8,9,10,11]): # TODO: put to_avoid None
        self.to_avoid = to_avoid
        self.system_model = system_model
        self.delta_t, self.nI = delta_t, nI
        self.pf_list = np.array(system_model.components_last_results)[:, 1]
        if to_avoid is not None:
            self.pf_list[to_avoid] = -1

    def to_observe(self):
        to_inspect = []
        if self.system_model.components_list[0].last_results['t'] % self.delta_t == 0:
            to_inspect = np.argpartition(self.pf_list, -int(self.nI))[-int(self.nI):]
            
        return to_inspect
    
    def to_repair(self):
        pass
