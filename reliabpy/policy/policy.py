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
    def __init__(self, system_model, delta_t, nI, to_avoid=[8,9,10,11], last_year_action=False): # TODO: put to_avoid None
        self.to_avoid = to_avoid
        self.system_model = system_model
        self.delta_t, self.nI = delta_t, nI
        self.to_avoid = to_avoid
        self.last_year_action = last_year_action

    def to_observe(self):
        to_inspect = []
        pf_list = np.array([x['pf'] for x in self.system_model.step_results.values()])
        if self.to_avoid is not None:
            pf_list[self.to_avoid] = -1
        if self.system_model.components_list[0].last_results['t'] % self.delta_t == 0:
            to_inspect = np.argpartition(pf_list, -int(self.nI))[-int(self.nI):]
        if ~self.last_year_action and self.system_model.t == self.system_model.lifetime:
            to_inspect = []
            
        return to_inspect
    
    def to_repair(self):
        return self.system_model.to_inspect
