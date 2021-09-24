import numpy as np

def select_highest_VoI(pf_array, n):
    return pf_array.argsort()[-n:][::-1]

# TODO: implemnet user defined policy
class UserDefined:
    def __init__(self, inspection_map):
        self.inspection_map = inspection_map
    
    def import_model(self, system_model):
        self.system_model = system_model

    def to_observe(self):
        t = int(self.system_model.components_list[0].last_results['t'])
        if t > 20:
            return list()
        components_to_insp = self.inspection_map.loc[t].dropna().index
        to_inspect = list()
        for component in components_to_insp:
            to_inspect.append(self.inspection_map.columns.get_loc(component))
        return to_inspect
    
    def to_repair(self):
        output = self.system_model.get_step_results('output', dtype='list')
        detected = np.array(output) == 'D'
        ids = [i for i, x in enumerate(detected) if x]
        return ids

class DoNothing:
    def __init__(self, system_model):
        pass

    def to_observe(self):
        pass
    
    def to_repair(self):
        pass

class HeuristicRules:
    def __init__(self, delta_t, nI, to_avoid=None, last_year_action=False):
        self.delta_t, self.nI = delta_t, nI
        self.to_avoid = to_avoid
        self.last_year_action = last_year_action

    def import_model(self, system_model):
        self.system_model = system_model

    def to_observe(self):
        to_inspect = []
        pf_list = np.array([x['pf'] for x in self.system_model.step_results.values()])
        t = int(self.system_model.components_list[0].last_results['t'])
        if self.to_avoid is not None:
            pf_list[self.to_avoid] = -1
        if t % self.delta_t == 0:
            to_inspect = np.argpartition(pf_list, -int(self.nI))[-int(self.nI):]
        if ~self.last_year_action and self.system_model.t == self.system_model.lifetime:
            to_inspect = []
            
        return to_inspect
    
    def to_repair(self):
        output = self.system_model.get_step_results('output', dtype='list')
        detected = np.array(output) == 'D'
        ids = [i for i, x in enumerate(detected) if x]
        return ids
