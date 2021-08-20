from copy import deepcopy as dcopy
from tabulate import tabulate
from reliabpy.policy.policy import HeuristicRules
from reliabpy.commons.post_processing import plot_system

import numpy as np

class Component:
    def __init__(self, id, inference_model, inspection):
        self.id = id
        self.inference_model = inference_model
        self.inspection = inspection
        self.t, self.pf, self.obs, self.action = list(), list(), list(), list()

    def store(self, obs = None, action = None):
        t      = self.inference_model.t
        pf     = self.inference_model.get_prob_fail() # TODO change to pf
        
        self.last_results =  {"t" : t, "pf" : pf, "obs" : obs, "action" : action}

        self.t.append(t) 
        self.pf.append(pf) 
        self.obs.append(obs) 
        self.action.append(action) 

        return t, pf, obs, action

    def get_results(self, dtype="dict"):
        if dtype == "dict":
            results = {"time" : self.t, "pf" : self.pf, "obs" : self.obs, "action" : self.action}
        return results

    def __str__(self):
        datatype = "DataType: StructuralComponent\n===============================\n"
        comp_name = f"- Component name: <<{self.name}>>\n"
        table = "- Results table\n" + tabulate(
            {"time" : self.t, "pf" : self.pf, "obs" : self.obs, "action" : self.action},
            headers = "keys", tablefmt="pretty")
        return datatype + comp_name + table

class SystemModel:
    def __init__(self, components_reliability_models_list, 
                 policy_rules=HeuristicRules, policy_parameters={"delta_t":5, "nI":3}, 
                 system_model = None,
                 costs= {'c_c' : 5.0, 'c_i' : 1.0, 'c_r' : 10.0, 'c_f' : 10000, 'r' : 0.02}):
        
        self.step_results = []
        self.components_list = []
        for component in components_reliability_models_list:

            _temp_Component = Component(
                component, 
                dcopy(components_reliability_models_list[component]['inference']),
                components_reliability_models_list[component]['inspection']
            )

            self.components_list.append(_temp_Component)

            self.step_results.append(_temp_Component.store())
        
        self.policy_rules = policy_rules(self, **policy_parameters)
    
    
    def foward_one_timestep(self):
        self.step_results = []
        for component in self.components_list:

            component.inference_model.predict()
            self.step_results.append(component.store())
        
        if self.policy_rules is not None:
            self.to_inspect = self.policy_rules.to_observe()
            if len(self.to_inspect) is not 0:
                for i in self.to_inspect:
                    component = self.components_list[i]
                    component.inference_model.update(component.inspection)
                    self.step_results.append(component.store())
            self.to_repair = self.policy_rules.to_repair()
            if len(self.to_repair) is not 0:
                for i in self.to_repair:
                    component = self.components_list[i]
                    component.inference_model.perform_action()
                    self.step_results.append(component.store())
    
    def system_reliability(self):
        # TODO: system reliability function
        pass

    def compute_costs(self):
        # TODO: cost function
        pass

    def get_results(self):
        system = dict()
        for component in self.components_list:
            temp = component.get_results()
            system[component.id] = temp
        return system
        

    def run(self, lifetime):
        for timestep in range(lifetime):
            self.foward_one_timestep()
    
    def post_process(self, savefolder):
        plot_system(self.get_results(), savefolder)
        

