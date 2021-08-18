import numpy as np
from reliabpy.models import inference
from copy import deepcopy as dcopy
from tabulate import tabulate

class Component:
    def __init__(self, id, inference_model):
        self.id = id
        self.inference_model = inference_model
        self.t, self.pf, self.obs, self.action = list(), list(), list(), list()

    def store(self):
        t      = self.inference_model.t
        pf     = self.inference_model.get_prob_fail()
        obs    = None
        action = None

        self.last_results =  {"t" : t, "pf" : pf, "obs" : obs, "action" : action}

        self.t.append(t) 
        self.pf.append(pf) 
        self.obs.append(obs) 
        self.action.append(action) 

    def __str__(self):
        datatype = "DataType: StructuralComponent\n===============================\n"
        comp_name = f"- Component name: <<{self.name}>>\n"
        table = "- Results table\n" + tabulate(
            {"time" : self.t, "pf" : self.pf, "obs" : self.obs, "action" : self.action},
            headers = "keys", tablefmt="pretty")
        return datatype + comp_name + table
    
    def get_results(self, dtype="dict"):
        if dtype == "dict":
            results = {"id" : self.id, "time" : self.t, "pf" : self.pf, "obs" : self.obs, "action" : self.action}
        return results

class SystemModel:
    def __init__(self, components_reliability_models_list):
        
        # transforming to dataclass Component
        self.components_list = []
        for component in components_reliability_models_list:
            _temp_Component = Component(
                component, 
                dcopy(components_reliability_models_list[component]['inference'])
            )

            _temp_Component.store()

            self.components_list.append(_temp_Component)
    
    def foward_one_timestep(self):
        for component in self.components_list:
            
            component.inference_model.predict()
            component.store()
            
            # TODO: update 

            # TODO: action

            print(component.inference_model.t, component.inference_model.get_prob_fail())
    
    def get_results(self):
        system = list()
        for component in self.components_list:
            temp = component.get_results()
            system.append(temp)
        return system

    def run(self, lifetime):
        for timestep in range(lifetime):
            self.foward_one_timestep()
        

if __name__ == '__main__':

    from reliabpy.readwrite.ANAST import import_DBN_input_data

    atmosphetic_zone_inputs = import_DBN_input_data("C:\\Developments\\reliabpy\PhD\\transition_matrices\\atm\\dr_OUT.mat")
    submerged_zone__inputs = import_DBN_input_data("C:\\Developments\\reliabpy\PhD\\transition_matrices\\sub\\dr_OUT.mat")
    buried_zone_inputs = import_DBN_input_data("C:\\Developments\\reliabpy\PhD\\transition_matrices\\bur\\dr_OUT.mat")

    atmosphetic_zone_model = inference.DynamicBayesianNetwork(*atmosphetic_zone_inputs)
    submerged_zone_model = inference.DynamicBayesianNetwork(*submerged_zone__inputs)
    buried_zone_model = inference.DynamicBayesianNetwork(*buried_zone_inputs)

    components_reliability_models_list = {
        'atm1' : {
            'inference' : atmosphetic_zone_model},
        'atm2' : {
            'inference' : atmosphetic_zone_model},
        'atm3' : {
            'inference' : atmosphetic_zone_model},
        'atm4' : {
            'inference' : atmosphetic_zone_model},
        'sub1' : {
            'inference' : submerged_zone_model},
        'sub2' : {
            'inference' : submerged_zone_model},
        'sub3' : {
            'inference' : submerged_zone_model},
        'sub4' : {
            'inference' : submerged_zone_model},
        'bur1' : {
            'inference' : buried_zone_model},
        'bur2' : {
            'inference' : buried_zone_model},
        'bur3' : {
            'inference' : buried_zone_model},
        'bur4' : {
            'inference' : buried_zone_model}
    }
    

    monopile = SystemModel(components_reliability_models_list)
    monopile.run(lifetime=20)
    results = monopile.get_results()

    print()

