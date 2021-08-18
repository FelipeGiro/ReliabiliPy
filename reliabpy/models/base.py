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
    def __init__(self, components_reliability_models_list, policy=None):
        
        # transforming to dataclass Component
        self.components_list = []
        for component in components_reliability_models_list:
            _temp_Component = Component(
                component, 
                dcopy(components_reliability_models_list[component]['inference'])
            )

            _temp_Component.store()

            self.components_list.append(_temp_Component)
        
        self.policy = policy
    
    def foward_one_timestep(self):
        for component in self.components_list:
            
            component.inference_model.predict()
            component.store()
            if self.policy is not None:
                print("Select the components with highest VoI")
                # TODO: update
  
            if self.policy is not None:
                print("Select components with detected crack to repair")
                # TODO: action

            if self.policy is not None:
                print("Repair as new")
    
    def get_results(self):
        system = list()
        for component in self.components_list:
            temp = component.get_results()
            system.append(temp)
        return system

    def run(self, lifetime):
        for timestep in range(lifetime):
            self.foward_one_timestep()
        

