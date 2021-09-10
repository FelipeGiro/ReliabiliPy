from copy import deepcopy as dcopy
from tabulate import tabulate
from reliabpy.commons.post_processing import OneEpisode

import numpy as np

class ComponentLevel:
    """
    Component Level
    ===============
    
    Class for reliability computation of one structural component only.

    Parameters:
    -----------
    id : str
        Identifier of the component
    inference_model : intializated inference model class 
        Initialized inferenece model (e.g.: Monte Carlo simulation or 
        dynamic Bayesian network)
    inspection : initilizated inspection model class
        Initialized inspection model (e.g.: probability of detection)
    """
    def __init__(self, id, inference_model, inspection):
        self.id = id
        self.inference_model = inference_model
        self.inspection = inspection
        self.t, self.pf, self.action, self.output = list(), list(), list(), list()

    def store(self):
        """
        Store (component level)
        =======================
        
        Store the time, probabiility of failure, observation results, and action for current timestep.

        Parameters:
        -----------
        action : Any
            Information about action on the component observation.
        output : Any
            Information about the component observation.
        
        Returns:
        --------
        t : int
            Current timestep.
        pf : float
            Current probability of failure (for timestep t).
        obs : any.
            Current observation (for timestep t).
        action : any
            Curretn action (for timestep t).

        """
        t      = self.inference_model.t
        pf     = self.inference_model.pf
        action = self.inference_model.action
        output = self.inference_model.output
        
        self.last_results =  {"t" : t, "pf" : pf, "action" : action, "output" : output}

        self.t.append(t) 
        self.pf.append(pf) 
        self.action.append(action) 
        self.output.append(output) 

        return t, pf, output, action
    
    def predict(self, store=True):
        """
        Predict
        =======

        Predict the state for the next step. Also know as transtion, 
        propagation, etc.

        Parameters:
        -----------
        store : Boolean
            store and returns the values of current state.
        """
        self.inference_model.predict() 
        if store: self.store()

    def update(self, store=True):
        """
        Update
        =======

        Update state face a pre-defined inspection technique.

        Parameters:
        -----------
        store : Boolean
            store and returns the values of current state.
        """
        self.inference_model.update(self.inspection) 
        if store: self.store()

    def perform_action(self, store=True):
        """
        Perform action
        ==============

        Perform a action in the component (inspections, repair, etc).

        Parameters:
        -----------
        store : Boolean
            store and returns the values of current state.
        """
        self.inference_model.perform_action() 
        if store: self.store()

    def get_results(self, dtype="dict", to_print=False):
        """
        Get results (component level)
        =============================

        Get the results for the components lifetime.
        """
        if to_print:
            self.t = np.array(self.t, dtype=int)
        if dtype == "dict":
            results = {"time" : self.t, "pf" : self.pf, "action" : self.action, "output" : self.output}
        return results

    def __str__(self):
        title = f"Structural Component: {self.id}\n"
        table = tabulate(self.get_results(to_print=True), headers="keys", tablefmt="pretty")
        return title + table

class SystemLevel:
    """
    System Level
    ============

    Class for reliability computation of the entire structural system.

    Parameters:
    -----------
    components_reliability_models_list : list of initilizated Component Level class
        List of all structural components models
    policy_rules : policy rules initilizated class
        Policy rules class 
    system_dependancies : system dependency initilizated class
        Dependancies model class
    cost_model : cost model initilized class
        Cost model class
    """
    def __init__(self, components_reliability_models_list, 
                 policy_rules, 
                 system_dependancies = None,
                 cost_model = None):
        
        self.components_reliability_models_list = components_reliability_models_list
        self.system_dependancies = system_dependancies
        self.cost_model = cost_model
        self.policy_rules = policy_rules
        self.policy_rules.import_model(self)

        self._reset()
    
    def _reset(self):
        # set all value as initial
        self.step_results = {}
        self.components_list = []
        self.t = 0
        for component in self.components_reliability_models_list:
            _temp_Component = ComponentLevel(
                component, 
                dcopy(self.components_reliability_models_list[component]['inference']),
                self.components_reliability_models_list[component]['inspection']
            )
            _temp_Component.store()

            self.components_list.append(_temp_Component)
            self.step_results[_temp_Component.id] = _temp_Component.last_results
        
        self.system_pf = None

        self._system_reliability()
    
    
    def forward_one_timestep(self):
        """
        Foward one timestep (system level)
        ==================================
        
        Advance one time step for the system (all components). 
        It included the update and the actions.
        """
        self.t += 1
        ### for every component ###
        # prediction
        for component in self.components_list:
            component.predict()
            self.step_results = self.get_step_results()
        self._system_reliability()
        
        if self.policy_rules is not None:
            # update
            self.to_inspect = self.policy_rules.to_observe()
            if len(self.to_inspect) is not 0:
                for i in self.to_inspect:
                    self.components_list[i].update()
                    self.step_results = self.get_step_results()
                self._system_reliability()

            # action
            self.to_repair = self.policy_rules.to_repair()
            if len(self.to_repair) is not 0:
                for i in self.to_repair:
                    self.components_list[i].perform_action()
                    self.step_results = self.get_step_results()
                self._system_reliability()

    def _system_reliability(self):
        step_results = list(self.step_results.values())
        pf_list = [x['pf'] for x in step_results]
        t__pf = (self.t, self.system_dependancies.compute_system_pf(pf_list))
        if self.system_pf is not None:
            self.system_pf.append(t__pf)
        else:
            self.system_pf = [t__pf]
         
    def get_step_results(self, variable_name=False, dtype='dict'):
        """
        Get step results
        ================

        Get the current state of every component in the system.

        Return:
        -------
        variable_name : str
            it can assume 't' for time step, 'pf' for the probability of 
            failure, 'action' for action taken, 'obs' for observation result 
        step_results : list
            List of Component Level current results.
        """
        if dtype == 'dict':
            step_results = dict()
            for component in self.components_list:
                temp = component.last_results
                if variable_name:
                    temp = temp[variable_name]
                step_results[component.id] = temp
            return step_results
        if (dtype == 'list') or (dtype == 'np.array'):
            step_results = list()
            for component in self.components_list:
                temp = component.last_results
                if variable_name:
                    temp = temp[variable_name]
                step_results.append(temp)
            if dtype == 'np.array':
                step_results = np.array(step_results)
            return step_results

    def get_components_results(self):
        """
        Get step results
        ================

        Get the current state of every component in the system.

        Return:
        -------
        step_results : list
            List of Component Level current results.
        system_pf : float
            system probability of failure
        """
        system = dict()
        for component in self.components_list:
            temp = component.get_results()
            system[component.id] = temp
        return system, self.system_pf
        
    def run(self, lifetime):
        """
        Run model
        =========

        Run model for a given lifetime.

        Parameters:
        -----------
        lifetime : int
            Simulation horizon of the structural system.
        """
        self.lifetime = lifetime
        for timestep in range(lifetime):
            self.forward_one_timestep()
        self.cost_model.compute_cost_breakdown(self)
    
    def post_process(self, savefolder, plot=True, excel=True):
        """
        Post-proccess
        =============
        
        Export results in graphs and tables.

        Parameters:
        -----------
        savefolder : path
            Path with the folder to save the results. 
        plot : bool
            Save graphical results in PNG.
        excel : bool
            Save table results in Excel spreadsheets.
        """
        post = OneEpisode(self, savefolder)
        if plot:
            post.plot_overview()
        if excel:
            post.to_excel()
        

