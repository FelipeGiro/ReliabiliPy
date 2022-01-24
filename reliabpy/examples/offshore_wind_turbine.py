from reliabpy.models.base import SystemLevel
from reliabpy.models.inference import DynamicBayesianNetwork 
from reliabpy.policy.policy import HeuristicRules, UserDefined
from reliabpy.models.system_effects import System_of_Subsystems
from reliabpy.models.cost import InspectionMaintenance
from reliabpy.readwrite.ANAST import import_DBN_input_data
# from reliabpy.policy import optimization

import os

class Simple:
    """
    Simple model
    ============

    Simple model for an offshore wind turbine with 12 component equally
    distributed into 3 deterioration zones: atmospheric, submerged, and 
    burried. 
    
    Models used:
    - Inference: dynamic Bayesian network.
    - Observation: probability of inspection.
    - Action : perfect repair.
    - System dependencies : series system of k-out-of-n subsystems.
    - Cost model: for inspetion and maintenance.
    """
    def __init__(self, input_folder="data\\transition_matrices", output_folder="data"):
        self.input_folder = input_folder
        self.output_folder = output_folder

    def mount_model(self, 
                    zone_k = {"atm":3, "sub":3, "bur":3}, 
                    policy_rules = HeuristicRules(delta_t = 4, nI = 6, to_avoid=[8,9,10,11])):
        """
        Mount model
        ===========
        
        Mount the entire model and submodels (with inputs parameters already setup).

        Returns:
        --------
        model : Initialized system model
            System model for the offshroe wind turbine.
        """
        # TODO: change when we have transittion matrix
        atmosphetic_zone_inputs = import_DBN_input_data(os.path.join(self.input_folder, "atm\\dr_OUT.mat"))
        submerged_zone__inputs = import_DBN_input_data(os.path.join(self.input_folder, "sub\\dr_OUT.mat"))
        buried_zone_inputs = import_DBN_input_data(os.path.join(self.input_folder, "bur\\dr_OUT.mat"))

        atmosphetic_zone_model = DynamicBayesianNetwork(*atmosphetic_zone_inputs)
        submerged_zone_model = DynamicBayesianNetwork(*submerged_zone__inputs)
        buried_zone_model = DynamicBayesianNetwork(*buried_zone_inputs)

        atmosphetic_repair = DynamicBayesianNetwork(*atmosphetic_zone_inputs)
        submerged_repair = DynamicBayesianNetwork(*submerged_zone__inputs)

        components_reliability_models_list = {
            'atm1' : {
                'inference' : atmosphetic_zone_model,
                'inspection': 'normal', 
                'repair'    : atmosphetic_repair},
            'atm2' : {
                'inference' : atmosphetic_zone_model,
                'inspection': 'normal', 
                'repair'    : atmosphetic_repair},
            'atm3' : {
                'inference' : atmosphetic_zone_model,
                'inspection': 'normal', 
                'repair'    : atmosphetic_repair},
            'atm4' : {
                'inference' : atmosphetic_zone_model,
                'inspection': 'normal', 
                'repair'    : atmosphetic_repair},
            'sub1' : {
                'inference' : submerged_zone_model,
                'inspection': 'bad', 
                'repair'    : submerged_repair},
            'sub2' : {
                'inference' : submerged_zone_model,
                'inspection': 'bad', 
                'repair'    : submerged_repair},
            'sub3' : {
                'inference' : submerged_zone_model,
                'inspection': 'bad', 
                'repair'    : submerged_repair},
            'sub4' : {
                'inference' : submerged_zone_model,
                'inspection': 'bad', 
                'repair'    : submerged_repair},
            'bur1' : {
                'inference' : buried_zone_model,
                'inspection': None, 
                'repair'    : None},
            'bur2' : {
                'inference' : buried_zone_model,
                'inspection': None, 
                'repair'    : None},
            'bur3' : {
                'inference' : buried_zone_model,
                'inspection': None, 
                'repair'    : None},
            'bur4' : {
                'inference' : buried_zone_model,
                'inspection': None, 
                'repair'    : None}
        }

        zone_assingment = ['atm', 'atm', 'atm', 'atm', 'sub', 'sub', 'sub', 'sub', 'bur', 'bur', 'bur', 'bur']

        self.monopile = SystemLevel(
            components_reliability_models_list, 
            policy_rules = policy_rules,
            system_dependancies = System_of_Subsystems(zone_assingment, zone_k),
            cost_model = InspectionMaintenance(c_c=5.0, c_i=1.0, c_r=10.0, c_f=10000, r=0.02)
        )

        return self.monopile

    def run_one_episode(self):
        """
        Run one episode
        ===============

        Run model for a lifetime of 20 years.

        Return:
        -------
        cost_breakdown : dict
            Dicionary with the cost breakdown:
                C_C : campain costs
                C_I : inspection costs
                C_R : repair costs
                R_F : risk of failure
        """
        self.monopile.run(lifetime=20)
        return self.monopile.cost_breakdown
    
    def save_results(self, savefolder = None):
        if savefolder is not None:
            self.monopile.post_process(savefolder)

class _Simple_ComponentLevel(Simple):
    def mount_model(self, component):
        """
        Mount model in component level
        ==============================

        Just for study case
        """
        # TODO: change when we have transittion matrix
        zone_inputs = import_DBN_input_data(os.path.join(self.input_folder, component + "\\dr_OUT.mat"))

        zone_model = DynamicBayesianNetwork(*zone_inputs)

        repair = DynamicBayesianNetwork(*zone_inputs)

        component_reliability_models_list = {
            component + '1' : {
                'inference' : zone_model,
                'inspection': 'normal', 
                'repair'    : repair}
        }

        self.monopile = SystemLevel(
            component_reliability_models_list, 
            policy_rules = HeuristicRules(delta_t = 6, nI = 5, to_avoid=None),
            system_dependancies = System_of_Subsystems([component], {component:1}),
            cost_model = InspectionMaintenance(c_c=5.0, c_i=1.0, c_r=10.0, c_f=834, r=0.02)
        )
    
    def run_one_episode(self):
        self.monopile.run(lifetime=20)

        return self.monopile.cost_breakdown

if __name__ == '__main__':
    # import pandas as pd
    # df_inspmap = pd.read_excel("C:\\Developments\\reliabpy\\PhD\\OWT_12comp_3detzones\\ComponentLevel\\InspectionMap.xlsx",
    #                             index_col=0)
    zone_k = {'atm':3, 'sub':3, 'bur':3}

    model = Simple()
    model.mount_model(zone_k)
    model.run_one_episode()
    model.save_results('data')

    # model = _Simple_ComponentLevel()
    # model.mount_model('atm')
    # model.run_one_episode()
    # model.save_results('data')
