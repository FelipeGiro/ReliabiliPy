from reliabpy.models.base import SystemLevel
from reliabpy.models.inference import DynamicBayesianNetwork 
from reliabpy.policy.policy import HeuristicRules
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
    def __init__(self, input_folder="C:\\Developments\\reliabpy\PhD\\transition_matrices", output_folder="C:\\Developments\\reliabpy\\PhD\examples"):
        self.input_folder = input_folder
        self.output_folder = output_folder

    def mount_model(self):
        """
        Mount model
        ===========
        
        Mount the entire model and submodels (with inputs parameters already setup).

        Returns:
        --------
        model : Initialized system model
            System model for the offshroe wind turbine.
        """
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
        zone_k = [3,3,3]

        self.monopile = SystemLevel(
            components_reliability_models_list, 
            policy_rules = HeuristicRules(delta_t = 50, nI = 6),
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
    @staticmethod
    def optimize_heuristic_based_policy(self, n_samples= 10000):
        # TODO or to exclude: policy optimization
        n_samples

    
if __name__ == '__main__':
    model = Simple()
    model.mount_model()
    model.run_one_episode()
    model.save_results('C:\\Developments\\reliabpy\\PhD\\examples')
