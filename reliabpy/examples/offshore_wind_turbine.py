from reliabpy.models.base import SystemModel
from reliabpy.models.inference import DynamicBayesianNetwork 
from reliabpy.models.system_effects import System_of_Subsystems
from reliabpy.models.cost import InspectionMaintenance
from reliabpy.readwrite.ANAST import import_DBN_input_data

import os

class Simple:
    def __init__(self, input_folder="C:\\Developments\\reliabpy\PhD\\transition_matrices\\atm", output_folder="C:\\Developments\\reliabpy\\PhD\examples"):
        self.input_folder = input_folder
        self.output_folder = output_folder

    def mount_model(self):
        atmosphetic_zone_inputs = import_DBN_input_data(os.path.join(self.input_folder, "dr_OUT.mat"))
        submerged_zone__inputs = import_DBN_input_data(os.path.join(self.input_folder, "dr_OUT.mat"))
        buried_zone_inputs = import_DBN_input_data(os.path.join(self.input_folder, "dr_OUT.mat"))

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

        self.monopile = SystemModel(
            components_reliability_models_list, 
            system_dependancies = System_of_Subsystems(zone_assingment, zone_k),
            cost_model = InspectionMaintenance(c_c=5.0, c_i=1.0, c_r=10.0, c_f=10000, r=0.02)
        )

    def run_one_episode(self):
        self.monopile.run(lifetime=20)
        self.monopile.post_process(self.output_folder)
    
if __name__ == '__main__':
    model = Simple()
    model.mount_model()
    model.run_one_episode()
