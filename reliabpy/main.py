from reliabpy.models.inference import DynamicBayesianNetwork 
from reliabpy.models.base import SystemModel
from reliabpy.readwrite.ANAST import import_DBN_input_data
from reliabpy.commons.visualization import plot_system

atmosphetic_zone_inputs = import_DBN_input_data("C:\\Developments\\reliabpy\PhD\\transition_matrices\\atm\\dr_OUT.mat")
submerged_zone__inputs = import_DBN_input_data("C:\\Developments\\reliabpy\PhD\\transition_matrices\\sub\\dr_OUT.mat")
buried_zone_inputs = import_DBN_input_data("C:\\Developments\\reliabpy\PhD\\transition_matrices\\bur\\dr_OUT.mat")

atmosphetic_zone_model = DynamicBayesianNetwork(*atmosphetic_zone_inputs)
submerged_zone_model = DynamicBayesianNetwork(*submerged_zone__inputs)
buried_zone_model = DynamicBayesianNetwork(*buried_zone_inputs)

components_reliability_models_list = {
    'atm1' : {
        'inference' : atmosphetic_zone_model,
        'inspection_parameters': 'normal'},
    'atm2' : {
        'inference' : atmosphetic_zone_model,
        'inspection_parameters': 'normal'},
    'atm3' : {
        'inference' : atmosphetic_zone_model},
        'inspection_parameters': 'normal',
    'atm4' : {
        'inference' : atmosphetic_zone_model,
        'inspection_parameters': 'normal'},
    'sub1' : {
        'inference' : submerged_zone_model,
        'inspection_parameters': 'bad'},
    'sub2' : {
        'inference' : submerged_zone_model,
        'inspection_parameters': 'bad'},
    'sub3' : {
        'inference' : submerged_zone_model,
        'inspection_parameters': 'bad'},
    'sub4' : {
        'inference' : submerged_zone_model,
        'inspection_parameters': 'bad'},
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
plot_system(monopile.get_results(), savefolder= 'C:\\Developments\\reliabpy\\PhD\examples\\system.png')

print('--- end ---')