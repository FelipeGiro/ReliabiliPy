from reliabpy.models.base import SystemModel
from reliabpy.models.inference import DynamicBayesianNetwork 
from reliabpy.readwrite.ANAST import import_DBN_input_data

def run():
    atmosphetic_zone_inputs = import_DBN_input_data("C:\\Developments\\reliabpy\PhD\\transition_matrices\\atm\\dr_OUT.mat")
    submerged_zone__inputs = import_DBN_input_data("C:\\Developments\\reliabpy\PhD\\transition_matrices\\sub\\dr_OUT.mat")
    buried_zone_inputs = import_DBN_input_data("C:\\Developments\\reliabpy\PhD\\transition_matrices\\bur\\dr_OUT.mat")

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


    monopile = SystemModel(components_reliability_models_list)
    monopile.run(lifetime=20)
    monopile.post_process('C:\\Developments\\reliabpy\\PhD\examples\\system.png')

    print('--- end ---')

if __name__ == '__main__':
    import cProfile, pstats
    import subprocess
    profiler = cProfile.Profile()
    profiler.enable()
    run()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.dump_stats('profile.dat')
    
    # subprocess.call(r"snakeviz profile.dat")