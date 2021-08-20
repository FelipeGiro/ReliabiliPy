from reliabpy.models.inference import DynamicBayesianNetwork 
from reliabpy.models.base import SystemModel
from reliabpy.readwrite.ANAST import import_DBN_input_data
from reliabpy.commons.visualization import plot_system

def run():
    atmosphetic_zone_inputs = import_DBN_input_data("C:\\Developments\\reliabpy\PhD\\transition_matrices\\atm\\dr_OUT.mat")
    submerged_zone__inputs = import_DBN_input_data("C:\\Developments\\reliabpy\PhD\\transition_matrices\\sub\\dr_OUT.mat")
    buried_zone_inputs = import_DBN_input_data("C:\\Developments\\reliabpy\PhD\\transition_matrices\\bur\\dr_OUT.mat")

    atmosphetic_zone_model = DynamicBayesianNetwork(*atmosphetic_zone_inputs)
    submerged_zone_model = DynamicBayesianNetwork(*submerged_zone__inputs)
    buried_zone_model = DynamicBayesianNetwork(*buried_zone_inputs)

    components_reliability_models_list = {
        'atm1' : {
            'inference' : atmosphetic_zone_model,
            'inspection': 'normal'},
        'atm2' : {
            'inference' : atmosphetic_zone_model,
            'inspection': 'normal'},
        'atm3' : {
            'inference' : atmosphetic_zone_model,
            'inspection': 'normal'},
        'atm4' : {
            'inference' : atmosphetic_zone_model,
            'inspection': 'normal'},
        'sub1' : {
            'inference' : submerged_zone_model,
            'inspection': 'bad'},
        'sub2' : {
            'inference' : submerged_zone_model,
            'inspection': 'bad'},
        'sub3' : {
            'inference' : submerged_zone_model,
            'inspection': 'bad'},
        'sub4' : {
            'inference' : submerged_zone_model,
            'inspection': 'bad'},
        'bur1' : {
            'inference' : buried_zone_model,
            'inspection': None},
        'bur2' : {
            'inference' : buried_zone_model,
            'inspection': None},
        'bur3' : {
            'inference' : buried_zone_model,
            'inspection': None},
        'bur4' : {
            'inference' : buried_zone_model,
            'inspection': None}
    }


    monopile = SystemModel(components_reliability_models_list)
    monopile.run(lifetime=20)
    plot_system(monopile.get_results(), savefolder= 'C:\\Developments\\reliabpy\\PhD\examples\\system.png')

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