# TODO: transform this file

import numpy as np
import pandas as pd
from SystemReliability import SysReliab
from time import time, sleep
from datetime import datetime, timedelta
import os
from glob import glob

import pprint

import pickle

def computeMCS(inputs, results_folder):
    Class, n_samples, inspec_rules = inputs[0], inputs[1], inputs[2]
    
    if type(inspec_rules) is dict:
        delta_t = str(inspec_rules['delta_t'])
        p_th    = str(inspec_rules['p_th'])
        nI      = str(inspec_rules['nI'])
        
    elif type(inspec_rules) is np.ndarray:
        delta_t = str(-1)
        p_th    = str(-1)
        nI      = str(-1)
    else:
        print('*** NO assinged inspection policy ***')
        
    filename = os.path.join(results_folder,
                            datetime.now().strftime('%Y%m%d_%H%M%S_') + \
                            '_t' + delta_t + '_p' + p_th +  \
                            '_n' + nI + '_s' + str(n_samples) + '.pickle')
    # pickle file
    outfile = open(filename, 'wb')
    
    
    for samples in range(n_samples):
        
        Class.DBN_inference(inspec_rules)
        sample_result = [Class.CC, Class.CI, Class.CR, Class.RF, Class.CT]
        pickle.dump(sample_result, outfile)
        # results_list.append(sample_result)
        
    outfile.close()

#%%

def HeristicsEvaluation(input_file):
    
    print ('\nINPUT FILE:', input_file, '\n')
    results_folder = os.path.dirname(input_file)
    
    file = open(input_file,'r')
    inputs = eval(file.read())
    file.close()
    
    globals().update(inputs)
    
    # Initiate system Reliability Class for ALL simulations
    OWT = SysReliab(lifetime, comp_names, KoutN_params, cost_dict)
    OWT.T_import(T_folderpaths) # import transition matrix just once
    
    # for user information
    input_text = 'T: {}\nLifetime: {}\nComponents: {}\nK-out-the-N: {}\nCost Model: {}'.format(
            T_folderpaths, lifetime, comp_names, KoutN_params, cost_dict)
    print('=== inputs ===')
    print(input_text)
    
    #%% Mounting the inputs
    
    # from the Heuristic parameters  
    inputs_list = []
    
    ref = [OWT, 10, {'delta_t':-1, 'p_th':1.0, 'nI':0}]
    pth_range    = heuristics['pth_range']
    deltat_range = heuristics['deltat_range']
    nI_range     = heuristics['nI_range']
    
    if inspec_map:
        InspecMap = pd.read_excel(
            os.path.join(results_folder,'SysReliab__IN_InspecMap.xlsx'),
            index_col=0).values
        inputs_list.append(ref)
        inputs_list.append([OWT, n_samples, InspecMap])
    else:    
        inputs_list.append(ref) # no inspection policy
        for p_th in pth_range:
            for delta_t in deltat_range:
                for nI in nI_range:
                    inputs_list.append([
                        OWT, 
                        n_samples,
                        {'delta_t':delta_t, 
                         'p_th':p_th, 
                         'nI':nI}])
    
    n_policies = len(inputs_list)
    total_simulations = n_policies*n_samples
    print()
    print('- Number of policies:', n_policies, '| Total no. simulations:', total_simulations)
    exptime = timedelta(seconds=total_simulations*0.095)
    print('- Expected computation time:', str(exptime))
    print('       Start time          :', datetime.now())
    print('       (Expected) End time :', datetime.now() + exptime)

    #%% Series computation
    # 60 seconds: for 100 samples and 9 sets (900 simulations)
    # 83 with pickle
    
    start = time()
    results_list = []
    for inputs in inputs_list:
        computeMCS(inputs, results_folder)
    end = time()
    print('- Simulations completed! ElapsedTime :', timedelta(seconds=end-start))
    
    #%% importing pickle
    
    filesnames   = glob(results_folder + '/*.pickle')
    policy_result_list = list()
    
    for filename in filesnames:
        # get info from name
        
        policy_simulations = list()
        with open(os.path.join(filename), 'rb') as fr:
            try:
                while True:
                    policy_simulations.append(pickle.load(fr))
            except EOFError:
                pass
        temp_str = os.path.basename(filename).split('_')
        policy_settings = [
            int(float(temp_str[3][1:])), float(temp_str[4][1:]),
            int(temp_str[5][1:]), int(temp_str[6].split('.')[0][1:])]
        policy_result = np.mean(policy_simulations, axis=0)
        
        policy_result_list.append(np.append(policy_settings, policy_result))
        
    
    #%% storing results
    df_results = pd.DataFrame(
        data=policy_result_list,
        columns=['delta_t', 'p_th', 'nI', 'n_samples', 'CC', 'CI', 'CR', 'RF', 'CT'])
    
    file_path = os.path.join(results_folder, 
        'HeuristicsEvaluation.xlsx')
    df_results.to_excel(file_path)
    print('- saved in:', file_path)
    

#%% Run
if __name__ == '__main__':
    # list to run. 
    # First comumn is to perform or not the run
    run_list = [
        # k-out-of-n Series System
        [False,'data/MCS_HeurEval/Sys1out4/_input.txt'],
        [False,'data/MCS_HeurEval/Sys2out4/_input.txt'],
        [False,'data/MCS_HeurEval/Sys3out4/_input.txt'],
        [False,'data/MCS_HeurEval/Sys4out4/_input.txt'],
        # Individual Zone System
        [False,'data/MCS_HeurEval/CombZone/Comp1out4_atm/_input.txt'],
        [False,'data/MCS_HeurEval/CombZone/Comp1out4_sub/_input.txt'],
        [False,'data/MCS_HeurEval/CombZone/Comp2out4_atm/_input.txt'],
        [False,'data/MCS_HeurEval/CombZone/Comp2out4_sub/_input.txt'],
        [False,'data/MCS_HeurEval/CombZone/Comp3out4_atm/_input.txt'],
        [False,'data/MCS_HeurEval/CombZone/Comp3out4_sub/_input.txt'],
        [False,'data/MCS_HeurEval/CombZone/Comp4out4_atm/_input.txt'],
        [False,'data/MCS_HeurEval/CombZone/Comp4out4_sub/_input.txt'],
        # Simple Series System
        [False,'data/MCS_HeurEval/SysSeries/_input.txt'],
        # Individual Component
        [False,'data/MCS_HeurEval/CombComp/Comp1out4_atm/_input.txt'],
        [False,'data/MCS_HeurEval/CombComp/Comp1out4_sub/_input.txt'],
        [False,'data/MCS_HeurEval/CombComp/Comp2out4_atm/_input.txt'],
        [False,'data/MCS_HeurEval/CombComp/Comp2out4_sub/_input.txt'],
        [False,'data/MCS_HeurEval/CombComp/Comp3out4_atm/_input.txt'],
        [False,'data/MCS_HeurEval/CombComp/Comp3out4_sub/_input.txt'],
        [False,'data/MCS_HeurEval/CombComp/Comp4out4_atm/_input.txt'],
        [False,'data/MCS_HeurEval/CombComp/Comp4out4_sub/_input.txt'],
        # DNVGL
        [False,'data/MCS_HeurEval\DNVGL\sys1out4\_input.txt'],
        [False,'data/MCS_HeurEval\DNVGL\sys1out4_atm\_input.txt'],
        [False,'data/MCS_HeurEval\DNVGL\sys1out4_sub\_input.txt'],
        [False,'data/MCS_HeurEval\DNVGL\sys1out4_atm_sub\_input.txt'],
        [False,'data/MCS_HeurEval\DNVGL\sys2out4\_input.txt'],
        [False,'data/MCS_HeurEval\DNVGL\sys2out4_atm\_input.txt'],
        [False,'data/MCS_HeurEval\DNVGL\sys2out4_sub\_input.txt'],
        [False,'data/MCS_HeurEval\DNVGL\sys2out4_atm_sub\_input.txt'],
        [False,'data/MCS_HeurEval\DNVGL\sys3out4\_input.txt'],
        [False,'data/MCS_HeurEval\DNVGL\sys3out4_atm\_input.txt'],
        [False,'data/MCS_HeurEval\DNVGL\sys3out4_sub\_input.txt'],
        [False,'data/MCS_HeurEval\DNVGL\sys3out4_atm_sub\_input.txt'],
        [False,'data/MCS_HeurEval\DNVGL\sys4out4\_input.txt'],
        [False,'data/MCS_HeurEval\DNVGL\sys4out4_atm\_input.txt'],
        [False,'data/MCS_HeurEval\DNVGL\sys4out4_sub\_input.txt'],
        [False,'data/MCS_HeurEval\DNVGL\sys4out4_atm_sub\_input.txt'],
        # Redundancy
        [False, 'data/MCS_HeurEval/redundancy_play/ref/_input.txt'],
        [False, 'data/MCS_HeurEval/redundancy_play/atm/_input.txt'],
        [False, 'data/MCS_HeurEval/redundancy_play/sub/_input.txt'],
        [False, 'data/MCS_HeurEval/redundancy_play/all/_input.txt'],
        [False, 'data/MCS_HeurEval/redundancy_play/atm_sub/_input.txt'],
        [False, 'data/MCS_HeurEval/redundancy_play/comp_atm/_input.txt'],
        [False, 'data/MCS_HeurEval/redundancy_play/comp_sub/_input.txt'],
        # with monopile dimensions (for paper)
        [False, 'data/MCS_HeurEval/Monopile/ref/_input.txt'],
        [False, 'data/MCS_HeurEval/Monopile/atm/_input.txt'],
        [False, 'data/MCS_HeurEval/Monopile/sub/_input.txt'],
        [False, 'data/MCS_HeurEval/Monopile/all/_input.txt'],
        [False, 'data/MCS_HeurEval/Monopile/atm_sub/_input.txt'],
        [False, 'data/MCS_HeurEval/Monopile/comp_atm/_input.txt'],
        [False, 'data/MCS_HeurEval/Monopile/comp_sub/_input.txt'],
        # with monopiles dimentions time-based policy
        [False, 'data/MCS_HeurEval/Monopile/DNV_ref/_input.txt'],
        [False, 'data/MCS_HeurEval/Monopile/DNV_atm/_input.txt'],
        [False, 'data/MCS_HeurEval/Monopile/DNV_sub/_input.txt'],
        [False, 'data/MCS_HeurEval/Monopile/DNV_all/_input.txt'],
        [False, 'data/MCS_HeurEval/Monopile/DNV_atm_sub/_input.txt'],
        [False, 'data/MCS_HeurEval/Monopile/DNV_comp_atm/_input.txt'],
        [False, 'data/MCS_HeurEval/Monopile/DNV_comp_sub/_input.txt'],
        # with monopiles dimentions individual compnent policy optimization
        # in system level
        [True, 'data/MCS_HeurEval/Monopile/IndComp_ref/_input.txt'],
        [True, 'data/MCS_HeurEval/Monopile/IndComp_atm/_input.txt'],
        [True, 'data/MCS_HeurEval/Monopile/IndComp_sub/_input.txt'],
        [True, 'data/MCS_HeurEval/Monopile/IndComp_all/_input.txt'],
        [True, 'data/MCS_HeurEval/Monopile/IndComp_atm_sub/_input.txt']
        
    ]
    
    # Verify the data
    all_inputs = list()
    for run_set in run_list:
        
        file = open(run_set[1],'r')
        inputs_dict = eval(file.read())
        file.close()
        
        all_inputs.append(inputs_dict)
    
    df = pd.DataFrame(all_inputs)
    df.to_excel('data/MCS_HeurEval/inputs_resume.xlsx')
    
    # ask for user verification
    print('\nSee data/MCS_HeurEval/inputs_resume.xlsx to check the inputs')
    input('Press ENTER to continue...')
    
    # RUN!
    for run_set in run_list:
        if run_set[0]:
            HeristicsEvaluation(run_set[1])
            
    print('\n=== end of all simulations ===')

    
    


    
    