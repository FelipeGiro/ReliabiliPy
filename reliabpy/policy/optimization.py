import numpy as np
from itertools import product
from datetime import datetime, timedelta
import time
import pickle
import os
from glob import glob
from tabulate import tabulate
import copy
import pandas as pd

from reliabpy.policy.policy import HeuristicRules

class HeuristicBased:
    def __init__(self, model, save_folder, project_name = 'optimization'):
        self.model = model
        self.save_folder = save_folder 
        self.project_name = project_name
    
    def mount_policies_to_search(self, delta_t_array, nI_array, n_samples):
        self.policies = list(product(delta_t_array, nI_array))
        self.n_samples = n_samples
        self.left_samples = n_samples*len(self.policies)
        
    def run_samples(self):
        
        self.start_time = datetime.now()
        self.save_folder = os.path.join(self.save_folder, self.start_time.strftime("%Y%m%d_%H%M%S_") + self.project_name)
        os.mkdir(self.save_folder)

        with open(os.path.join(self.save_folder, "_input.txt"), 'w') as input_reg:
            input_txt = tabulate(self.policies, headers=['Delta_t', 'nI'])
            input_reg.write(input_txt)

        print(f"=== start of simulation : {self.start_time} ===")
        for delta_t, nI in self.policies:
            start = datetime.now()
            policy = HeuristicRules(delta_t, nI)
            policy.import_model(self.model.monopile)
            self.model.monopile.policy_rules = policy

            with open(os.path.join(self.save_folder, datetime.now().strftime("d%Y%m%dt%H%M%S") + f"__s_{self.n_samples}__deltat_{delta_t}__nI_{nI}.npy"), 'wb') as outfile:
                for samples in range(self.n_samples):
                    sample_results = self.model.run_one_episode()
                    pickle.dump(sample_results, outfile)
                    self.model.monopile._reset()

            end = datetime.now()
            self.left_samples -= self.n_samples
            episode_time = (end - start)/self.n_samples
            print(f'- Mean episode time: {episode_time} | Remaining time: {episode_time*self.left_samples}')
            print('\t Expect to finish at:', datetime.now() + episode_time*self.left_samples)
                
        self.end_time = datetime.now()
        print(f"=== end of simulation : {self.end_time} ===")
        print(f"Elsapsed time : {self.end_time - self.start_time}")

    def process_data(self, load_folder=None):
        if load_folder is None:
            self.load_folder = self.save_folder
        else:
            self.load_folder =  load_folder
        
        output_path_list = glob(os.path.join(self.load_folder, '*.npy'))
        writer = pd.ExcelWriter(os.path.join(self.load_folder, 'OptimizarionResults.xlsx'), engine='xlsxwriter')
        for output_path in output_path_list:
            # getting the policy parameters
            policy_dict = dict()
            for temp in os.path.basename(os.path.splitext(output_path)[0]).split('__')[1:]:
                variable, value = temp.split('_')
                policy_dict[variable] = float(value)

            policy_samples = []
            with open(output_path, "rb") as f:
                while True:
                    try:
                        policy_samples.append(pickle.load(f))
                    except EOFError:
                        break
            delta_t, nI = int(policy_dict['deltat']), int(policy_dict['nI'])
            df_temp = pd.DataFrame(policy_samples)
            df_temp.to_excel(writer, sheet_name=f't {delta_t} nI {nI}')
        writer.save()
if __name__ == '__main__':
    from reliabpy.examples.offshore_wind_turbine import Simple 

    model = Simple()
    model.mount_model()
    opt = HeuristicBased(model, "C:\\Developments\\reliabpy\\PhD\\examples")
    opt.mount_policies_to_search(delta_t_array=[5,10,15], nI_array=[2,4,8], n_samples=25)
    opt.run_samples()
    opt.process_data()

    print()