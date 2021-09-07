import numpy as np
from itertools import product
from datetime import datetime, timedelta
import time
import pickle
import os
from tabulate import tabulate

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
        
    def run(self):
        
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

            with open(os.path.join(self.save_folder, datetime.now().strftime("d%Y%m%dt%H%M%S") + f"__s_{self.n_samples}__deltat_{delta_t}__nI_{nI}"), 'wb') as outfile:
                episodes = list()
                for samples in range(self.n_samples):
                    pickle.dump(self.model.run_one_episode(), outfile)
                    self.model.monopile._reset()

            end = datetime.now()
            self.left_samples -= self.n_samples
            episode_time = (end - start)/self.n_samples
            print(f'- Mean episode time: {episode_time} | Remaining time: {episode_time*self.left_samples}')
            print('\t Expect to finish at:', datetime.now() + episode_time*self.left_samples)
                
        self.end_time = datetime.now()
        print(f"=== end of simulation : {self.end_time} ===")
        print(f"Elsapsed time : {self.end_time - self.start_time}")

if __name__ == '__main__':
    from reliabpy.examples.offshore_wind_turbine import Simple 

    model = Simple()
    model.mount_model()
    opt = HeuristicBased(model, "C:\\Developments\\reliabpy\\PhD\\examples")
    opt.mount_policies_to_search(delta_t_array=[3,4,5,6,7,8,9,10,11,12,13,14,15,16], nI_array=[1,2,3,4,5,6,7,8], n_samples=25)
    opt.run()

    print()