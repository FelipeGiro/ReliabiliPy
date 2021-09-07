import numpy as np
from itertools import product
from datetime import datetime, timedelta
import time
import pickle
import os

from reliabpy.policy.policy import HeuristicRules

class HeuristicBased:
    def __init__(self, model, save_folder, project_name = 'optimization'):
        self.model = model
        self.save_folder = save_folder 
        self.project_name = project_name
    
    def mount_policies_to_search(self, delta_t_array, nI_array, n_samples):
        combinations = product(delta_t_array, nI_array)
        self.policies = []
        self.n_samples = n_samples

        print("Policies to search")
        print("==================")
        i = 0
        for delta_t, nI in combinations:
            self.policies.append((delta_t, nI))
            i += 1
            print(f'- {i}: Delta_t : {delta_t} years | nI : {nI} | n_samples : {n_samples}')
        self.left_samples = i*n_samples
        print('Total number of samples:', self.left_samples)
        input('Press enter to continue... ')
        
    def run(self):
        
        self.start_time = datetime.now()
        os.mkdir(os.path.join(self.save_folder, self.start_time.strftime("%Y%m%d_%H%M%S_") + self.project_name))

        print(f"=== start of simulation : {self.start_time} ===")
        opt_cost = np.inf
        outfile = open(os.path.join(self.save_folder, f"policies_results.pickle"), 'wb')
        for delta_t, nI in self.policies:
            start = datetime.now()
            policy = HeuristicRules(delta_t, nI)
            policy.import_model(self.model.monopile)
            self.model.monopile.policy_rules = policy

            episodes = list()
            for samples in range(self.n_samples):
                episodes.append(list(self.model.run_one_episode().values()))
                self.model.monopile._reset()
            policy_costs = np.mean(episodes, axis=0)
            total_cost = policy_costs.sum()
            policy_result = delta_t, nI, self.n_samples, policy_costs, total_cost, np.std(episodes, axis=0), np.max(episodes, axis=0), np.min(episodes, axis=0)
            
            if opt_cost > total_cost:
                opt_cost = total_cost
                print(f"{datetime.now()} - Best policy so far: delta_t : {delta_t} | nI : {nI} | total_cost : {opt_cost}")

            pickle.dump(policy_result, outfile)

            end = datetime.now()
            self.left_samples -= self.n_samples
            episode_time = (end - start)/self.n_samples
            print(f'- Mean episode time: {episode_time} | Remaining time: {episode_time*self.left_samples}')
            print('\t Expect to finish at:', datetime.now() - episode_time*self.left_samples)
                
        self.end_time = datetime.now()
        print(f"=== end of simulation : {self.end_time} ===")
        print(f"Elsapsed time : {self.end_time - self.start_time}")

if __name__ == '__main__':
    from reliabpy.examples.offshore_wind_turbine import Simple 

    model = Simple()
    model.mount_model()
    opt = HeuristicBased(model, "C:\\Developments\\reliabpy\\PhD\\examples")
    opt.mount_policies_to_search(delta_t_array=[3,4,5,6,7,8,9,10,11,12,13,14,15,16], nI_array=[1,2,3,4,5,6,7,8], n_samples=10000)
    opt.run()

    print()