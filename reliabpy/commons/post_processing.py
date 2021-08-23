import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd

class OneEpisode:
    def __init__(self, system_model, savefolder=False):
        self.system_model = system_model
        self.savefolder = savefolder

    def plot_overview(self):

        components_dict, system_pf = self.system_model.get_components_results()

        fig, axes = plt.subplots(3, figsize=(10,10), sharex=True)
        
        ax = axes[0]
        ax.set_title('Components probability of failure')
        for key in components_dict:
            ax.plot(components_dict[key]['time'], components_dict[key]['pf'], color='k', alpha=0.2)
        ax.legend(components_dict.keys(), ncol=3)
        
        ax = axes[1]
        ax.set_title('System probability of failure')
        ax.plot(np.array(system_pf)[:, 0], np.array(system_pf)[:, 1])

        ax = axes[2]
        t, C_C, C_I, C_R, R_F = self.system_model.yearly_costs_breakdown.values()
        width = 0.35
        ax.set_title('Cost breakdown')
        ax.bar(t, R_F, width, label = 'Failure', color= 'darkred')
        ax.bar(t, C_C, width, bottom = R_F, label = 'Campaign', color = '0.5')
        ax.bar(t, C_I, width, bottom = R_F + C_C, label = 'Inspection', color = 'darkblue')
        ax.bar(t, C_R, width, bottom = R_F + C_C + C_I, label = 'Repair', color = 'darkgreen')
        ax.legend()
        

        if self.savefolder:
            plt.savefig(os.path.join(self.savefolder, 'SystemReliability.png'))
        else:
            plt.plot()

    def to_excel(self):
        df_costs = pd.DataFrame(self.system_model.yearly_costs_breakdown)
        df_costs.set_index('t', inplace=True)

        obs_map = self._build_map(self.system_model.system_obs, index=df_costs.index)
        action_map = self._build_map(self.system_model.system_obs, index=df_costs.index)

        df_obs = pd.DataFrame(obs_map, index=df_costs.index)
        df_action = pd.DataFrame(action_map, index=df_costs.index)

        writer = pd.ExcelWriter(os.path.join(self.savefolder, 'SystemReliability.xlsx'), engine='xlsxwriter')

        df_costs.to_excel(writer, sheet_name='Cost_Breakdown')
        df_obs.to_excel(writer, sheet_name='Observation_Map')
        df_action.to_excel(writer, sheet_name='Action_Map')
        
        writer.save()

    def _build_map(self, comp_dict, index):

        for key, value in comp_dict.items():
            base_array = np.zeros_like(index, dtype=bool)
            base_array[value] = True
            comp_dict[key] = base_array
        
        return comp_dict

# TODO: a function to transform the computed data in DataFrame
    

