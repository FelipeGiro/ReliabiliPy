import numpy as np 
import matplotlib.pyplot as plt
import os

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

# TODO: a function to transform the computed data in DataFrame
    

