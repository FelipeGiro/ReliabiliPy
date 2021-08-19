import numpy as np 
import matplotlib.pyplot as plt

def plot_system(results_dict, savefolder=False):
    fig, axes = plt.subplots(3, figsize=(10,10))
    
    ax = axes[0]
    ax.set_title('Components probability of failure')
    for key in results_dict:
        ax.plot(results_dict[key]['time'], results_dict[key]['pf'])
    ax.legend(results_dict.keys())
    
    ax = axes[1]
    ax.set_title('System probability of failure')
    # TODO: implement system

    ax = axes[2]
    ax.set_title('Cost breakdown')
    # TODO: implement cost breakdown

    if savefolder:
        plt.savefig(savefolder)
    else:
        plt.plot()
    
    

