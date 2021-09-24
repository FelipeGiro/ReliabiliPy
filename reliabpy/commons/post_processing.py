import numpy as np 
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import pandas as pd

class OneEpisode:
    """
    Processing for one episode
    ==========================
    
    Class focused in processing results for just one episode.

    Parameters:
    system_model : system model object
        System model after the simulation.
    savefolder : path
        Save folder to save all results.
    """
    def __init__(self, system_model, savefolder=False):
        self.system_model = system_model
        self.savefolder = savefolder

    def plot_overview(self):
        """
        Plot overview
        =============

        Show or save a figure with three plots:
        1) Probability of failure of all components.
        2) Probability of failure of the entire structural system.
        3) Cost breakdown.

        Where abscissas are the timestep in years. 
        """
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
        t, C_C, C_I, C_R, R_F, C_T = self.system_model.yearly_costs_breakdown.values()
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

    def plot_interactive(self):
        # TODO: function documentation
        components_dict, system_pf = self.system_model.get_components_results()
        
        fig = go.Figure()
        for key in components_dict:
            fig.add_trace(
                go.Scatter(
                    x=components_dict[key]['time'], 
                    y=components_dict[key]['pf'],
                    line=dict(color="#4d4d4d"),
                    name = key,
                    hovertemplate='<b>Year</b>: %{x}<br><b>P<sub>F</sub></b>:%{y}<br><b>Action:</b> %{text}',
                    text=[str(a) +' | '+ str(o) for a, o in zip(components_dict[key]['action'], components_dict[key]['output'])])
            )
        fig.add_trace(
            go.Scatter(
                x=np.array(system_pf)[:, 0], 
                y=np.array(system_pf)[:, 1],
                line=dict(color='#c20000'),
                name='System',
                hovertemplate='Year: %{x}<br>P_F:%{y}'
            )
        )
        fig.update_layout(
            xaxis = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 1
            )
)
        fig.write_html(os.path.join(self.savefolder, 'SystemReliability.html'))

    def to_excel(self):
        """
        Save to excel
        =============

        Save the results in a Excel spreadsheet with the tabs:
        - CostBreakdown : yearly cost breakdown
        - <component_id> : each tab have its own tab
        - System : system probability of failure
        """
        writer = pd.ExcelWriter(os.path.join(self.savefolder, 'SystemReliability.xlsx'), engine='xlsxwriter')

        df_costs = pd.DataFrame(self.system_model.yearly_costs_breakdown)
        df_costs.set_index('t', inplace=True)
        df_costs.to_excel(writer, sheet_name='Cost_Breakdown')

        # system
        df_system = pd.DataFrame(self.system_model.system_pf)
        df_system.columns = ['t', 'pf']
        df_system.set_index('t', inplace=True)
        df_system.to_excel(writer, sheet_name="System_PF")

        # components
        for component in self.system_model.components_list:
            temp_df = pd.DataFrame({
            't' : component.t, 
            'pf' : np.array(component.pf), 
            'action' : component.action, 
            'output' : component.output})
            temp_df.set_index('t', inplace=True)
            temp_df.to_excel(writer, sheet_name=component.id)
        
        writer.save()


    def to_excel_depreciated(self):
        """
        Save to excel
        =============

        Save the results in a Excel spreadsheet with the tabs:
        - CostBreakdown : yearly cost breakdown
        - ObservationMap : time and location of observations on components
        - ActionMap : time and location of actions on components 
        """
        df_costs = pd.DataFrame(self.system_model.yearly_costs_breakdown)
        df_costs.set_index('t', inplace=True)

        df_action, df_output = self._build_maps(self.system_model.components_list)

        # TODO: put this in another function
        list_df, comp_names = list(), list()
        for component in self.system_model.components_list:
            comp_names.append(component.id)
            t = np.array(component.t, dtype=float)
            pf = component.pf
            mask = np.append([False], ~(t[1:] - t[:-1]).astype(bool))
            if any(mask):
                add_sth = 0.000001
                t[mask] += np.linspace(add_sth, add_sth*mask.sum(), num=mask.sum())
            list_df.append(pd.DataFrame(data=pf, index=t, columns=['pf']))
        system_pf = np.array(self.system_model.system_pf)
        t = system_pf[:, 0]
        pf = system_pf[:, 1]
        list_df.append(pd.DataFrame(index = system_pf[:, 0], data = system_pf[:, 1]))
        mask = np.append([False], ~(t[1:] - t[:-1]).astype(bool))
        if any(mask):
            add_sth = 0.000001
            t[mask] += np.linspace(add_sth, add_sth*mask.sum(), num=mask.sum())
        comp_names.append("SYSTEM")
        df_pfs = pd.concat(list_df, axis=1, join='outer')
        df_pfs.columns = comp_names

        writer = pd.ExcelWriter(os.path.join(self.savefolder, 'SystemReliability.xlsx'), engine='xlsxwriter')

        df_costs.to_excel(writer, sheet_name='Cost_Breakdown')
        df_action.to_excel(writer, sheet_name='Action_Map')
        df_output.to_excel(writer, sheet_name='Output_Map')
        df_pfs.to_excel(writer, sheet_name='Prob_Failure')
        
        writer.save()

    def _build_maps(self, components_list):
        list_i, list_a, list_o = list(), list(), list()
        for comp in components_list:
            id, t, a, o = comp.id, comp.t, comp.action, comp.output
            list_i.append(id)
            list_a.append(pd.DataFrame(a, index=t))
            list_o.append(pd.DataFrame(o, index=t))

        df_action = pd.concat(list_a, axis=1, join='outer')
        df_output = pd.concat(list_o, axis=1, join='outer')

        df_action.columns = list_i
        df_output.columns = list_i

        return df_action, df_output

class MonteCarlo:
    def __init__(self, folderpath):
        all_policies = list()
        with open(folderpath, 'rb') as file:
            while True:
                try:
                    pickle_file = pickle.load(file)
                    all_policies.append(pickle_file)
                except EOFError:
                    break
        self.all_policies = all_policies

if __name__ == '__main__':
    test = MonteCarlo("C:\\Developments\\reliabpy\\PhD\\examples\\policies_results.pickle")
