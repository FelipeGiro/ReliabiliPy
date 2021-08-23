# Costs
# 2019 - Luque, Straub - Risk-based optimal inspection strategies for 
# structural systems using dynamic Bayesian networks 
# Table 4, case 1

import numpy as np

class InspectionMaintenance:
    def __init__(self, c_c=5.0, c_i=1.0, c_r=10.0, c_f=10000, r=0.02):
        self.c_c, self.c_i, self.c_r, self.c_c, self.c_f, self.r =  c_c, c_i, c_r, c_c, c_f, r
    
    def compute_cost_breakdown(self, system_model):
        
        system_obs, system_action = dict(), dict()
        t, pf = np.vstack(system_model.system_pf).T
        unique_mask = np.diff(t) == 1
        delta_pf = np.diff(pf)[unique_mask]
        abs_t = np.unique(t).astype(int) # t[1:][unique_mask]

        C_C, C_I, C_R, R_F = np.zeros_like(abs_t, dtype=float), np.zeros_like(abs_t, dtype=float), np.zeros_like(abs_t, dtype=float), np.zeros_like(abs_t, dtype=float)
        for component in system_model.components_list:
            y_t = (1 - self.r)**(np.array(component.t))
            if any(component.obs):
                comp_t = np.array(component.t)
                t_obs = comp_t[component.obs]
                t_action = comp_t[component.action]

                C_I[t_obs] += self.c_i*(1 - self.r)**t_obs
                C_R[t_action] += self.c_r*(1 - self.r)**t_action

                system_obs[component.id]= np.array(comp_t)[component.obs]
                system_action[component.id] = np.array(comp_t)[component.action]
        
        t_temp = np.unique(np.concatenate(list(system_obs.values())))
        C_C[t_temp] += self.c_c*(1 - self.r)**(t_temp)
        
        R_F[abs_t[1:]] = self.c_f*delta_pf*(1 - self.r)**abs_t[1:]
        
        system_model.system_obs = system_obs
        system_model.system_action = system_action
        system_model.yearly_costs_breakdown =  {'t' : abs_t, 'C_C' : C_C, 'C_I' : C_I, 'C_R' : C_R, 'R_F' : R_F}
        system_model.cost_breakdown = {'C_C' : C_C.sum(), 'C_I' : C_I.sum(), 'C_R' : C_R.sum(), 'R_F' : R_F.sum()}