# Costs
# 2019 - Luque, Straub - Risk-based optimal inspection strategies for 
# structural systems using dynamic Bayesian networks 
# Table 4, case 1

class InspectionMaintenance:
    def __init__(self, c_c=5.0, c_i=1.0, c_r=10.0, c_f=10000, r=0.02):
        self.c_c, self.c_i, self.c_r, self.c_c, self.r =  c_c, c_i, c_r, c_c, r
    
    def compute_cost_breakdown(self, system_model):
        C_C, C_I, C_R, C_C = 0, 0, 0, 0
        for component in system_model.components_list:
            pass
            # TODO: compute system costs
            # C_C += 
            # C_I +=
            # C_R +=
            # C_C +=