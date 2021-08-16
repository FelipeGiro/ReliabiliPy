from reliabpy.method import MonteCarlo, DynamicBayesianNetwork
from reliabpy.readwrite.ANAST import *
from reliabpy.deterioration import FractureMechanics as fm
from scipy.special import gamma

import matplotlib.pyplot as plt

lnC_mean, lnC_std = import_calibrated_values("C:\\Developments\\reliabpy\PhD\\transition_matrices\\cal_out_atm.mat")
DFF, lifetime, a0_mean, a_crit, description, h, sn_params, m, q_cov, n_samples, n = import_component_inputs("C:\\Developments\\reliabpy\PhD\\transition_matrices\\0atm_SNparams.mat")
q_mean = import_weilbull_mean("C:\\Developments\\reliabpy\\PhD\\transition_matrices\\q_out_atm.mat")
q = np.random.normal(q_mean, q_mean*q_cov, n_samples)

C = np.random.lognormal(lnC_mean, lnC_std, n_samples)
a_0 = np.random.exponential(a0_mean, n_samples)
S = q*gamma(1.0+1.0/h)

Y_g = fm.GeometricFactor.lognormal(n_samples=n_samples)
det_model = fm.Paris_Erdogan()
det_model.initialize(a_0, m, n, C, S, Y_g)
function = det_model.propagate

# MCS
mcs = MonteCarlo.foward_propagation()
mcs.initialize(a_0, function, a_crit)

for t in range(lifetime):
    mcs.predict()
    # TODO: update for MCS
    # after and from MonteCarlo class

# DBN
T_path = "PhD\\transition_matrices\\dr_out_atm.mat"
T, b0, discretizations = import_DBN_input_data(T_path)

dbn = DynamicBayesianNetwork.DeteriorationRate()
dbn.initialize(T, discretizations, b0)
for t in range(lifetime):
    dbn.predict()
    # TODO: update for DBN
    # after and from DynamicBayesianNetwork class

plt.plot(*mcs.get_pfs(), label='MCS', ls='--')
plt.plot(*dbn.get_pfs(), label='DBN')
plt.legend()
plt.show()
print()
