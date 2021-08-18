import unittest
import numpy as np
import numpy.testing as np_test
from reliabpy.models.inference import MonteCarloSimulation, DynamicBayesianNetwork, metrics
from reliabpy.readwrite.ANAST import *
from reliabpy.models.deterioration import *
from scipy.special import gamma

class TestALL(unittest.TestCase):
    def setUp(self):
        # TODO: test imports
        # change the import files
        lnC_mean, lnC_std = import_calibrated_values("C:\\Developments\\reliabpy\PhD\\transition_matrices\\atm\\cal_out.mat")
        DFF, lifetime, a0_mean, a_crit, description, h, sn_params, m, q_cov, n_samples, n = import_component_inputs("C:\\Developments\\reliabpy\PhD\\transition_matrices\\atm\\_SNparams.mat")
        q_mean = import_weilbull_mean("C:\\Developments\\reliabpy\\PhD\\transition_matrices\\atm\\q_out.mat")
        q = np.random.normal(q_mean, q_mean*q_cov, n_samples)
        C = np.random.lognormal(lnC_mean, lnC_std, n_samples)
        a_0 = np.random.exponential(a0_mean, n_samples)
        S = q*gamma(1.0+1.0/h)

        Y_g = GeometricFactor.lognormal(n_samples=n_samples)
        det_model = Paris_Erdogan()
        det_model.initialize(a_0, m, n, C, S, Y_g)
        function = det_model.propagate

        inspection_years = [9, 17]

        # DBN
        T_path = "PhD\\transition_matrices\\atm\\dr_out.mat"
        T, b0, discretizations = import_DBN_input_data(T_path)

        self.dbn = DynamicBayesianNetwork(T, discretizations, b0)
        self.dbn.force_detection, self.dbn.force_notdetection = False, True
        while self.dbn.t <= lifetime:
            self.dbn.predict()

            if self.dbn.t in inspection_years:
                self.dbn.update(parameters={'quality': 'bad'})

        # MCS
        self.mcs = MonteCarloSimulation(a_0, function, a_crit)
        self.mcs.force_detection, self.mcs.force_notdetection = False, True

        while self.mcs.t <= lifetime:
            self.mcs.predict()
            
            if self.mcs.t in inspection_years:
                self.mcs.update(parameters={'quality': 'bad'})

    def test_rmse(self):
        rmse_pf = metrics.pf_rmse(self.mcs, self.dbn)

        self.assertLess(rmse_pf, 1e-3)

if __name__ == '__main__':
    unittest.main()