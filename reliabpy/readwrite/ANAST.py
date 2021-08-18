import numpy as np
from scipy.io import loadmat
import os
from scipy.special import gamma
from reliabpy.models.deterioration import GeometricFactor, Paris_Erdogan

"""
ANAST import functions
======================

these function are to rapidly import MatLab ANAST results. 

possible TODO: transform the rest of Matlab code into python
"""

def import_DBN_input_data(path):
    '''
    Import DBN input data
    =====================

    function to import dr_out.mat data.

    File author: Palbo Morato

    Parameter
    ---------
    path : file path
        file path of dr_out.mat
    
    Return:
    -------
    T : matrix 
        transtiion matrix (n, n)
    b0 : array 
        initial belief state (n)
    discretizations : dict
        discitonary with th einformation of t and a discretization
    '''
    MatLab_file = loadmat(path)['dr_env'][0]
    
    aint = MatLab_file['aint'][0]
    T = MatLab_file['T0'][0].toarray()
    b0 = MatLab_file['b0'][0]
    
    discretizations = {'t': np.linspace(0,21,22), 'a':aint[0]}
    
    return T, b0, discretizations

def import_component_inputs(path):
    '''
    Import components imputs
    ========================

    function to import data for SN parameters.
    
    File author: Felipe Giro

    Parameter
    ---------
    path : file path
        file path of dr_out.mat
    
    Return:
    -------
    DFF : float
        Design fatigue factor
    lifetime : int
        structure lifetime
    a0_mean : float
        initial crack size mean of exponential distribution 
    a_crit : float
        critical crack size
    description : string
        component description
    h : float
        TODO: update that
    sn_params : dict 
        parameters of SN curve (la1, la2, m1, m2)
    m : float
        crack growth parameter
    q_cov : float
        Weibull covariance
    n_samples : int
        number of samples (for Monte Carlo Simulation)
    n_cycles : int
        Number of cycles
    '''

    M = loadmat(path)

    DFF, lifetime, a0_mean, a_crit, \
    description, h, \
    sn_params, m, \
    q_cov, n_samples, n_cycles =  \
    float(M['FDF']), int(M['T']), float(M['a0_mean']), float(M['acrit']), \
    M['description'][0], float(M['h']), \
    { your_key: float(M[your_key]) for your_key in ['la1', 'la2', 'm1', 'm2'] },\
    float(M['m']), \
    float(M['q_cov']), int(M['samp']), int(M['v'])

    return DFF, lifetime, a0_mean, a_crit, description, h, sn_params, m, q_cov, n_samples, n_cycles

def import_calibrated_values(path):
    '''
    Import calibrated values
    ========================

    function to import cal_out.mat data.

    File author: Pablo Morato

    Parameter
    ---------
    path : file path
        file path of cal_out.mat
    
    Return:
    -------
    lnC_mean : float
        C mean of logaritmic distribution
    lnC_std : float
        C standart deviation of logaritmic distribution
    '''

    M = loadmat(path)
    lnC_mean, lnC_std = M['lnC_cal'][0,0], M['lnC_cal'][0,1]

    return lnC_mean, lnC_std

def import_weilbull_mean(path):
    '''
    Weibull mean
    ============

    function to import q_out.mat data.

    File author: Pablo Morato

    Parameter
    ---------
    path : file path
        file path of cal_out.mat

    Return
    ------
    q : float
        Weibull paramater mean
    '''
    
    M = loadmat(path)
    q = M['q_det'][0,0]

    return q

def get_deterioration_model(folder_path):
    lnC_mean, lnC_std = import_calibrated_values(os.path.join(folder_path, "cal_out.mat"))
    DFF, lifetime, a0_mean, a_crit, description, h, sn_params, m, q_cov, n_samples, n = import_component_inputs(os.path.join(folder_path,"_SNparams.mat"))
    q_mean = import_weilbull_mean(os.path.join(folder_path,"q_out.mat"))
    q = np.random.normal(q_mean, q_mean*q_cov, n_samples)
    C = np.random.lognormal(lnC_mean, lnC_std, n_samples)
    a_0 = np.random.exponential(a0_mean, n_samples)
    S = q*gamma(1.0+1.0/h)

    Y_g = GeometricFactor.lognormal(n_samples=n_samples)
    det_model = Paris_Erdogan()
    det_model.initialize(a_0, m, n, C, S, Y_g)
    function = det_model.propagate

    import_DBN_input_data()

    return function



