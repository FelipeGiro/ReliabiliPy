# -*- coding: utf-8 -*-
"""
NORMAL AND LOGNORMAL RELATIONS
==============================

this script trqnsformans the parameters from lognormal to normal and 
vice-versa.
"""
import numpy as np

def N2logN(mean, std):
    '''
    NORMAL TO LOGNORMAL
    ===================
    
    Converts the mean and standard deviation of a Normal distribution to a 
    LogNormal distribution.
    
    Parameters
    ----------
    mean : float
        Normal mean.
    std : float
        Normal standard deviation.

    Returns
    -------
    l_mean : float
        LogNormal mean.
    l_std : float
        LogNormal standard deviation.

    '''
    l_mean = np.exp(mean + 0.5*std**2)
    l_std2 = np.exp(2*mean + std**2)*(np.exp(std**2) - 1)
    return l_mean, np.sqrt(l_std2)

def logN2N(l_mean, l_std):
    '''
    LOGNORMAL TO NORMAL
    ===================

    Converts the mean and standard deviation of a LogNormal distribution to a 
    Normal distribution.

    Parameters
    ----------
    l_mean : float
        LogNormal mean.
    l_std : float
        LogNormal standard deviation.

    Returns
    -------
    mean : float
        Normal mean.
    std : float
        Normal standard deviation.

    '''
    mean = 2*np.log(l_mean) - 0.5*np.log(l_std**2 + l_mean**2)
    std2 = -2*np.log(l_mean) + np.log(l_std**2 + l_mean**2)
    return mean, np.sqrt(std2)

if __name__ == '__main__':
    ####################
    # JUST FOR TESTING #
    ####################
    
    import matplotlib.pyplot as plt
    from scipy.stats import norm, lognorm
    
    
    ##### Testing for the statistical parameters transformation ######
    
    # initial parameters
    ln_mean = 2.
    lh_std  = 8.
   
    mean, std = logN2N(ln_mean, lh_std)
    
    ln_mean, ln_std = N2logN(mean, std)
    
    # statistical distribution
    N   = np.random.normal(   mean, std, size=10000)
    lnN = np.random.lognormal(mean, std, size=10000)
    
    all_dist = [
        [N        , lnN],
        [np.exp(N), np.log(lnN)]
    ]
    all_mean = [
        [np.mean(all_dist[0][0]), np.mean(all_dist[0][1])],
        [np.mean(all_dist[1][0]), np.mean(all_dist[1][1])]
    ]
    all_std = [
        [np.std(all_dist[0][0]), np.std(all_dist[0][1])],
        [np.std(all_dist[1][0]), np.std(all_dist[1][1])]
    ]
    
    all_stats = [
        [norm.fit(   all_dist[0][0]), lognorm.fit(all_dist[0][1])],
        [lognorm.fit(all_dist[1][0]), norm.fit(   all_dist[1][1])]
    ]
    
    # plotting
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    axes[0,0].hist(all_dist[0][0], bins=300)
    axes[0,1].hist(all_dist[0][1], bins=300, range=(0,8))
    axes[1,0].hist(all_dist[1][0], bins=300, range=(0,8))
    axes[1,1].hist(all_dist[1][1], bins=300)
    
    axes[0,0].plot()
    axes[0,1].plot()
    axes[1,0].plot()
    axes[1,1].plot()
    
    axes[0,0].set_title('Normal: mean={:.3f} | std={:.3f}'.format(
        all_mean[0][0], all_std[0][0]))
    axes[0,1].set_title('Lognormal: mean={:.3f} | std={:.3f}'.format(
        all_mean[0][1], all_std[0][1]))
    axes[1,0].set_title('Normal to LogNormal: mean={:.3f} | std={:.3f}'.format(
        all_mean[1][0], all_std[1][0]))
    axes[1,1].set_title('Lognormal to Normal: mean={:.3f} | std={:.3f}'.format(
        all_mean[1][1], all_std[1][1]))
    
    expo = np.random.exponential(1, size=10000)
    
    plt.show()

if __name__ == '__main__':
    ####################
    # JUST FOR TESTING #
    ####################
    
    from scipy.stats import norm, lognorm
    
    mean, std = 8, 3
    x = 5
    
    ### for individual statistical values ###
    value_N    = norm(mean, std).pdf(x)
    value_logN = lognorm(scale=np.exp(mean), s=std).pdf(np.log(x))
    
    print('\nOrigin\n======')
    print('- Normal Value    :', value_N)
    print('- LogNormal Value :', value_logN)
    
    new_value_N    = lognorm(mean, std).pdf(np.exp(value_logN))*value_logN
    new_value_logN = value_N
    
    print('\nTransformed\n===========')
    print('- Normal Value    :', new_value_N)
    print('- LogNormal Value :', new_value_logN)
    
    
    