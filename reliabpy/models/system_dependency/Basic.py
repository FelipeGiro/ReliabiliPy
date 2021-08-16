import numpy as np

def comp_k_out_of_n(pf, k):
    '''
    COMPUTE PROBABILITY OF FAILURE OF THE SYSTEM RELIABILITY
    ========================================================
    
    The system is good if and only if at least K of its n components are good.
    
    => Model assumptions:
        1. Components and the system are 2-state (eg, good or bad)
        2. Component states are statiscally independent
        3. The system is good if and only if at least k of its n components 
           are good
    
    Source: IEEE TRANSACTIONS ON RELIABILITY, VOL. R-33, NO. 4, OCTOBER 1984
            page 321, R.E. Barlow and K.D. Heidtmann
    
    Parameters
    ----------
    pf : array-like
        the probability of failure of each element
    k : integer
        the number of components necessary to make the system work

    Retuns
    ------
    PF_sys : float
        system probability of failure
    '''
    
    pf = np.array(pf)
    n = pf.size
    
    # in case of no element
    if n == 0:
        return np.nan
    
    # n = len(pf)
    # k = ncomp-1
    PF_sys = np.zeros(1)
    nk = n-k
    m = k+1
    A = np.zeros(m+1)
    A [1] = 1
    L = 1
    for j in range(1,n+1):
        h = j + 1
        Rel = 1-pf[j-1]
        if nk < j:
            L = h - nk
        if k < j:
            A[m] = A[m] + A[k]*Rel
            h = k
        for i in range(h, L-1, -1):
            A[i] = A[i] + (A[i-1]-A[i])*Rel
    PF_sys = 1-A[m]
    return PF_sys   