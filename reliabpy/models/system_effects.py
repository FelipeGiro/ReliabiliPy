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

class System_of_Subsystems:
    """
    System of Subsystems
    ====================

    Separate all systems components in k-out-of-n susbsystems according to an 
    assingment vector. Subsystems are in series.

    Parameters:
    -----------
    assignments : array
        array with subsystem assignment of each component.
    k_list : array
        list of k values of each subsystem.
    """
    def __init__(self, assignments, k_list):
        self.assignments = np.array(assignments)
        self.k_list = np.array(k_list)
        self.zones = np.unique(self.assignments)

    def compute_system_pf(self, pf_list):
        """
        Compute system P_f
        ==================

        Compute system probability of failure.

        Parameters:
        -----------
        pf_list : array
            list of probabilities of failure of each component.
        
        Returns:
        --------
        pf_sys : float
            probability of failure of th eentire system
        """
        pf_list = np.array(pf_list)
        subsystem_pfs = []
        for zone, k in zip(self.zones, self.k_list):
            zone_pfs = pf_list[self.assignments == zone]
            subsystem_pfs.append(comp_k_out_of_n(zone_pfs, k))
        return comp_k_out_of_n(subsystem_pfs, len(subsystem_pfs))
