import numpy as np

def get_PoD_settings(quality):
    '''
    Get Probability of detection settings
    =====================================

    Source: 
    -------
    DNVGL-RP-C210_RBI - Section 11
    Table 11-1 PoD curves for EC, MPI, ACFM

    Parameter:
    ----------
    quality : str
        inspection paramters. It can be "good", "normal", or "bad".

    '''

    if quality == "good":
        return {'X0':0.40, 'b':1.43}
    elif quality == "normal":
        return {'X0':0.45, 'b':0.90}
    elif quality =="bad":
        return {'X0':1.16, 'b':0.90}


def PoD(a, X0, b):
    return 1.0 - (1.0/(1.0 + (a/X0)**b))