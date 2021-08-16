import numpy as np

class Probability_of_Detection:
    @staticmethod
    def get_settings(quality, inverse=False):
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

        if inverse:
            function = Probability_of_Detection.inv_function
        else:
            function = Probability_of_Detection.function

        if quality == "good":
            return {'X0':0.40, 'b':1.43}, function
        elif quality == "normal":
            return {'X0':0.45, 'b':0.90}, function
        elif quality =="bad":
            return {'X0':1.16, 'b':0.90}, function

    def function(a, X0, b):
        return 1.0 - (1.0/(1.0 + (a/X0)**b))

    def inv_function(P, X0, b):
        return X0*(1.0/(1.0 - P) - 1.0)**(1.0/b)