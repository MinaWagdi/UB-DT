import warnings
# warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import rcParams

from operator import itemgetter

from math import log, pi, pow, exp, lgamma, sqrt
import numpy as np
from typing import Callable
from math import ceil, floor
from operator import itemgetter
from sortedcontainers import SortedKeyList
from operator import add
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import logging
import time
import os
import pickle
import sys
from equalFreq import generate_EqualFreqSteps
# import MDL_Criteria

logger = logging.getLogger('causalml')
logging.basicConfig(level=logging.INFO)

from scipy.special import comb
import scipy.special as sc

from operator import itemgetter
import operator
import bisect


dvLnStar=[]
dvC0Max=[]

def universal_code_natural_numbers(k: int):
    """
    Compute the universal code for integers presented by Rissanen in
    'A Universal Prior for Integers and Estimation by Minimum Description Length', Rissanen 1983
    :param k:
    :return:
    """
    dC0 = 2.86511  # First value computed following the given estimation formula, as e(3)=65536 + d_log2^5 / (1-d_log2)
    d_log2 = log(2.0)

    if k < 1:
        raise ValueError("Universal code is defined for natural numbers over 1")
    else:
        # Initialize code length cost to log_2(dC0)
        d_cost = log(dC0) / d_log2

        # Add log_2*(k)
        d_cost += log_2_star(k)

        # Go back to the natural log
        d_cost *= d_log2

        return d_cost    
def log_2_star(k: int):
    """
    Computes the term log_2*(k)=log_2(k) + log_2(log_2(k)) + ...  of Rissanen's code for integers
    so long as the terms are positive
    :param k:
    :return:
    """
    d_log2 = log(2.0)
    d_cost = 0.0
    d_logI = log(1.0 * k) / d_log2

    if k < 1:
        raise ValueError("Universal code is defined for natural numbers over 1")
    else:
        while d_logI > 0:
            d_cost += d_logI
            d_logI = log(d_logI) / d_log2

        return d_cost    


    
    
def ComputeLnStarAndC0MaxTables():
    d_log2 = log(2.0) 
    dLogI=0
    i=0
    dCost=0

    dvLnStar.append(0)
    dvC0Max.append(1)
    for i in range(2000):
        dCost = 0 
        dLogI = log(1.0 * (i + 1)) / d_log2 
        while (dLogI > 0):
            dCost += dLogI 
            dLogI = log(dLogI) / d_log2 
        dvLnStar.append(dCost) 
        dvC0Max.append(dvC0Max[i-1] + 2**-dCost) 

def C0Max(nMax):
    d_log2 = log(2.0) 
    nE3 = 65536 
    dC0Max=0

    if nMax < 1:
        raise ValueError("nMax should be over 1")

    if len(dvLnStar) == 0 or len(dvC0Max) == 0:
        ComputeLnStarAndC0MaxTables() 

    if nMax < len(dvC0Max):
        dC0Max = dvC0Max[-1]
    else:
        dC0Max = dvC0Max[-1]
        
        if len(dvC0Max)< nE3:
            if (nMax < nE3):
                dC0Max += (d_log2**4) * (log(log(log(log(nMax * 1.0) / d_log2) / d_log2) / d_log2) / d_log2 - log(log(log(log(len(dvC0Max)* 1.0) / d_log2) / d_log2) / d_log2) / d_log2) 
            else:
                dC0Max += (d_log2**4) * (1 - log(log(log(log(len(dvC0Max) * 1.0) / d_log2) / d_log2) / d_log2) / d_log2) + pow(d_log2, 5) * log(log(log(log(log(nMax * 1.0) / d_log2) / d_log2) / d_log2) / d_log2) / d_log2 
        else:
            dC0Max += (d_log2**4) * (log(log(log(log(log(nMax * 1.0) / d_log2) / d_log2) / d_log2) / d_log2) / d_log2 - log(log(log(log(log(len(dvC0Max) * 1.0) / d_log2) / d_log2) / d_log2) / d_log2) / d_log2) 
	
    return dC0Max
    

def BoundedNaturalNumbersUniversalCodeLength(n,nMax):
    d_log2 = log(2.0)
    d_cost = 0.0
    
    if n<1:
        raise ValueError("Universal code is defined for natural numbers over 1")
    else:
        dCost = log(C0Max(nMax))/d_log2
        
        dCost += log_2_star(n)

        dCost *= d_log2
        return dCost

