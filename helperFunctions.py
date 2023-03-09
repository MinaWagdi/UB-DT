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

from stats_rissanen import universal_code_natural_numbers
from stats_rissanen import log_2_star
# import MDL_Criteria

logger = logging.getLogger('causalml')
logging.basicConfig(level=logging.INFO)

from scipy.special import comb
import scipy.special as sc

from operator import itemgetter
import operator
import bisect
import stats_rissanen
from sklift.metrics import (
    uplift_at_k, uplift_auc_score, qini_auc_score, weighted_average_uplift
)


_Log_Fact_Table = []
binomialFunctionAccessCount=0

def log_fact(n):
#     start_counter(1)
   
    """
    Compute log(fact(n))
    :param n:
    :return: value of log(fact(n))
    """
    # print("\t\t asked for log_fact(n=%d)"%n)
    # Use approximation for large n
    if n > 1e6:
        # print('\t\t Using approximation : res=%.f' %log_fact_approx(n))
        return log_fact_approx(n)
    # computation of values, tabulation in private array
    else:
        
        size = len(_Log_Fact_Table)
        if n >= size:
            print("SHOULD GO IN HERE only once when n is equal to the length of the data n is ",n)
            if size == 0:
                _Log_Fact_Table.append(0)
                size = len(_Log_Fact_Table)
            while size <= n:
                # print('%d<=%d' %(size,n))
                _Log_Fact_Table.append(log(size) + _Log_Fact_Table[size - 1])
                size = size + 1
        return _Log_Fact_Table[n]

def log_binomial_coefficient(n: int, k: int):
    """
    Computes the log of the binomial coefficient  (n
                                                   k)
    (log of the total number of combinations of k elements from n)
    :param n: Total number of elements
    :param k: Number of selected elements
    :return:
    """
    
    global _Log_Fact_Table
    
    global binomialFunctionAccessCount
    binomialFunctionAccessCount+=1
    try:
        nf = _Log_Fact_Table[n]
        kf = _Log_Fact_Table[k]
        nkf = _Log_Fact_Table[n - k]
    except:
        print("length of log_fact table is ",len(_Log_Fact_Table))
        print("n is ",n)
        raise
    return (nf - nkf) - kf

     
