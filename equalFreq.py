import numpy as np
import logging
import math
import time
import math
from sortedcontainers import SortedKeyList
from operator import add
from operator import itemgetter

_nb_counter = []
_start_counter = []
_start_time_counter = []
_deltatime_counter = []
_NumberOfCounters=25
for i in range(0,_NumberOfCounters):
    _nb_counter.append(0)
    _start_counter.append(False)
    _start_time_counter.append(time.time())
    _deltatime_counter.append(0)


def start_counter(i):
    _nb_counter[i]=_nb_counter[i]+1
    _start_counter[i]=True
    _start_time_counter[i]=time.time()  
    
def stop_counter(i):
    _start_counter[i]=False
    diff = time.time()  - _start_time_counter[i]
    _deltatime_counter[i] = _deltatime_counter[i]+diff
    _start_time_counter[i]=time.time()


def generate_EqualFreqSteps(Y, n_bins):
    """
    Create thresholds to split the goal variable into n_bins containing the same number of goal variables.
    :param Y: np.array or list : The goal variable
    :param n_bins: int : The number of bins to create
    :return: list[int] : The (n_bins - 1) thresholds
    """
    thresholds_list = []
    
    start_counter(4)
    sorted_values = sorted(Y.reshape(-1))
    n = len(sorted_values)
    stop_counter(4)

    # Let's define the list of the possible thresholds :
    start_counter(5)
    unique_sorted_values = np.unique(sorted_values)

    possible_thresholds = unique_sorted_values[:-1] + (unique_sorted_values[1:] - unique_sorted_values[:-1])/2
    
    stop_counter(5)
    
    for bin_index in range(1, n_bins + 1):
        
        ideal_bin_index = bin_index * (n / (n_bins + 1))
        if bin_index <= (n_bins + 1) / 2.0:
            ideal_bin_index = math.floor(ideal_bin_index)
        else:
            ideal_bin_index = math.ceil(ideal_bin_index)
        ideal_bin_position = sorted_values[ideal_bin_index]
        
        start_counter(6)
        # Let's find the closest possible thresholds to the ideal bin position
        tmp_difference = np.abs(possible_thresholds - ideal_bin_position)
        stop_counter(6)
        
        start_counter(8)
        val=np.amin(tmp_difference)
        min_value_indexes = np.argwhere(tmp_difference == val).flatten().tolist()
        stop_counter(8)
        if bin_index < (n_bins + 1) / 2.0:
            closest_bin_position = possible_thresholds[min_value_indexes[0]]
        else:
            closest_bin_position = possible_thresholds[min_value_indexes[-1]]

        thresholds_list.append(closest_bin_position)
    
    start_counter(7)
    thresholds_list = np.unique(thresholds_list)
    stop_counter(7)
    
    if len(thresholds_list) < n_bins:
        logging.warning(str((n_bins - len(thresholds_list))) + " bin" + ('s are' if (n_bins - len(thresholds_list)) > 1 else ' is') + " empty because of the bin generation strategy.")

    return thresholds_list
