#
#
# Hans-Rainer Kloeckner
#
# MPIfR 2026
#
# Provide some statistics function usfull to handle 
# SKAMPI data sets
#
#


import numpy as np
from astropy.stats import mad_std, median_absolute_deviation




def data_stats(data,stats_type='mean',sigma=-1,spwd=1):
    """
    return mean and derivation of the input data
    that can be sub-divided
    """

    if spwd > 1:
        data = np.array_split(data,spwd)
        axis = 1
    else:
        axis = 0

    if stats_type == 'madmadmedian':
        #
        data_mean      = median_absolute_deviation(data,axis=axis)
        data_std       = mad_std(data,axis=axis)

    elif stats_type == 'madmean':
        #
        data_mean      = np.mean(data,axis=axis) 
        data_std       = mad_std(data,axis=axis)

    elif stats_type == 'madmedian':
        #
        data_mean      = np.median(data,axis=axis) 
        data_std       = mad_std(data,axis=axis)

    elif stats_type == 'median':
        data_mean      = np.median(data,axis=axis)
        data_std       = np.std(data,axis=axis)

    elif stats_type == 'quantile':
        data_low       = np.quantile(data, 0.02)
        data_up        = np.quantile(data, 0.98)
        #
        data_mean      = data_low
        data_std       = data_up
        stats_type     = 'q_low, q_high '+stats_type
    else:
        data_mean      = np.mean(data,axis=axis)
        data_std       = np.std(data,axis=axis)

    if sigma == -1:
        return data_mean, data_std, stats_type+' [mean, std]'
    else:
        if stats_type != 'quantile':
            return data_mean - sigma * data_std, data_mean + sigma * data_std, stats_type+' [low, up boundary]'
        else:
            return data_low,data_up, stats_type+' [low, up boundary]'
