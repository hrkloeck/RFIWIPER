# HRK 2024
#
# Hans-Rainer Kloeckner
# hrk@mpifr-bonn.mpg.de 
#
#
# RFI Mitigation Lib for SKAMPI SCANS
# 
# To be consistent with numpy masked arrays 
# data that is bad will be masked asd True or 1
# 
# --------------------------------------------------------------------


import numpy as np
import numpy.ma as ma
from copy import deepcopy
from scipy.signal import convolve2d,correlate
from scipy import stats
import time
# import heat from submodule
import sys
sys.path.append("heat/")
import heat as ht


def data_stats(data,stats_type='mean',accur=100):
    """
    return mean and deviation of the input data
    """

    if stats_type == 'madmadmedian':
        #
        from astropy.stats import mad_std, median_absolute_deviation
        #
        data_mean      = median_absolute_deviation(data)
        data_std       = mad_std(data)

    elif stats_type == 'madmean':
        #
        from astropy.stats import mad_std
        #
        data_mean      = np.mean(data) 
        data_std       = mad_std(data)

    elif stats_type == 'madmedian':
        #
        from astropy.stats import mad_std
        #
        data_mean      = np.median(data) 
        data_std       = mad_std(data)

    elif stats_type == 'median':
        data_mean      = np.median(data)
        data_std       = np.std(data)

    elif stats_type == 'kdemean':
        data_mean,data_std = kdemean(data,accuracy=1000)

    elif stats_type == 'plothist':
        #
        import matplotlib.pyplot as plt
        #
        # Just as a tool to help debugging
        #
        sigma  = 3
        cuts1  = np.mean(data) - sigma * np.std(data)
        cuts2  = np.mean(data) + sigma * np.std(data)

        print('%e'%cuts1,'--','%e'%cuts2)
        
        fig, ax = plt.subplots()
        yvalues,bins,patch = ax.hist(data,density=True,bins=accur)
        plt.plot([cuts1,cuts1],[min(yvalues),max(yvalues)],'-,','b')
        plt.plot([cuts2,cuts2],[min(yvalues),max(yvalues)],'-,','b')
        plt.show()
        #
        sys.exit(-1)

    else:
        data_mean      = np.mean(data)
        data_std       = np.std(data)


    return data_mean, data_std, stats_type

def data_stats_vect(data,padded_elements, stats_type='mean',accur=100):
    """
    vectorized calculation of mean and deviation over the binned data

    Parameters
    ----------
    data : np.ndarray
        input data, shape (n, m) with n = number of bins, m = number of channels per bin
    """
    if data.ndim == 1:
        raise ValueError("Input data must be 2D, (bins, channels)")

    if stats_type == 'madmadmedian':
        #
        from astropy.stats import mad_std, median_absolute_deviation
        #
        data_mean      = median_absolute_deviation(data, axis=1)
        data_std       = mad_std(data, axis=1)
        # last element affected by padding, correct
        data_mean[-1] = median_absolute_deviation(data[-1, :-padded_elements])
        data_std[-1] = mad_std(data[-1, :-padded_elements])

    elif stats_type == 'madmean':
        #
        from astropy.stats import mad_std
        #
        data_mean      = np.mean(data, axis=1) 
        data_std       = mad_std(data, axis=1)
        # last element affected by padding, correct
        data_mean[-1] = np.mean(data[-1, :-padded_elements])
        data_std[-1] = mad_std(data[-1, :-padded_elements])

    elif stats_type == 'madmedian':
        #
        from astropy.stats import mad_std
        #
        data_mean      = np.median(data, axis=1) 
        data_std       = mad_std(data, axis=1)
        # last element affected by padding, correct
        data_mean[-1] = np.median(data[-1, :-padded_elements])
        data_std[-1] = mad_std(data[-1, :-padded_elements])

    elif stats_type == 'median':
        data_mean      = np.median(data, axis=1)
        data_std       = np.std(data, axis=1)
        # last element affected by padding, correct
        data_mean[-1] = np.median(data[-1, :-padded_elements])
        data_std[-1] = np.std(data[-1, :-padded_elements])

    elif stats_type == 'kdemean':
        raise NotImplementedError("Vectorized Kernel Density Estimation mean not implemented yet")
        #data_mean,data_std = kdemean(data,accuracy=1000)

    elif stats_type == 'plothist':
        #
        import matplotlib.pyplot as plt
        #
        # Just as a tool to help debugging
        #
        sigma  = 3
        data_mean = np.mean(data, axis=1)
        data_std = np.std(data, axis=1)
        # last element affected by padding, correct
        data_mean[-1] = np.mean(data[-1, :-padded_elements])
        data_std[-1] = np.std(data[-1, :-padded_elements])
        cuts1  = data_mean - sigma * data_std
        cuts2  = data_mean + sigma * data_std

        print('%e'%cuts1,'--','%e'%cuts2)
        
        fig, ax = plt.subplots()
        yvalues,bins,patch = ax.hist(data.flatten(),density=True,bins=accur)
        plt.plot([cuts1,cuts1],[min(yvalues),max(yvalues)],'-,','b')
        plt.plot([cuts2,cuts2],[min(yvalues),max(yvalues)],'-,','b')
        plt.show()
        #
        sys.exit(-1)

    else:
        data_mean      = np.mean(data, axis=1)
        data_std       = np.std(data, axis=1)
        # last element affected by padding, correct
        data_mean[-1] = np.mean(data[-1, :-padded_elements])
        data_std[-1] = np.std(data[-1, :-padded_elements])        

    return data_mean, data_std, stats_type




def boundary_mask_data(data,reference_data,sigma,stats_type='mean',do_info=False):
    """
    use upper and lower thresholds to mask out data
    data is a unflagged (e.g. compressed dataset)
    reference_data original data

    """

    # determine the mean and std of the data
    #
    data_mean,data_std,stats_type = data_stats(data,stats_type)

    # selecing all data within the boundaries ion
    #
    select = np.logical_and(reference_data > data_mean - sigma * data_std, reference_data < data_mean + sigma * data_std)

    # Note in NUMPY MASKED arrays a bolean value of True (1) is considered to be masked out
    #
    # default all data is bad
    #
    data_shape    = np.array(reference_data).shape
    mask          = np.ones(data_shape)

    # so good data is indicated by zero 
    #
    mask[select]  = 0

    if do_info:
        print('data ',np.cumprod(data_shape)[-1],' markes as bad ',np.cumprod(data_shape)[-1]-np.count_nonzero(select))
        
    return mask.astype(bool)



def smooth_kernels(smk_type):
    """
    """
    # ----------------------------------------------
    #
    # here are some examples of kernels for an ambitions user that may want to play with it
    #
    if smk_type == 'box':
        kernel      = [[1,1,1],[1,1,1],[1,1,1]]        # boxcar
    if smk_type == 'cross':
        kernel      = [[0,1,0],[1,1,1],[0,1,0]]        # cross
    if smk_type == 'robx':
        kernel      = [[1,0],[0,-1]]                   # Roberts operator di/dx 
    if smk_type == 'roby':
        kernel      = [[0,1],[-1,0]]                   # Roberts operator di/dy
    if smk_type == 'scharrx':
        kernel      = [[-3,0,3],[-10,0,10],[-3,0,3]]   # Scharr operator di/dx
    if smk_type == 'scharry':
        kernel     = [[3,10,3],[0,0,0],[-3,-10,-3]]   # Scharr operator di/dy
    if smk_type == 'sobelx':
        kernel      = [[-1,0,1],[-2,0,2],[-1,0,1]]     # Sobel operator di/dx
    if smk_type == 'sobely':
        kernel      = [[1,2,1],[0,0,0],[-1,-2,-1]]     # Sobel operator di/dy
    if smk_type == 'canny':
        kernel       = [[2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]] 
    if smk_type == 'prewittx':        
        kernel   = [[-1,0,1],[-1,0,1],[-1,0,1]]   # Prewitt operator di/dx
    if smk_type == 'prewitty':
        kernel   = [[1,1,1],[0,0,0],[-1,-1,-1]]   # Prewitt operator di/dy

    if smk_type == 'laplace':
        kernel   = [[0,1,0],[1,-4,1],[0,1,0]]   # Laplace filter


    #ddxxfilter  = [[1,-2,1]]                    # differential
    #ddyyfilter  = [[1],[-2],[1]]                # differential 
    #dddxyfilter = [[-1/4.,0,1/4.],[0,0,0],[1/4.,0,-1/4.]]  # differential 
    #

    return kernel



def kdemean(x,accuracy=1000):
    """
     use the Kernel Density Estimation (KDE) to determine the mean
    
    (http://jpktd.blogspot.com/2009/03/using-gaussian-kernel-density.html )
    """
    from scipy.stats import gaussian_kde
    from numpy import linspace,min,max,std,mean
    from math import sqrt,log
    
    if mean(x) == std(x):
            print('kde mean = std')
            return(mean(x),std(x))

    max_range = max(np.abs([min(x),max(x)]))

    # create instance of gaussian_kde class
    gk     = gaussian_kde(x)

    vra    = linspace(-1*max_range,max_range,accuracy)
    vraval = gk.evaluate(vra)

    # get the maximum
    #
    x_value_of_maximum = vra[np.argmax(vraval)]

    # Devide data
    difit = vraval / max(vraval)
    #
    # and select values from 0.5
    sel = difit >= 0.4999

    idx_half_power = list(difit).index(min(difit[sel]))

    if idx_half_power >= accuracy -1:
        return(mean(x),std(x))

    delta_accuracy = max([abs(vra[idx_half_power-1] - vra[idx_half_power]),\
                              abs(vra[idx_half_power+1] - vra[idx_half_power])])

    fwhm = abs(x_value_of_maximum - vra[idx_half_power])


    # factor 2 is because only one side is evaluated
    sigma = 2*fwhm/(2*sqrt(2*log(2)))

    # safety net
    # is the KDE is not doing a good job
    #
    if sigma > std(x):
        return(mean(x),std(x))

    return(x_value_of_maximum,abs(sigma)+delta_accuracy)



def boundary_range(xdata,ydata,usedbinning,ydata_mask,bound_sigma=3,stats_type='mean',smooth_kernel=31):
    """
    provide a sigma boundary 
    """
    import copy

    from time import process_time

    # To get the upper and lower boundaries we used binned data and its statistics
    # 
    stats_x_data,stats_x_datastd             = checkerstats(xdata,usedbinning,stats_type)
    stats_y_data,stats_y_datastd,stats_mask  = checkerstats_sel(ydata,usedbinning,ydata_mask,stats_type)
        

    # determine a sigma boundary
    #
    # freq values
    x_boundary_values = stats_x_data
    x_boundary_values = np.insert(x_boundary_values,0,xdata[0])
    x_boundary_values = np.insert(x_boundary_values,-1,xdata[-1])
    #
    # VECTORIZED X BOUNDARY VALUES
    # x range
    #
    # x_boundary_values = np.zeros((len(stats_x_data)+2,), dtype = np.array(stats_x_data).dtype)
    # x_boundary_values[0] = xdata[0]
    # x_boundary_values[-1] = xdata[-1]
    # x_boundary_values[1:-1] = stats_x_data
       
    # amp values
    y_boundary_values_up  = stats_y_data + bound_sigma * stats_y_datastd
    #
    y_boundary_values_up  = np.insert(y_boundary_values_up,0,y_boundary_values_up[0])
    y_boundary_values_up  = np.insert(y_boundary_values_up,-1,y_boundary_values_up[-1])

    y_boundary_values_low = stats_y_data - bound_sigma * stats_y_datastd
    #
    y_boundary_values_low = np.insert(y_boundary_values_low,0,y_boundary_values_low[0])
    y_boundary_values_low = np.insert(y_boundary_values_low,-1,y_boundary_values_low[-1])

    # Here check on the boundary 
    # 
    #start = time.perf_counter()
    boundary_stats_up   = data_stats(y_boundary_values_up,stats_type='median',accur=100)
    boundary_stats_low  = data_stats(y_boundary_values_low,stats_type='median',accur=100)
    #end = time.perf_counter()
    #print(f"Time for data_stats : {end - start:0.7f} seconds")
    #
    #start = time.perf_counter()
    if usedbinning > 10:

        if boundary_stats_up[0] != boundary_stats_up[1]:

            bound_select         = np.logical_or(y_boundary_values_up >= boundary_stats_up[0] + bound_sigma * boundary_stats_up[1],\
                                                     y_boundary_values_low <= boundary_stats_low[0] - bound_sigma * boundary_stats_low[1])  # True for bad data
        else:
            bound_select         = np.zeros(len(y_boundary_values_low)).astype(bool)
        
    else:
            bound_select         = np.zeros(len(y_boundary_values_low)).astype(bool)
    #end = time.perf_counter()
    #print(f"Time for bound_select : {end - start:0.7f} seconds")

    # case there are also bins the have been marked by 
    # checkerstats_selcheckerstats_sel
    #
    if max(stats_mask) > 0:
        stats_mask    = stats_mask.astype(bool)
        stats_mask    = np.insert(stats_mask,0,bound_select[0])
        stats_mask    = np.insert(stats_mask,len(stats_mask),bound_select[-1])
        bound_select  = np.logical_or(bound_select,stats_mask.astype(bool))


    # Interpolate and convolve the boundary region 
    # of the data and use this to mark bad data
    #
    interp_boundary_up         = np.interp(xdata,np.array(x_boundary_values)[np.invert(bound_select)],\
                                               np.array(y_boundary_values_up)[np.invert(bound_select)])
    # check timeing of process
    #
    interp_boundary_low        = np.interp(xdata,np.array(x_boundary_values)[np.invert(bound_select)],\
                                               np.array(y_boundary_values_low)[np.invert(bound_select)])
    #
    #start = time.perf_counter()
    if smooth_kernel > 3:
        boundary_up                = convolve_1d_data(interp_boundary_up,smooth_type='wiener',smooth_kernel=smooth_kernel)
        boundary_low               = convolve_1d_data(interp_boundary_low,smooth_type='wiener',smooth_kernel=smooth_kernel)
    else:
        boundary_up  = interp_boundary_up
        boundary_low = interp_boundary_low
    #end = time.perf_counter()
    #print(f"Time for boundary_up, boundary_low : {end - start:0.7f} seconds")
    # select on the boundary data
    #
    grad_select         = np.logical_or(ydata >= boundary_up,ydata <= boundary_low)  # True for bad data


    return grad_select,boundary_up,boundary_low




def flag_spec_by_smoothing(fg_spec,freq,cleanup_spec_mask,splitting,kernel_sizes,kernel_sequence_type,smooth_type,usedbinning,bound_sigma,stats_type,smooth_bound_kernel,clean_bins,idx=0,mtque=None,njobs=1):
    """
    specially for doing single azimuthe scans
    Parameters
    ----------
    fg_spec : np.ndarray
        1D array of the spectrum to be cleaned
    freq : np.ndarray
        1D array of the frequencies
    cleanup_spec_mask : np.ndarray
        1D array of the mask for the spectrum
    splitting : list
        list of indices to split the spectrum into subarrays
    kernel_sizes : list
        list of kernel sizes for smoothing
    smooth_type : list
        list of smoothing types
    usedbinning : list
        list of binning sizes
    bound_sigma : list
        list of sigma values for boundary range
    stats_type : list
        list of statistics types
    smooth_bound_kernel : list
        list of kernel sizes for boundary smoothing
    idx : int
        index of the spectrum
    mtque : multiprocessing.Queue
        queue for multiprocessing
    njobs : int
        number of jobs for multiprocessing
    """
    from time import process_time

    if np.sum(cleanup_spec_mask) != len(cleanup_spec_mask):

        # do the smooth flagging
        #
        grad_select = np.array([]).astype(bool)
        for sp in range(len(splitting)-1):

            if splitting[sp+1] == -1:
                sp_freq       = freq[splitting[sp]:]
                sp_data       = np.array(fg_spec)[splitting[sp]:]
                sp_data_mask  = cleanup_spec_mask[splitting[sp]:]
            else:
                sp_freq       = freq[splitting[sp]:splitting[sp+1]]
                sp_data       = np.array(fg_spec)[splitting[sp]:splitting[sp+1]]
                sp_data_mask  = cleanup_spec_mask[splitting[sp]:splitting[sp+1]]


            if (len(sp_freq)%usedbinning[sp]) != 0:
                print('\nCAUTION \n\nThe setting produced sub-arrays with no equal sizes.')
                print('Subrray size is and used binning parameter ',len(sp_freq),usedbinning[sp])
                print('Change usedbinning parameter in main programme\n\n')
                sys.exit(-1)


            grad_selectsp = flag_smoothing(sp_freq,sp_data,sp_data_mask,smooth_type=smooth_type[sp],kernel_sizes=kernel_sizes[sp],kernel_sequence_type=kernel_sequence_type[sp],\
                                               usedbinning=usedbinning[sp],bound_sigma=bound_sigma[sp],stats_type=stats_type[sp],\
                                               smooth_bound_kernel=smooth_bound_kernel[sp])
            print("In flag_spec_by_smoothing: grad_selectsp, grad_select = ", grad_selectsp.shape, grad_select.shape)
            grad_select = np.append(grad_select,grad_selectsp)


        # clean up process is of the order of 0.04 sec
        #
        print("In flag_spec_by_smoothing: grad_select = ", grad_select.shape)
        final_sp_mask = clean_up_1d_mask(grad_select,clean_bins,setvalue=True)

    else:

        final_sp_mask = cleanup_spec_mask.astype(bool)


    if mtque != None:
        resultdic =  {}
        resultdic[njobs] = [idx,final_sp_mask]
        mtque.put(resultdic)

    return final_sp_mask

def flag_spec_by_smoothing_ht(fg_spectra,freq,cleanup_spectra_mask,splitting,kernel_sizes,kernel_sequence_type,smooth_type,usedbinning,bound_sigma,stats_type,smooth_bound_kernel,clean_bins,idx=0):
    """
    specially for doing  azimuth scans in batch

    Parameters
    ----------
    fg_spectra : ht.DNDarray
        2D array of the spectra to be cleaned, shape (n, m) with n = timestamps, m = number of channels per timestamp
    freq : np.ndarray or ht.DNDarray
        1D array of the frequencies
    cleanup_spectra_mask : ht.DNDarray
        2D array of the mask for the spectra, shape (n, m) with n = timestamps, m = number of channels per timestamp
    splitting : list
        list of indices to split the spectrum into subarrays
    kernel_sizes : list
        list of kernel sizes for smoothing
    smooth_type : list
        list of smoothing types
    usedbinning : list
        list of binning sizes
    bound_sigma : list
        list of sigma values for boundary range
    stats_type : list
        list of statistics types
    smooth_bound_kernel : list
        list of kernel sizes for boundary smoothing
    idx : int
        index of the spectrum
    """
    from time import process_time

    need_flagging = ht.sum(cleanup_spectra_mask, axis=1) != cleanup_spectra_mask.shape[1]
    fg_spectra = fg_spectra[need_flagging]
    cleanup_spectra_mask = cleanup_spectra_mask[need_flagging]
    # allocate grad_select for all spectra
    grad_select = ht.zeros(fg_spectra.shape, split=0, device=fg_spectra.device) 

    for sp in range(len(splitting)-1):
        if splitting[sp+1] == -1:
            sp_slice = slice(splitting[sp], None)
            # sp_freq = freq[splitting[sp]:]
            # sp_data = fg_spectra[:, splitting[sp]:]
            # sp_data_mask = cleanup_spectra_mask[splitting[sp]:]
        else:
            sp_slice = slice(splitting[sp], splitting[sp+1])
            # sp_freq = freq[splitting[sp]:splitting[sp+1]]
            # sp_data = fg_spectra[:, splitting[sp]:splitting[sp+1]]
            # sp_data_mask = cleanup_spectra_mask[:, splitting[sp]:splitting[sp+1]]

        sp_freq = freq[sp_slice]
        sp_data = fg_spectra[:, sp_slice]
        sp_data_mask = cleanup_spectra_mask[:, sp_slice]

        if (len(sp_freq)%usedbinning[sp]) != 0:
            print('\nCAUTION \n\nThe setting produced sub-arrays with no equal sizes.')
            print('Subrray size is and used binning parameter ',len(sp_freq),usedbinning[sp])
            print('Change usedbinning parameter in main programme\n\n')
            sys.exit(-1)

        grad_select[:, sp_slice] = flag_smoothing_ht(sp_freq, sp_data, sp_data_mask, smooth_type=smooth_type[sp], kernel_sizes=kernel_sizes[sp], kernel_sequence_type=kernel_sequence_type[sp],\
                                            usedbinning=usedbinning[sp], bound_sigma=bound_sigma[sp], stats_type=stats_type[sp], smooth_bound_kernel=smooth_bound_kernel[sp])

    final_spectra_mask = clean_up_1d_mask(grad_select, clean_bins, setvalue=True)
    # if np.sum(cleanup_spectra_mask) != len(cleanup_spectra_mask):

    #     # do the smooth flagging
    #     #
    #     grad_select = np.array([]).astype(bool)
    #     for sp in range(len(splitting)-1):

    #         if splitting[sp+1] == -1:
    #             sp_freq       = freq[splitting[sp]:]
    #             sp_data       = np.array(fg_spec)[splitting[sp]:]
    #             sp_data_mask  = cleanup_spec_mask[splitting[sp]:]
    #         else:
    #             sp_freq       = freq[splitting[sp]:splitting[sp+1]]
    #             sp_data       = np.array(fg_spec)[splitting[sp]:splitting[sp+1]]
    #             sp_data_mask  = cleanup_spec_mask[splitting[sp]:splitting[sp+1]]


    #         if (len(sp_freq)%usedbinning[sp]) != 0:
    #             print('\nCAUTION \n\nThe setting produced sub-arrays with no equal sizes.')
    #             print('Subrray size is and used binning parameter ',len(sp_freq),usedbinning[sp])
    #             print('Change usedbinning parameter in main programme\n\n')
    #             sys.exit(-1)


    #         grad_selectsp = flag_smoothing(sp_freq,sp_data,sp_data_mask,smooth_type=smooth_type[sp],kernel_sizes=kernel_sizes[sp],kernel_sequence_type=kernel_sequence_type[sp],\
    #                                            usedbinning=usedbinning[sp],bound_sigma=bound_sigma[sp],stats_type=stats_type[sp],\
    #                                            smooth_bound_kernel=smooth_bound_kernel[sp])

    #         grad_select = np.append(grad_select,grad_selectsp)


    #     # clean up process is of the order of 0.04 sec
    #     #
    #     final_sp_mask = clean_up_1d_mask(grad_select,clean_bins,setvalue=True)

    # else:

    #     final_sp_mask = cleanup_spec_mask.astype(bool)

    # return final_sp_mask


def batch_convolve_1d_data(data,smooth_type='hamming',smooth_kernel=3):
    """
    Vectorized version of convolve_1d_data. Uses PyTorch batch convolutions instead of scipy convolve, and Heat as a backend for parallelization.

    Parameters
    ----------
    data_2d : np.ndarray
        2D array of data to be smoothed, shape (n, m) with n = timestamps, m = number of channels per timestamp
    smooth_type : str
        type of smoothing kernel. Options are 'hamming', 'wiener'. Default is 'hamming', which convolves the 1D data with Hamming window of size `smooth_kernel`
    smooth_kernel : int
        size of the smoothing kernel. Default is 3    
    """
    import torch

    if smooth_type == 'hamming':
        sm_kernel = torch.hamming_window(smooth_kernel, dtype=data.larray.dtype)
        sm_kernel = ht.array(sm_kernel, device=data.device)
    elif smooth_type == 'wiener':
        raise NotImplementedError("Heat backend: Wiener smoothing not implemented yet")
    else:
        raise ValueError(f"Unknown smoothing type: {smooth_type}")
    
    # batch-convolve each timestamp with the smoothing kernel
    sm_data = ht.convolve(data, sm_kernel, mode='same') / sm_kernel.sum()

    return sm_data

def convolve_1d_data(data,smooth_type='hanning',smooth_kernel=3):
    """
    """
    from scipy.signal import wiener,gaussian,medfilt,convolve
    from scipy.signal.windows import hamming #, hanninng #hanning,convolve,hamming,gaussian,medfilt


    if smooth_type == 'hamming':
        sm_kernel = hamming(smooth_kernel)
        sm_data   = convolve(data,sm_kernel,mode='same') / sum(sm_kernel)

    elif smooth_type == 'gaussian':
        sm_kernel = gaussian(smooth_kernel,smooth_kernel)
        sm_data   = convolve(data,sm_kernel,mode='same') / sum(sm_kernel)

    elif smooth_type == 'median':
         sm_data = medfilt(data,smooth_kernel)

    elif smooth_type == 'wiener':
#         print("In convolve_1d_data: smooth_kernel = ", smooth_kernel)
         # TODO: implement vectorized version of wiener in Heat
         sm_data = wiener(data,smooth_kernel)

    elif smooth_type == 'kernelinput':
        sm_data   = convolve(data,smooth_kernel,mode='same')

    else:
        sm_data = deepcopy(data)

    return sm_data


def kernel_sequence(kernel_sizes,kernel_sequence_type='middle_fast'):
    """
    return kernel_sequence
    caution this impacts on the processing time and the
    quality of the resulting spectrum
    """

    kernel = []
    for i in range(1,kernel_sizes):
        #
        if kernel_sequence_type == 'middle_fast':
             if (3*i**2)%2 == 0:
                 kernel.append(3*i**2 + 1)
             else:
                 kernel.append(3*i**2)

        elif kernel_sequence_type == 'fast':
            kernel.append(2**i)

        else:
            # slow
            #
            # a step by step kernel generates a
            # cleaner spectrum, but takes 10 sec 
            # per time step
            #
            kernel.append(2 * i + 1)

    return kernel

def flag_smoothing(freq,spec,spec_mask,smooth_type='wiener',kernel_sizes=2,kernel_sequence_type='middle_fast',usedbinning=1,bound_sigma=4,stats_type='median',smooth_bound_kernel=31):
    """
    """
    import matplotlib.pyplot as plt
    from copy import copy

    from time import process_time

    grad_select     = copy(spec_mask)
    grad_select_org = copy(spec_mask)


    # set the kernel sequence
    #
    kernel = kernel_sequence(kernel_sizes,kernel_sequence_type)


    info_fg   = []
    info_fg_k = []

#    print("In flag_smoothing: len(kernel) = ", len(kernel), type(kernel))
#    print("In flag_smoothing: kernel = ", kernel)
    for k in kernel:
        
        grad_select_org = grad_select

        # smooth the original spectrum and subtract it from the
        # original dataset
        #
        #start = time.perf_counter()
        sm_data   = convolve_1d_data(spec,smooth_type=smooth_type,smooth_kernel=k)
        #end = time.perf_counter()
        #print(f"In flag_smoothing: convolve_1d_data : {end - start:0.7f} seconds")
        resi_data = spec - sm_data

        # generate a flag based on the boundary region
        #
        #start = time.perf_counter()
        grad_select,boundary_up,boundary_low = boundary_range(freq,resi_data,usedbinning,np.invert(grad_select),\
                                                                  bound_sigma=bound_sigma,stats_type=stats_type,\
                                                                  smooth_kernel=smooth_bound_kernel)
        #end = time.perf_counter()
        #print(f"In flag_smoothing: boundary_range : {end - start:0.7f} seconds")
        # and combine with the previous                                                          
        grad_select = np.logical_or(grad_select,grad_select_org)

    return grad_select

def flag_smoothing_ht(freq,spec,spec_mask,smooth_type='wiener',kernel_sizes=2,kernel_sequence_type='middle_fast',usedbinning=1,bound_sigma=4,stats_type='median',smooth_bound_kernel=31):
    """
    Flagging by smoothing the spectrum. Vectorized/batch version using Heat as a backend for parallelization.

    Parameters
    ----------
    freq : np.ndarray or ht.DNDarray
        1D array of the frequencies
    spec : ht.DNDarray
        2D array of the spectra to be cleaned, shape (n, m) with n = timestamps, m = number of channels per timestamp
    spec_mask : ht.DNDarray
        2D array of the mask for the spectra, shape (n, m) with n = timestamps, m = number of channels per timestamp
    smooth_type : str
        type of smoothing kernel. Options are 'hamming', 'wiener'. Default is 'wiener', which convolves the 1D data with Wiener filter of size `smooth_kernel`
    kernel_sizes : int
        number of kernel sizes to use for smoothing. Default is 2
    kernel_sequence_type : str
        type of kernel sequence. Options are 'middle_fast', 'fast', 'slow'. Default is 'middle_fast'
    usedbinning : int
        binning size to use for statistics calculation. Default is 1
    bound_sigma : int
        sigma value for boundary range. Default is 4
    stats_type : str
        type of statistics to use for boundary range. Options are 'mean', 'madstd', 'madmedian', 'median'. Default is 'median'
    smooth_bound_kernel : int
        size of the smoothing kernel for boundary range. Default is 31
    """
    import matplotlib.pyplot as plt
    from copy import copy

    from time import process_time

    grad_select     = copy(spec_mask)
    grad_select_org = copy(spec_mask)


    # set the kernel sequence
    #
    kernel = kernel_sequence(kernel_sizes,kernel_sequence_type)


    info_fg   = []
    info_fg_k = []

    print("In flag_smoothing: len(kernel) = ", len(kernel), type(kernel))
    print("In flag_smoothing: kernel = ", kernel)
    for k in kernel:
        
        grad_select_org = grad_select

        # smooth the original spectrum and subtract it from the
        # original dataset
        #
        #start = time.perf_counter()
        sm_data   = batch_convolve_1d_data(spec,smooth_type=smooth_type,smooth_kernel=k)
        #end = time.perf_counter()
        #print(f"In flag_smoothing: convolve_1d_data : {end - start:0.7f} seconds")
        resi_data = spec - sm_data

        # generate a flag based on the boundary region
        #
        #start = time.perf_counter()
        grad_select,boundary_up,boundary_low = boundary_range(freq,resi_data,usedbinning,np.invert(grad_select),\
                                                                  bound_sigma=bound_sigma,stats_type=stats_type,\
                                                                  smooth_kernel=smooth_bound_kernel)
        #end = time.perf_counter()
        #print(f"In flag_smoothing: boundary_range : {end - start:0.7f} seconds")
        # and combine with the previous                                                          
        grad_select = np.logical_or(grad_select,grad_select_org)

    return grad_select

def clean_up_1d_mask(mask,clean_bins=[[1,0,1]],setvalue=1):
    """
    just clean up single entries
    """
    from copy import deepcopy

    # convert the boolmask into integer
    inputmask  = deepcopy(mask).astype(int)


    for bins in clean_bins:
        # convolve teh mask with a kernel (clean_bins)
        #
        conv_mask  = convolve_1d_data(inputmask,'kernelinput',bins)

        # expected maximum of convolution
        expected_max_correlation = np.sum(bins)

        # find the maxima
        maxcor = conv_mask == expected_max_correlation
        maxcor_idx = np.where(maxcor == True)[0].tolist()

        # 
        for cidx in maxcor_idx:
            if len(bins)%2 > 0:
                inputmask[cidx-int((len(bins)-1)/2):cidx+int((len(bins)-1)/2) + 1] = setvalue
            else:
                inputmask[cidx-int(len(bins)/2):cidx+int(len(bins)/2)] = setvalue


    return inputmask.astype(bool)


def clean_up_1d_mask_ht(mask,bins=[[True,False,True]],setvalue=True):
    """
    vectorized sliding window mask matching

    Parameters
    ----------
    mask : ht.DNDarray
        2D array of the mask to be cleaned, shape (n, m) with n = timestamps, m = number of channels per timestamp
    bins : list
        list of masks to be used for cleaning
    setvalue : bool
        value to set the mask to. Default is True
    """
    from copy import deepcopy
    inputmask = deepcopy(mask)

    timestamps = inputmask.shape[0]
    channels = inputmask.shape[1]
    for k in range(len(bins)):
        # # original 
        # for i in range(len(inputmask)-len(bins[k])):
        #         isequal = inputmask[i:i+len(bins[k])] == bins[k]
                        
        #         if np.cumsum(isequal.astype(int))[-1] == len(bins[k]):
        #             inputmask[i:i+len(bins[k])-1] = setvalue
        

        # --> vectorized search and smaller loops (140x speedup):
        # create a 3D version of inputmask with `len(inputmask)-len(bins[k])` rows and `len(bins[k])` columns for each timestamp
        # Example: if inputmask[0] were [0, 1, 2, 3, 4, 5], and len(bins[k]) = 3, then
        # inputmask_2d = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
        shape_3d = (timestamps, channels-len(bins[k]), len(bins[k]))
        inputmask_3d = ht.zeros(shape_3d, dtype=ht.bool, split=0, device=inputmask.device)
        for j in range(len(bins[k])):
            # local torch operation to speed up the process
            # NB `larray` is the process-local array of the DNDarray (a torch tensor)
            # TODO: replace this loop with ht.unfold once merged
            inputmask_3d[:, :, j].larray = inputmask.larray[:, j:channels-len(bins[k])+j]
        # find the rows where all elements are equal to the elements in bins[k]            
        isequal_3d = ht.where((inputmask_3d == bins[k]).sum(axis=-1) == len(bins[k]))
        # set the values in original inputmask to setvalue
        for i in isequal_3d[0]:
            inputmask[:, i:i+len(bins[k])-1] = setvalue
        # free up memory            
        del inputmask_3d
        
    return inputmask


def checkerstats(data,split,stats_type):
    """
    Calculate the statistics of the data on `split` number of subarrays
    """
    sp_data = np.array_split(data,split)

    # # vectorized implementation before update 12.6.2024 # #
    # # pad data with zeros to make sure it is divisible by split
    # # store number of padded zeros to remove them at the end
    # padded_elements = split - len(data) % split
    # data = np.pad(data, (0, padded_elements), mode='constant', constant_values=(0, 0))
    # # reshape data to a 2D array, each row is a bin
    # data = data.reshape((split, -1))
    # # calculate statistics
    # statsmean, statsstd, _ = data_stats_vect(data, padded_elements, stats_type, accur=100)
    # # reshape data back to 1D array and remove padded zeros
    # data = data.flatten()[:-padded_elements]
    # return statsmean, statsstd

    if stats_type == 'madmadmedian':
        #
        from astropy.stats import mad_std, median_absolute_deviation
        #
        data_mean      = median_absolute_deviation(sp_data,axis=1)
        data_std       = mad_std(sp_data,axis=1)

    elif stats_type == 'madmean':
        #
        from astropy.stats import mad_std
        #
        data_mean      = np.mean(sp_data,axis=1) 
        data_std       = mad_std(sp_data,axis=1)

    elif stats_type == 'madmedian':
        #
        from astropy.stats import mad_std
        #
        data_mean      = np.median(sp_data,axis=1) 
        data_std       = mad_std(sp_data,axis=1)

    elif stats_type == 'median':
        data_mean      = np.median(sp_data,axis=1)
        data_std       = np.std(sp_data,axis=1)

    else:
        print('bugger not know')
        sys.exit(-1)

    return data_mean,data_std


def checkerstats_sel(data,split,select,stats_type):
    """
    note that the mask (1/True) is indicating bad data 
    """

    sp_data        = np.array_split(data,split)
    sp_data_select = np.invert(np.array_split(select,split)) # this is from old times

    data_masked    = ma.masked_array(sp_data,sp_data_select)


    if stats_type == 'madmadmedian':
        #
        from astropy.stats import mad_std, median_absolute_deviation
        #
        data_mean      = median_absolute_deviation(data_masked,axis=1)
        data_std       = mad_std(data_masked,axis=1)

    elif stats_type == 'madmean':
        #
        from astropy.stats import mad_std
        #
        data_mean      = np.mean(data_masked,axis=1) 
        data_std       = mad_std(data_masked,axis=1)

    elif stats_type == 'madmedian':
        #
        from astropy.stats import mad_std
        #
        data_mean      = np.median(data_masked,axis=1) 
        data_std       = mad_std(data_masked,axis=1)

    elif stats_type == 'median':
        data_mean      = np.median(data_masked,axis=1)
        data_std       = np.std(data_masked,axis=1)

    else:
        print('bugger not know')
        sys.exit(-1)

    
    return data_mean,data_std,data_mean.mask.astype(int)


def flag_impact(mask,orgmask):
    """
    provide info about the flagging impact
    """

    # in case to check the impact of the flagging process
    #
    f_mask                = difference_mask(mask.astype(bool),orgmask.astype(bool))
    f_mask_info           = check_mask(f_mask.astype(int))

    return f_mask_info

def difference_mask(mask,orgmask):
    """
    difference mask between mask and orgmask
    """
    
    new_mask             = deepcopy(mask).astype(bool)

    equal_mask           = np.logical_and(mask.astype(bool),orgmask.astype(bool))

    new_mask[equal_mask] = False

    return new_mask
    

def str_in_strlist(string,strlist):
    """
    """
    isthere = False
    for k in strlist:
        if string.count(k) > 0:
            isthere = True 
            break
    return isthere
    
