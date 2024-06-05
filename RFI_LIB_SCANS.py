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
from scipy.signal import convolve2d
from scipy import stats

def data_stats(data,stats_type='mean',accur=100):
    """
    return mean and derivation of the input data
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
        data_mean,data_std = kdemean(data,accucary=1000)

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



def kdemean(x,accucary=1000):
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

    vra    = linspace(-1*max_range,max_range,accucary)
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

    if idx_half_power >= accucary -1:
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

    # To get the upper and lower boundaries we used binned data and its statistics
    # 
    stats_x_data,stats_x_datastd  = checkerstats(xdata,usedbinning,stats_type)
    stats_y_data,stats_y_datastd,stats_mask  = checkerstats_sel(ydata,usedbinning,ydata_mask,stats_type)


    # determine the boundary region 
    #
    # x range
    #
    x_boundary_values = []
    x_boundary_values.append(xdata[0])
    for d in range(len(stats_x_data)):
        x_boundary_values.append(stats_x_data[d])
    x_boundary_values.append(xdata[-1])


    # y range 
    #
    #
    y_boundary_values_up  = []
    y_boundary_values_low = []
    #
    y_boundary_values_up.append(stats_y_data[0] + bound_sigma * stats_y_datastd[0])
    y_boundary_values_low.append(stats_y_data[0] - bound_sigma * stats_y_datastd[0])
    for d in range(len(stats_y_data)):
        y_boundary_values_up.append(stats_y_data[d] + bound_sigma * stats_y_datastd[d])
        y_boundary_values_low.append(stats_y_data[d] - bound_sigma * stats_y_datastd[d])
    y_boundary_values_up.append(stats_y_data[-1] + bound_sigma * stats_y_datastd[-1])
    y_boundary_values_low.append(stats_y_data[-1] - bound_sigma * stats_y_datastd[-1])


    # Here check on the boundary 
    # 
    boundary_stats_up   = data_stats(y_boundary_values_up,stats_type='median',accur=100)
    boundary_stats_low  = data_stats(y_boundary_values_low,stats_type='median',accur=100)

    #
    if usedbinning > 10:

        if boundary_stats_up[0] != boundary_stats_up[1]:

            bound_select         = np.logical_or(y_boundary_values_up >= boundary_stats_up[0] + bound_sigma * boundary_stats_up[1],\
                                                     y_boundary_values_low <= boundary_stats_low[0] - bound_sigma * boundary_stats_low[1])  # True for bad data
        else:
            bound_select         = np.zeros(len(y_boundary_values_low)).astype(bool)
        
    else:
            bound_select         = np.zeros(len(y_boundary_values_low)).astype(bool)


    # in case there are faulty values in
    # checkerstats_selcheckerstats_sel function
    # these has been marked
    #
    if max(stats_mask) > 0:
        stats_mask    = stats_mask.astype(bool)
        stats_mask    = np.insert(stats_mask,0,bound_select[0])
        stats_mask    = np.insert(stats_mask,len(stats_mask),bound_select[-1])
        bound_select  = np.logical_or(bound_select,stats_mask.astype(bool))

    
    # Interpolate and convolve the boundary region 
    # of the data and use this to mark bad data
    #
    interp_boundary_up         = np.interp(xdata,np.array(x_boundary_values)[np.invert(bound_select)],np.array(y_boundary_values_up)[np.invert(bound_select)])
    interp_boundary_low        = np.interp(xdata,np.array(x_boundary_values)[np.invert(bound_select)],np.array(y_boundary_values_low)[np.invert(bound_select)])
    #
    if smooth_kernel > 3:
        boundary_up                = convolve_1d_data(interp_boundary_up,smooth_type='wiener',smooth_kernel=smooth_kernel)
        boundary_low               = convolve_1d_data(interp_boundary_low,smooth_type='wiener',smooth_kernel=smooth_kernel)
    else:
        boundary_up  = interp_boundary_up
        boundary_low = interp_boundary_low

    # select on the boundary data
    #
    grad_select         = np.logical_or(ydata >= boundary_up,ydata <= boundary_low)  # True for bad data


    return grad_select,boundary_up,boundary_low



def flag_spec_by_smoothing(fg_spec,freq,cleanup_spec_mask,splitting,kernel_sizes,smooth_type,usedbinning,bound_sigma,stats_type,smooth_bound_kernel,idx=0,mtque=None,njobs=1):
    """
    specially for doing single azimuthe scans
    """

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


        grad_selectsp = flag_smoothing(sp_freq,sp_data,sp_data_mask,smooth_type=smooth_type[sp],kernel_sizes=kernel_sizes[sp],\
                                           usedbinning=usedbinning[sp],bound_sigma=bound_sigma[sp],stats_type=stats_type[sp],\
                                           smooth_bound_kernel=smooth_bound_kernel[sp])

        grad_select = np.append(grad_select,grad_selectsp)


    # clean up based on some pattern 
    #
    clean_bins = [\
                      [True,False,True],\
                      [True,False,False,True],\
                      [True,False,False,False,True],\
                      [True,False,False,False,False,True],\
                      [True,False,False,False,False,False,True],
                      [False,False,False,True,False,False,False]\
                      ]

    final_sp_mask = clean_up_1d_mask(grad_select,clean_bins,setvalue=True)

    if mtque != None:
        resultdic =  {}
        resultdic[njobs] = [idx,final_sp_mask]
        mtque.put(resultdic)

    return final_sp_mask


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
         sm_data = wiener(data,smooth_kernel)

    else:
        sm_data = deepcopy(data)

    return sm_data


def flag_smoothing(freq,spec,spec_mask,smooth_type='wiener',kernel_sizes=2,usedbinning=1,bound_sigma=4,stats_type='median',smooth_bound_kernel=31):
    """
    """
    import matplotlib.pyplot as plt
    from copy import copy

    grad_select     = copy(spec_mask)
    grad_select_org = copy(spec_mask)

    # a step by step kernel generates a
    # cleaner spectrum, but takes 10 sec 
    # per time step
    #
    # kernel = 2 * np.arange(kernel_sizes) + 1

    kernel = []
    for i in range(1,kernel_sizes):
        if (3*i**2)%2 == 0:
            kernel.append(3*i**2 + 1)
        else:
            kernel.append(3*i**2)

    info_fg   = []
    info_fg_k = []

    for k in kernel:
        
        grad_select_org = grad_select

        # smooth the original spectrum and subtract it from the
        # original dataset
        #
        sm_data   = convolve_1d_data(spec,smooth_type=smooth_type,smooth_kernel=k)
        resi_data = spec - sm_data

        # generate a flag based on the boundary region
        #
        grad_select,boundary_up,boundary_low = boundary_range(freq,resi_data,usedbinning,np.invert(grad_select),\
                                                                  bound_sigma=bound_sigma,stats_type=stats_type,\
                                                                  smooth_kernel=smooth_bound_kernel)

        # and combine with the previous                                                          
        grad_select = np.logical_or(grad_select,grad_select_org)


    return grad_select


def clean_up_1d_mask(mask,bins=[[True,False,True]],setvalue=True):
    """
    just clean up single entries
    """
    from copy import deepcopy
    inputmask = deepcopy(mask)

    for k in range(len(bins)):
        for i in range(len(inputmask)-len(bins[k])):
                isequal = inputmask[i:i+len(bins[k])] == bins[k]
                        
                if np.cumsum(isequal.astype(int))[-1] == len(bins[k]):
                    inputmask[i:i+len(bins[k])-1] = setvalue

    return inputmask



def checkerstats(data,split,stats_type):

    sp_data = np.array_split(data,split)

    statsmean = []
    statsstd  = []

    for sp in sp_data:
        stdata = data_stats(sp,stats_type,accur=100)
        statsmean.append(stdata[0])
        statsstd.append(stdata[1])

    return statsmean,statsstd

def checkerstats_sel(data,split,select,stats_type):
    """
    note that the mask (1/True) is indicating bad data 
    """
    sp_data        = np.array_split(data,split)
    sp_data_select = np.array_split(select,split)

    sel_mask       = np.zeros(len(sp_data))
    
    statsmean = []
    statsstd  = []

    for sp in range(len(sp_data)):
        spdata = sp_data[sp][sp_data_select[sp]]

        if len(spdata) > 1:
            stdata = data_stats(spdata,stats_type,accur=100)
        else:
            if len(statsmean) != 0:
                stdata = [statsmean[-1],statsstd[-1]]
                sel_mask[sp] = 1
            else:
                stdata = data_stats(data,stats_type,accur=100)
                sel_mask[sp] = 1

        statsmean.append(stdata[0])
        statsstd.append(stdata[1])

    return statsmean,statsstd,sel_mask


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
    
