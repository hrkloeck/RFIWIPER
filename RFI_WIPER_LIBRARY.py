# HRK 2025
#
# Hans-Rainer Kloeckner
# hrk@mpifr-bonn.mpg.de 
#
#
# RFI Mitigation Lib for SKAMPI WIPER
# 
# To be consistent with numpy masked arrays 
# data that is bad will be masked asd True or 1
# 
# --------------------------------------------------------------------


import numpy as np
import numpy.ma as ma
from scipy.signal import convolve2d,correlate
from scipy.interpolate import interp1d
from scipy import stats
import json
from time import process_time
from copy import copy
#
import RFI_WIPER_TOOLS_STATISTIC as STS


# ===========================================================================
#
# RFI mitigation techniques and functions working on single spectrum
# 
# the input data:
#                 - one dimensional array intensity per channels
#                 - one dimensional boolean array as mask excluding channels
#
# the output data:
#                 - one dimensional boolean array as mask excluding channels
#
# ===========================================================================


def flag_spectrum_by_growthratechanges(channel,intensity,mask,stats_type_threshold,sigma_threshold,envelop_bins,sigma_envelop,smooth_type_envelop,smooth_kernel_size_envelop,showenvelopplot=False):
                                                                                
    """
    determine the functional growthrate f(x) * f'(x) 
    
    use sigma threshold for flagging and envelop masking
    on the changes of the growth rate

    output is an updated mask
    """

    mask_select     = copy(mask)
    mask_select_org = copy(mask)


    interpol_channel, interpol_intensity = interpolate_spectrum(channel,intensity,mask_select,new_channels=None,interpol_type='extrapolate',mask_true_flag=True)

    growthrate = interpol_intensity * np.gradient(interpol_intensity,interpol_channel)

    growthrate = np.diff(growthrate,append=0)
    
    # determine new mask by thresholding 
    mask_select = threshold_data(data=growthrate,reference_data=growthrate,data_mask=mask_select,\
                                             sigma=sigma_threshold,stats_type=stats_type_threshold,mask_true_flag=True)
                                             
    # determine the number of channels per spwd
    #
    splitting_in_bins_base_2 = (np.log(len(channel)/envelop_bins)/np.log(2))

    # determine new mask by envelope thresholding        
    mask_select                          = error_envelope_xy(interpol_channel,growthrate,mask_select,splitting_in_bins_base_2,sigma_envelop,\
                                                                                smooth_type_envelop,smooth_kernel_size_envelop,interpol_type='extrapolate',\
                                                                 mask_true_flag=True,showenvelopplot=showenvelopplot)

    # update the mask 
    mask_select = np.logical_or(mask_select,mask_select_org)

    return mask_select


def flag_spectrum_by_thresholding_with_smoothing(channel,intensity,mask,smooth_type_intensity,smooth_kernel_sizes_intensity,envelop_bins,sigma_envelop,smooth_type_envelop,smooth_kernel_size_envelop,showenvelopplot=False):
                                                                                
    """
    iterrative smoothing the spectrum with increasing kernel sizes 
    and threshold the difference spectrum (original - smoothed)  

    output is an updated mask
    """

    mask_select     = copy(mask)
    mask_select_org = copy(mask)

    # determine the number of channels per spwd
    #
    splitting_in_bins_base_2 = (np.log(len(channel)/envelop_bins)/np.log(2))
    
    for k in smooth_kernel_sizes_intensity:
        
        mask_select_org = mask_select

        # 1. interpolate between flagged channels 
        # 2. subtranced smoothed spectrum from original 
        # 3. exclude outlyers
        # 4. update mask
        # 5. use updated mask and go to 1. repeat with next kernel size
        #
        interpol_channel, interpol_intensity = interpolate_spectrum(channel,intensity,mask_select,new_channels=None,interpol_type='extrapolate',mask_true_flag=True)
        
        smoo_amp_interpol_bslcorr            = interpol_intensity - convolve_1d_data(interpol_intensity,smooth_type=smooth_type_intensity,smooth_kernel_size=k)

        mask_select                          = error_envelope_xy(interpol_channel,smoo_amp_interpol_bslcorr,mask_select,splitting_in_bins_base_2,sigma_envelop,\
                                                                                smooth_type_envelop,smooth_kernel_size_envelop,interpol_type='extrapolate',\
                                                                     mask_true_flag=True,showenvelopplot=showenvelopplot)

        # update the mask 
        mask_select = np.logical_or(mask_select,mask_select_org)


    return mask_select


def flag_spectrum_by_basline_fitting(channel,intensity,mask,stats_type,envelop_bins,sigma_envelop,smooth_type_envelop,smooth_kernel_size_envelop):
    """
    Use scipy 
    """
    import pybaselines as pbsl

    # interpolate masked spectrum
    #
    interpol_channel, interpol_intensity = interpolate_spectrum(channel,intensity,mask,new_channels=None,interpol_type='extrapolate',mask_true_flag=True)

    
    # use pybasline and different methods to do a baseline fit
    #
    # https://pybaselines.readthedocs.io/en/v0.8.0/api/pybaselines/whittaker/index.html
    #
    regular_modpoly_0 = pbsl.whittaker.airpls(interpol_intensity)[0]
    regular_modpoly_1 = pbsl.whittaker.arpls(interpol_intensity)[0]
    regular_modpoly_2 = pbsl.whittaker.aspls(interpol_intensity)[0]
    regular_modpoly_3 = pbsl.whittaker.derpsalsa(interpol_intensity)[0]   # this is really good
    regular_modpoly_4 = pbsl.whittaker.drpls(interpol_intensity)[0]
    regular_modpoly_5 = pbsl.whittaker.iarpls(interpol_intensity)[0]
    regular_modpoly_6 = pbsl.whittaker.iasls(interpol_intensity)[0]
    regular_modpoly_7 = pbsl.whittaker.psalsa(interpol_intensity)[0]
    #
    baseline_fit_info = ['whittaker.airpls','whittaker.arpls','whittaker.aspls','whittaker.derpsalsa','whittaker.drpls',
                              'whittaker.iarpls','whittaker.iasls','whittaker.psalsa']

    baseline_fit_intensity  = np.array([regular_modpoly_0,regular_modpoly_1,regular_modpoly_2,regular_modpoly_3,regular_modpoly_4,\
                                       regular_modpoly_5,regular_modpoly_6,regular_modpoly_7])


    bsl_stats = []
    for bsli in range(baseline_fit_intensity.shape[0]):
        bsl_stats_data_mean, bsl_stats_data_std, bsl_stats_stats_type = STS.data_stats(interpol_intensity-baseline_fit_intensity[bsli],stats_type=stats_type)
        bsl_stats.append(bsl_stats_data_std/bsl_stats_data_mean)


    std_mean_ratio_min = np.argmin(bsl_stats)
    std_mean_ratio_max = np.argmax(bsl_stats)

    if std_mean_ratio_min == std_mean_ratio_min:
        std_mean_ratio_min = 3
        std_mean_ratio_max = 1
        baseline_fit_info[std_mean_ratio_min] = 'Caution has been set to: '+baseline_fit_info[std_mean_ratio_min]
    
    # use the baseline with the minium std_mean ratio
    #
    bslf_corr_intensity         = interpol_intensity - baseline_fit_intensity[std_mean_ratio_min]
    #
    mask_select                 = error_envelope_xy(interpol_channel,bslf_corr_intensity,mask,envelop_bins,sigma_envelop,\
                                                                                smooth_type_envelop,smooth_kernel_size_envelop,interpol_type='extrapolate',mask_true_flag=True)

    return mask_select, baseline_fit_info[std_mean_ratio_min]
                                                                                
def detect_peaks(row, window_size, n_sigma, replace='minimum'):
    '''
    Detects spikes in a spectrum using sigma clipping.
    Mask notation: True=masked (bad data), False=not masked (good data)

    From Tobias Winchen

    Parameters:
            row (masked array): 1D spectrum (one time slice over all channels) with mask
            window_size (int): Window size for rolling mean
            n_sigma (int): Factor for clipping threshold (n_sigma * std(original - smoothed spectrum))
            replace (string): function to replace flagged values for rolling mean in the next iteration

    Returns:
            spectrum (masked array): Spectrum with masked values replaced by the rolling mean
    '''

    from scipy.ndimage import uniform_filter1d

    # smooth spectrum with a rolling mean
    smooth = uniform_filter1d(row.data, window_size)
    
    if replace == 'mean':
        replace_val = uniform_filter1d(row.data, window_size)
    elif replace =='minimum':
        from scipy.ndimage import minimum_filter1d
        replace_val = minimum_filter1d(row.data, window_size)
    elif replace =='median':
        from scipy.ndimage import median_filter
        replace_val = median_filter(row.data, window_size)
    else:
        replace_val = uniform_filter1d(row.data, window_size)

    # calculate the diff
    diff = row.data - smooth

    # flag values with an absolute diffs greater than n_sigma times standard deviation
    clip = n_sigma * np.std(diff)
    mask = row.mask
    masked_idx_1 = np.where(diff >= clip)
    masked_idx_2 = np.where(diff < -clip)
    mask[masked_idx_1] = True
    mask[masked_idx_2] = True

    # replace the flagged values by smoothed values
    filled_spectrum = row.filled(replace_val)
    spectrum = ma.masked_array(filled_spectrum, mask=mask)

    return spectrum


def flag_spectrum_thresholding(intensity, mask, window_size_start, n_sigma, replace='minimum'):
    '''
    Run sigma clipping iterations until no further channels are flagged.
    Mask notation for mask_row: True=masked (bad data), False=not masked (good data)

    From Tobias Winchen
  
    Parameters:
            row (1D array): 1D spectrum (one time slice over all channels)
            mask_row (1D array): 1D mask of the same shape as row
            window_size_start (int): Window size for rolling mean for the first iteration.
                The window is increased by this size in every iteration
            n_sigma (int): Factor for clipping threshold (n_sigma * std(original - smoothed spectrum))
            replace (string): function to replace flagged values for rolling mean in the next iteration

    Returns:
            mask (1D boolean array): 1D mask with size of row
    '''
    from scipy.ndimage import uniform_filter1d
    
    mask_row     = copy(mask)
    row          = copy(intensity)
    
    smooth          = uniform_filter1d(row, 1000)
    filled_spectrum = ma.masked_array(row, mask=mask_row.copy()).filled(smooth)
    masked_spectrum = ma.masked_array(filled_spectrum, mask_row.copy())
    iteration = 1
    while True:
        flagged_channels_before = masked_spectrum.mask.sum()
        masked_spectrum = detect_peaks(
            masked_spectrum,
            window_size=window_size_start*(iteration),
            n_sigma=n_sigma,
            replace=replace,
        )
        flagged_channels_after = masked_spectrum.mask.sum()
        if (flagged_channels_after - flagged_channels_before == 0) or window_size_start*(iteration) > 0.1*len(row):
            break
        iteration+=1
    # return masked_spectrum.mask
    #return ma.masked_array(row, mask=masked_spectrum.mask)
    return masked_spectrum.mask


# ===========================================================================
#
# RFI mitigation techniques and functions working on waterfall spectrum
# 
# the input data:
#                 - two dimensional array intensity per time (axis=0) channels (axis=1)
#                 - two dimensional boolean array as mask excluding channels per time
#
# the output data:
#                 - two dimensional boolean array as mask excluding channels
#
# ===========================================================================


def flag_waterfall_by_noise_variation_in_channels(wtspectrum,mask,sigma_threshold,sigma_noise,stats_type_threshold,stats_type_noise,mask_true_flag=True):
    """
    Here each channel is treated as a single receiver 
    and the system noise of such a receiver is changing 
    linear with repect to the total power

    Note that might only be useful for single dish data
    """

    spectrum_masked             = ma.masked_array(wtspectrum,mask=mask,fill_value=np.nan)

    # flag channels on std outlieres axis=0 is time, axis=1 is channel
    #
    std_over_time_per_freq     = np.std(spectrum_masked,axis=0)


    # determine new mask by thresholding saturation
    #
    chan_mask = threshold_data(data=std_over_time_per_freq.data,reference_data=std_over_time_per_freq.compressed(),data_mask=std_over_time_per_freq.mask,\
                                             sigma=sigma_threshold,stats_type=stats_type_threshold,mask_true_flag=True)

    # update the original mask
    #
    for i in range(len(chan_mask)):
        if chan_mask[i] == 1:
            mask[:,i] = True

    # flag on a linear relation of std and mean
    #
    RFI_std_mean_mask_data_2  = ma.masked_array(wtspectrum,mask=mask,fill_value=np.nan)

    # does a linear fit to the std versus mean 
    #
    RFI_mean    = RFI_std_mean_mask_data_2.mean(axis=0)
    RFI_std     = RFI_std_mean_mask_data_2.std(axis=0)

    #
    linear_fit  = np.polyfit(RFI_mean,RFI_std, 2)
    lf_function = np.poly1d(linear_fit)
    #
    RFI_std_cor = RFI_std - lf_function(RFI_mean)
    #
    # -----

    # flag channels on std outlieres after subtracting the linear fit 
    #
    lf_mask             = threshold_data(data=RFI_std_cor.data,reference_data=RFI_std_cor.compressed(),data_mask=RFI_std_cor.mask,sigma=sigma_noise,\
                                                      stats_type=stats_type_noise,mask_true_flag=True)

                                                      
    # the last function cpuld also determined by error_envelope_xy
                                                      
    return lf_mask
    


def flag_waterfall_by_thresholding_with_smoothing_for_each_timestep(wtspectrum,mask,flagprocessing,smooth_type_intensity,\
                                                                        smooth_kernel_sizes_intensity,envelop_bins,sigma_envelop,smooth_type_envelop,smooth_kernel_size_envelop,toutput=True):
    """
    use the function flag_spectrum_by_thresholding_with_smoothing
    on each time step through the waterfall spectrum
    """
    from time import process_time

    masking_time    = process_time()
    org_mask        = copy(mask)

    for time in range(np.shape(wtspectrum)[0]):
        #
        # get the single averaged spectrum
        #
        #
        intensity                = wtspectrum[time,1:] # note exclude first channel to get base 2 number of channels
        channels                 = np.arange(len(intensity))
        intensity_mask           = org_mask[time,1:] # note exclude first channel to get base 2 number of channels

        # check if time has been flagged
        #
        if np.sum(intensity_mask) != len(intensity_mask):

            # note that flag_spectrum_by_thresholding_with_smoothing
            # need base 2 number of channels
            #
            smoo_select = flag_spectrum_by_thresholding_with_smoothing(channel=channels,intensity=intensity,mask=intensity_mask,\
                                                                           smooth_type_intensity=smooth_type_intensity,smooth_kernel_sizes_intensity=smooth_kernel_sizes_intensity,\
                                                                           envelop_bins=envelop_bins,sigma_envelop=sigma_envelop,smooth_type_envelop=smooth_type_envelop,\
                                                                           smooth_kernel_size_envelop=smooth_kernel_size_envelop)
                    
            for i in range(len(smoo_select)):
                if smoo_select[i] == True:
                    org_mask[time,i+1] = True
                    
    procces_time = process_time() - masking_time
    #
    if toutput:
        print(' WT each timestep FG time needed ',procces_time,' ')          

    return org_mask


def flag_waterfall_by_filtering_with_convolutional_smoothing(wtspectrum,mask,flagprocessing,kernels,kernel_types,sigma_threshold,stats_type='madmean',interpol_type='CloughTocher',gauss_sigma=0,gauss_size=0):
    """
    Waterfall spectrum convolution with specific filter 
    """
    
    from scipy.signal import convolve2d,oaconvolve

    #
    wtdata          = copy(wtspectrum[:])
    #
    mask_select     = copy(mask)
    mask_select_org = copy(mask)

    for k,kt in zip(kernels,kernel_types):
        
        mask_select_org = copy(mask_select)

        # 1. interpolate between flagged channels 
        # 2. convolve with specific filter 
        # 3. exclude outlyers
        # 4. update mask
        # 5. use updated mask and go to 1. repeat with next kernel size
        #
        interpolated_wtspectrum              = fill_masked_data(wtdata,mask_select_org,mask_true_flag=True,interpol_type=interpol_type,gauss_sigma=gauss_sigma,gauss_size=gauss_size)

        #interpolated_wtspectrum = interpolated_wtspectrum1 - np.median(interpolated_wtspectrum1,axis=0)[np.newaxis,:] 
        
        #convolved_interpolated_wtspectrum    = convolve2d(interpolated_wtspectrum,k,mode='same')  # 62.60706 sec
        #convolved_interpolated_wtspectrum    = oaconvolve(interpolated_wtspectrum,k,mode='same',axis=1) # 61.69937 sec
        convolved_interpolated_wtspectrum    = oaconvolve(interpolated_wtspectrum,k,mode='same') # 61.813621 sec

        # depending on the filter type either threshod is used on the filter result or
        # subtracting the original with the smoothed data
        #
        if kt == 'smooth':
            residual_data                        = interpolated_wtspectrum - convolved_interpolated_wtspectrum
        else:
            residual_data                        = convolved_interpolated_wtspectrum



        mask_select                          = threshold_data(data=residual_data,reference_data=residual_data.flatten(),\
                                                                  data_mask=mask_select_org,sigma=sigma_threshold,stats_type=stats_type,mask_true_flag=True)

        
        mask_select = np.logical_or(mask_select_org,mask_select)
        
    return mask_select



# ===========================================================================
#
# Here we are cleaning up the individual 1d or 2 d mask
# 
# ===========================================================================


def clean_up_1d_mask(mask,clean_bins=[[1,0,1]],setvalue=1):
    """
    just clean up single entries
    """
    from copy import deepcopy

    # convert the boolmask into integer
    inputmask  = deepcopy(mask).astype(int)


    for bins in clean_bins:
        # convolve the mask with a kernel (clean_bins)
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

def clean_up_2d_mask(mask,fraction_time=0.7,fraction_channel=0.7,mask_true_flag=True):
    """
    clean the mask in time and frequency if most of the data is masked out
    """

    # Clean up in time
    mask_sum_in_time                        = mask.sum(axis=0)
    sel_channel_integrated_over_time_fullfg = mask_sum_in_time != np.shape(mask)[0]
    sel_channel_integrated_over_time        = mask_sum_in_time > fraction_time*np.shape(mask)[0]
    # update the mask 
    sel_select_ch = np.logical_and(sel_channel_integrated_over_time,sel_channel_integrated_over_time_fullfg)

    for ch in range(len(sel_select_ch)):
        if sel_select_ch[ch] == mask_true_flag:
            mask[:,ch] = mask_true_flag
    # ==================
    
    # Clean up in channel
    mask_sum_in_channel                        = mask.sum(axis=1)
    sel_channel_integrated_over_channel_fullfg = mask_sum_in_channel != np.shape(mask)[1]
    sel_channel_integrated_over_channel        = mask_sum_in_channel > fraction_channel*np.shape(mask)[1]
    # update the mask 
    sel_select_t = np.logical_and(sel_channel_integrated_over_channel,sel_channel_integrated_over_channel_fullfg)

    for t in range(len(sel_select_t)):
        if sel_select_t[t] == mask_true_flag:
            mask[t,:] = mask_true_flag
    # ==================
    
    return mask



# ===========================================================================
#
# RFI mitigation techniques and functions working on meta data of the observation
# 
#                 - one dimensional array information per time
#                 - one dimensional boolean array as mask excluding time
#
# the output data:
#                 - one dimensional boolean array as mask excluding time
#
# ===========================================================================





# ############################################################################
#
# Functions that are useful
#
# Note: numpy masked arrays considers a bolean value of True (1) is to be masked out
#
#
# ############################################################################


def interpolate_spectrum(channel,intensity,mask,new_channels=None,interpol_type='extrapolate',mask_true_flag=True):
    """
    fill masked data with smooth or interpolated values 
    """

    from scipy.interpolate import interp1d
    
    if mask_true_flag:
        clean_channel    = channel[np.invert(mask)]
        clean_intensity  = intensity[np.invert(mask)]
    else:
        clean_channel    = channel[mask]
        clean_intensity  = intensity[mask]
        
    #
    if interpol_type == 'extrapolate':
        interpol_function   = interp1d(clean_channel,clean_intensity,fill_value='extrapolate')
        if new_channels == None:
            interpol_intensity  = interpol_function(channel)
            interpol_channel    = channel
        else:
            interpol_intensity  = interpol_function(new_channels)
            interpol_channel    = new_channels

    return interpol_channel, interpol_intensity
    


def fill_masked_data(data,mask,mask_true_flag=True,interpol_type='CloughTocher',gauss_sigma=0,gauss_size=0):
    """
    interpolates a 2 dimensional array 
    """

    from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
    from scipy.signal import convolve2d,oaconvolve



    if interpol_type == 'Gaussian':

        smooth_kernel   = gaussian_kernel(gauss_size,gauss_sigma,scale=1)
        
        #smooth_data     = convolve2d(data,smooth_kernel,mode='same')
        smooth_data     = oaconvolve(data,smooth_kernel,mode='same') # this is a super heavy improvement in time
        
        sel_data = mask == mask_true_flag

        data[sel_data]  = smooth_data[sel_data]

        return data

    else:
        time                       = np.arange(0,np.shape(data)[0])
        channel                    = np.arange(0,np.shape(data)[1])

        coords_time,coords_channel = np.meshgrid(channel,time)

        sel_data = mask == np.invert(mask_true_flag)

        points = np.array([list(coords_channel[sel_data]),list(coords_time[sel_data])]).T
        values = data[coords_channel[sel_data],coords_time[sel_data]]
    
        if interpol_type == 'CloughTocher':
            interpolation = CloughTocher2DInterpolator(points,values)
        else:
            interpolation = LinearNDInterpolator(points,values)
    
        return interpolation(coords_channel,coords_time)


# https://sustainabilitymethods.org/index.php/Outlier_Detection_in_Python

def error_envelope_xy(data_x,data_y,mask,bins,sigma,smooth_type,smooth_kernel,interpol_type='extrapolate',mask_true_flag=True,showenvelopplot=False):
    """
    determine the error envelope of a distribution of point along an axis 
    with varying errors

    input: two one dimensional arrays (use interpoled data) 
           define number of bins to splitt the data
           define sigma of the envelope
           define smooth type for smoothing the envelope
           define smooth kernel for smoothing the envelope

    output: updates mask
    """
    new_data_x              = None
    idata_x, idata_y        = interpolate_spectrum(data_x,data_y,mask,new_data_x,interpol_type,mask_true_flag)
    
    # split data in 
    bin_data_x              = np.array_split(idata_x,2**bins)
    bin_data_y              = np.array_split(idata_y,2**bins)
    #
    up_envelop              = sigma * np.std(bin_data_y,axis=1)
    smooth_envelop          = convolve_1d_data(up_envelop,smooth_type,smooth_kernel)
    #
    intepol_envelop_func    = interp1d(np.mean(bin_data_x,axis=1),smooth_envelop,fill_value=interpol_type)
    bound_up_interpoled     = intepol_envelop_func(data_x)
    bound_low_interpoled    = -1 * bound_up_interpoled
    #
    select_up               = data_y > bound_up_interpoled
    select_low              = data_y < bound_low_interpoled
    #
    mask[select_up]         = mask_true_flag
    mask[select_low]        = mask_true_flag

    if showenvelopplot:
        # This is for debugging purpose
        from matplotlib import pylab as plt
        plt.scatter(data_x,data_y)
        plt.plot(data_x,bound_up_interpoled)
        plt.plot(data_x,bound_low_interpoled)
        plt.show()
        print('\n\nOk you want to play around with the input parameter,\n envelop_xy_bins,envelop_xy_smooth_kernel, envelop_xy_smooth_kernel_size\n\n')
        sys.exit(-1)
    
    return mask
    

def threshold_data(data,reference_data,data_mask,sigma,stats_type='mean',mask_true_flag=True):
    """
    use upper and lower thresholds to mask out data
    data is a unflagged (e.g. compressed dataset)
    reference_data original data

    """

    # determine the lower and upper thresholds
    #
    ref_data_low,ref_data_up,stats_type = STS.data_stats(reference_data,stats_type,sigma,spwd=1)
    #
    select_up               = data > ref_data_up
    select_low              = data < ref_data_low
    #
    data_mask[select_up]    = mask_true_flag
    data_mask[select_low]   = mask_true_flag
    #
    return data_mask


def convolve_1d_data(data,smooth_type='hanning',smooth_kernel_size=3):
    """
    smooth a 1-dimension array with various smoothing kernels

    smooth_type
    smooth_kernel_size is the window size

    if smooth_type unkown the identical data will be returned
    """
    from scipy.signal import wiener,gaussian,medfilt,convolve,oaconvolve
    from scipy.signal.windows import hamming #, hanninng #hanning,convolve,hamming,gaussian,medfilt
    #
    from scipy.ndimage import uniform_filter1d
    from scipy.ndimage import minimum_filter1d
    from scipy.ndimage import median_filter
    
    if smooth_type == 'hamming':
        sm_kernel = hamming(smooth_kernel_size)
        sm_data   = convolve(data,sm_kernel,mode='same') / sum(sm_kernel)

    elif smooth_type == 'gaussian':
        sm_kernel = gaussian(smooth_kernel_size,smooth_kernel_size)
        sm_data   = convolve(data,sm_kernel,mode='same') / sum(sm_kernel)

    elif smooth_type == 'median':
         #sm_data = medfilt(data,smooth_kernel_size)
         sm_data = median_filter(data, smooth_kernel_size)

    elif smooth_type == 'wiener':
         sm_data = wiener(data,smooth_kernel_size)

    elif smooth_type == 'kernelinput':
        sm_data   = convolve(data,smooth_kernel_size,mode='same')
        #sm_data   = oaconvolve(data,smooth_kernel_size,mode='same')

    elif smooth_type == 'mean':
         sm_data = uniform_filter1d(data,smooth_kernel_size)

    elif smooth_type == 'minimum':
         sm_data = minimum_filter1d(data,smooth_kernel_size)
        
    else:
        sm_data = deepcopy(data)
        
    return sm_data


def smooth_kernels_2d(smk_type):
    """
    provide a smooth kerenel to be used by 2d convolution
    """
    # ----------------------------------------------
    #
    # here are some examples of kernels for edge detection to find strong RFI
    #

    # default is identity
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
        kernel     = [[3,10,3],[0,0,0],[-3,-10,-3]]    # Scharr operator di/dy
    if smk_type == 'sobelx':
        kernel      = [[-1,0,1],[-2,0,2],[-1,0,1]]     # Sobel operator di/dx
    if smk_type == 'sobely':
        kernel      = [[1,2,1],[0,0,0],[-1,-2,-1]]     # Sobel operator di/dy
    if smk_type == 'canny':
        kernel       = [[2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]] 
    if smk_type == 'prewittx':        
        kernel   = [[-1,0,1],[-1,0,1],[-1,0,1]]        # Prewitt operator di/dx
    if smk_type == 'prewitty':
        kernel   = [[1,1,1],[0,0,0],[-1,-1,-1]]        # Prewitt operator di/dy
    if smk_type == 'laplace':
        kernel   = [[0,1,0],[1,-4,1],[0,1,0]]          # Laplace filter

    return kernel

def gaussian_kernel(k,sigma=1.3,scale=15):
    """
    provides a gaussian kernel to smooth 
    images

    Note: Default setting resembles the canny filter with k=2
    """

    g_kernel = np.ones((2*k + 1,2*k + 1))

    for i in range(len(g_kernel)):
        for j in range(len(g_kernel[0])):
            g_kernel[i,j] = np.ceil(scale * np.exp(-1/(2*sigma**2) * ( (i - (k))**2 + (j - (k))**2 )))
        
    return g_kernel/np.sum(g_kernel)


def kernel_sequence(kernel_size_limit,kernel_sequence_type='FAST',reverse=True):
    """
    return kernel_sequence
    caution this impacts on the processing time and the
    quality of the resulting spectrum
    """

    if kernel_sequence_type == 'FAST':
        kernels = []
        i = 1
        while 2*i**2+1 < kernel_size_limit:
            kernels.append(2*i**2+1)
            i +=1
    
    if kernel_sequence_type == 'SLOW':
        kernels = []
        i = 1
        while 2*i+1 < kernel_size_limit:
            kernels.append(2*i+1)
            i +=1

    if reverse:
        kernels = kernels[::-1]

    return kernels
    

def filter_sequence(filters,filter_limit=0,sequence_type='FAST',gauss_sigma=None,gauss_size=None,reverse=True):
    """
    generate a sequence of 2d filter kernels
    """
    filter_seq      = []
    filter_seq_type = []    

    if sequence_type == 'INPUT':
        for f in filters:
            if isinstance(f,str):
                filter_seq.append(smooth_kernels_2d(f))
                filter_seq_type.append('filter')
            else:
                if hasattr(f,"__len__"): 
                    filter_seq.append(f)
                    filter_seq_type.append('filter')
                else:
                    filter_seq.append(gaussian_kernel(k,sigma=gauss_sigma,scale=gauss_size))
                    filter_seq_type.append('smooth')

    else:
        fsq  = kernel_sequence(filter_limit,sequence_type,reverse)
        for f in fsq:
            filter_seq.append(gaussian_kernel(f,sigma=gauss_sigma,scale=gauss_size))
            filter_seq_type.append('smooth')

    return filter_seq, filter_seq_type


def str_in_strlist(string,strlist):
    """
    """
    isthere    = False
    spt_string = string.split('/')

    for k in range(len(spt_string)):
        for l in range(len(strlist)):                
            if spt_string[k].count(strlist[l]) > 0:
                isthere = True
                break

    return isthere


