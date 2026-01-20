#
#
# Hans-Rainer Kloeckner
#
# MPIfR 2026
#
# History:
# - based on CHECK_SURVEY_SCANS.py
# - after Claudia's commit we have add
#    RFI versus mean flagging
#    baseline flagging
# - complete rewritten and consolidate
#   all the functions HRK 12/2025
# - added baseline fit and subtraction 01/26
#
#
import h5py
import sys
from copy import copy
import numpy as np
import numpy.ma as ma   # mask array stuff 
#
from optparse import OptionParser
#
from astropy.coordinates import SkyCoord
from astropy.time import Time
from time import process_time
#
# lib in the directory
import MPG_HDF5_libs as MPGHD 
import RFI_WIPER_LIBRARY as RFIL
import SKAMPI_TOOLS as ST
import SKAMPI_TOOLS_PLOT as STP



def main():
    

    # argument parsing
    #
    # ----
    parser       = new_argument_parser()
    (opts, args) = parser.parse_args()

    if opts.datafile == None or opts.help:
        parser.print_help()
        sys.exit()

    # set the parameters
    #
    data_file                 = opts.datafile
    use_data                  = opts.usedata
    use_noise_data            = opts.usenoisedata
    use_scan                  = opts.usescan
    not_use_chan_range        = opts.notusechanrange

    #donothavyflag             = opts.donothavyflag
    
    flagprocessing            = opts.flagprocessing
    dofgbyhand                = opts.hand_fg
    # 1d flagging input
    time_fg_sigma             = opts.time_fg_sigma
    saturation_fg_sigma       = opts.saturation_fg_sigma
    noise_fg_sigma            = opts.noise_fg_sigma
    growthrate_fg_sigma       = opts.growthrate_fg_sigma
    smooth_fg_sigma           = opts.smooth_fg_sigma
    smooth_thresh_fg_sigma    = opts.smooth_thresholding_fg_sigma
    bslf_fg_sigma             = opts.bslf_fg_sigma
    # 2d flagging input
    wf_bound_fg_sigma         = opts.wf_bound_fg_sigma
    wtbysmoothingrow_fg_sigma = opts.wtbysmoothingrow_fg_sigma
    wtbyfilter_fg_sigma       = opts.wtbyfilter_fg_sigma
    #
    docleanup_mask            = opts.docleanup_mask
    #
    doplot_final_spec         = opts.doplot_final_spec
    doplot_final_full_data    = opts.doplot_final_full_data
    doplot_with_invert_mask   = opts.invert_mask
    doplotobs                 = opts.doplotobs
    fspec_yrange              = opts.fspec_yrange
    pltsave                   = opts.pltsave
    savefinalspectrum         = opts.savefinalspectrum
    #
    reset_flag                = opts.reset_flag
    change_flag               = opts.change_flag
    savemask                  = opts.savemask
    loadmask                  = opts.loadmask
    #
    savebslfitspectrum        = opts.savebslfitspectrum    
    usebslfitspectrum         = opts.usebslfitspectrum
    #
    toutput                   = opts.toutput
    #
    velo_fg_sigma             = opts.velo_fg_sigma
    rad_dec_scan              = opts.rad_dec_scan
    getobsinfo                = opts.getobsinfo

    do_rfi_report = -1 # switch this off not used at the moment
    #do_rfi_report             = opts.do_rfi_report
    #do_rfi_report_sigma       = opts.do_rfi_report_sigma

    if toutput:
        print('\n== Flagging SKAMPI Data == \n')


    # load the hdf5 file
    #
    #
    obsfile = h5py.File(data_file)

    # extract OBSID 
    #    
    obs_id = obsfile.attrs['OBSID']

    if toutput:
        print('\n=== Investigate masking of the data in ',data_file)


    # get the data keys of the timestamping in the scans and spectra
    #
    timestamp_keys       = MPGHD.findkeys(obsfile,keys=['scan','timestamp'],exactmatch=True)
    #
    spectrum_keys        = MPGHD.findkeys(obsfile,keys=['scan','spectrum'],exactmatch=True)
    #

    if len(use_scan) == 0:
        scan_keys  = MPGHD.get_obs_info(timestamp_keys,info_idx=1)
    else:
        scan_keys  = np.unique(eval(use_scan))

    if len(use_data) == 0:
        data_keys = MPGHD.get_obs_info(MPGHD.get_obs_info(spectrum_keys,info_idx=2),info_idx=0,splittype='_')
    else:
        data_keys = np.unique(eval(use_data))

    if len(use_noise_data) == 0:
        noise_keys = MPGHD.get_obs_info(MPGHD.get_obs_info(spectrum_keys,info_idx=2),info_idx=1,splittype='_')
    else:
        noise_keys = np.unique(eval(use_noise_data))

    # define what to flag and to plot
    #
    use_data_fg               = data_keys #RFIL.cleanup_strg_input(use_data)
    use_noise_data            = noise_keys #RFIL.cleanup_strg_input(use_noise_data)
    use_scan                  = scan_keys #RFIL.cleanup_strg_input(use_scan)
    #
    plot_type                 = use_noise_data
    dofgbyhand                = eval(dofgbyhand)    
    #
    # #########  


    # ---------------------------------------------------------------------------------------------
    # Get some info about the observations
    # ---------------------------------------------------------------------------------------------
    if getobsinfo:
        ST.observation_info(obsfile,timestamp_keys,spectrum_keys,use_data_fg,plot_type,scan_keys,prt_info=True)
        sys.exit(-1)
    

    # ---------------------------------------------------------------------------------------------
    # Do the spectrum flagging of the data set
    # ---------------------------------------------------------------------------------------------

    # Here we get the default setting useful for SKAMPI in S-Band
    #
    para_code_input      = ST.get_json('RFI_WIPER_SETTINGS.json')

    
    # Set some input settings 
    #
    flag_on              = use_data                                # only use noise diode off to generate flags ['ND0','ND1'] would do all

    #
    # --------------------------------------------


    # time the FG processing
    #
    full_fg_time   = process_time()
    #
    full_new_mask = {}
    save_bslfit_spectra_data = {}
    #
    for d in timestamp_keys:

             #
             # select data base on input
             #
             if RFIL.str_in_strlist(d,use_data_fg) and RFIL.str_in_strlist(d,plot_type) and RFIL.str_in_strlist(d,scan_keys):

                time_data       = obsfile[d][:]
                #freq            = obsfile[d.replace('timestamp','frequency')][:][1:] # exclude the DC term
                #
                waterfall_data  = obsfile[d.replace('timestamp','')+'spectrum']
                new_mask        = np.zeros(waterfall_data.shape).astype(bool)

                if len(usebslfitspectrum) > 0:
                    if toutput:
                        print('\n\tWill use : ',usebslfitspectrum,' for bandpass subtraction.\n')

                    bslfit_data    = RFIL.load_data(usebslfitspectrum)[d.replace('timestamp','')]['baseline_fit_intensity']


                    # Subtract the baseline fit of the averaged spectrum from the waterfall data
                    #
                    waterfall_data = waterfall_data - bslfit_data[np.newaxis,:]

                if toutput:
                    print('\n\tgenerate mask for : ',d.replace('timestamp',''),'\n')


                # =============================================================================================

                
                # ---------------------------------------------------------------------------------------------
                # Mask the first and last channel 
                # ---------------------------------------------------------------------------------------------

                new_mask[:,0]    = True                    # exclude the DC term of the FFT spectrum in the full spectrum
                new_mask[:,-1]   = True                    # exclude the last channel of the spectrum


                # ---------------------------------------------------------------------------------------------
                # Define channel range not to be used 
                # ---------------------------------------------------------------------------------------------
                
                if len(eval(not_use_chan_range)) > 0:
                    blank_channels = eval(not_use_chan_range)
                    if blank_channels[1] > np.shape(new_mask)[1]:
                        blank_channels[1] =  np.shape(new_mask)[1]
                        
                    new_mask[:,blank_channels[0]+1:blank_channels[1]] = True
                    

                # ---------------------------------------------------------------------------------------------
                # spectrum flagging by input (BLC, TRC input)
                #
                # (flagged times will not be used in the spectrum flagging and is present in the final mask)
                # ---------------------------------------------------------------------------------------------

                if len(dofgbyhand) > 0:

                    az = obsfile[d.replace('timestamp','azimuth')][:]
                    el = obsfile[d.replace('timestamp','elevation')][:]

                    if toutput:
                            print('\t- Hand FG in time')
                    
                    for hfg in dofgbyhand:
                        new_mask[hfg[0][1]:hfg[1][1],hfg[0][0]:hfg[1][0]] = True
                        

                        if toutput:
                            print('\t\t idx: ',hfg)
                            # time info
                            time_range  = Time([time_data[hfg[0][1]],time_data[hfg[1][1]]],format='unix')
                            print('\t\t\t timerange: ',*time_range.to_value('isot'))
                            print('\t\t\t azimut range: ',az[hfg[0][1]],az[hfg[1][1]])
                            print('\t\t\t elevation range: ',el[hfg[0][1]],el[hfg[1][1]])


                # ---------------------------------------------------------------------------------------------
                # spectrum flagging individual times (by determine the mean and derivations of the ampl)
                #
                # DO NOT USE THE NOISE DIODE DATA HERE
                #
                # (flagged times will not be used in the spectrum flagging and is present in the final mask)
                # ---------------------------------------------------------------------------------------------
                
                if time_fg_sigma > 0:

                    # Here get the stats_type [default is madmean] to determine the mean and std
                    #
                    stats_type_auto_time = para_code_input['stats_type_flagging']['stats_type_auto_time']

                    if RFIL.str_in_strlist(d,flag_on):

                        waterfall_masked             = ma.masked_array(waterfall_data,mask=new_mask,fill_value=np.nan)
                        amplitude_time_data_masked   = np.median(waterfall_masked,axis=1)
                        
                        time_mask                    = RFIL.threshold_data(data=amplitude_time_data_masked.data,reference_data=amplitude_time_data_masked.compressed(),\
                                                                            data_mask=amplitude_time_data_masked.mask,sigma=time_fg_sigma,stats_type=stats_type_auto_time,mask_true_flag=True)

                        fg_time = 0 
                        for i in range(len(time_mask)):
                            if time_mask[i] == True:
                                new_mask[i,:] = True
                                fg_time +=1
                                
                        if toutput:
                            print('\t- FG in time')
                            print('\t\t flagged: ',fg_time,'time steps')


                # ---------------------------------------------------------------------------------------------
                # spectrum flagging by saturation information use threshold
                #
                # (flagged times will not be used in the spectrum flagging and is present in the final mask)
                # ---------------------------------------------------------------------------------------------

                if saturation_fg_sigma > 0:

                    # Here get the stats_type [default is madmedian] to determine the mean and std
                    #
                    stats_type_saturation = para_code_input['stats_type_flagging']['stats_type_saturation']
                    
                    satur = obsfile[d.replace('timestamp','saturated_samples')][:]
                    satur = satur.flatten()

                    # generate 1-dim mask in time from the 2-dim new_mask
                    #
                    full_mask        = ma.masked_array(new_mask.astype(int),mask=new_mask)
                    satur_time_mask  = np.sum(full_mask,axis=1).mask

                    
                    # determine new mask by thresholding saturation
                    satur_time_mask_mask = RFIL.threshold_data(data=satur,reference_data=satur,data_mask=satur_time_mask,\
                                                            sigma=saturation_fg_sigma,stats_type=stats_type_saturation,mask_true_flag=True)

                    
                    fg_sat = 0 
                    for i in range(len(satur_time_mask_mask)):
                            if satur_time_mask_mask[i] == 1:
                                new_mask[i,:] = True
                                fg_sat += 1
                                
                    if toutput:
                            print('\t- Saturation FG in time')
                            print('\t\t flagged: ',fg_sat,'time steps')

                            
                # ---------------------------------------------------------------------------------------------
                # flagging by scan velocity use thresholds
                #
                # (flagged times will not be used in the spectrum flagging and is present in the final mask)
                # ---------------------------------------------------------------------------------------------

                if velo_fg_sigma > 0:

                    if rad_dec_scan == False:
                        sc_data_x = obsfile[d.replace('timestamp','azimuth')][:]
                        sc_data_y = obsfile[d.replace('timestamp','elevation')][:]
                    else:
                        sc_data_x = obsfile[d.replace('timestamp','dec')][:]
                        sc_data_y  = obsfile[d.replace('timestamp','ra')][:]

                    # Determine the velocity
                    #
                    scan_velo_vec = np.sqrt(np.abs(np.gradient(sc_data_x)) + np.abs(np.gradient(sc_data_y)))


                    # Here get the stats_type [default is madmedian] to determine the mean and std
                    #
                    stats_type_velocity = para_code_input['stats_type_flagging']['stats_type_velocity']
                    
                    # generate 1-dim mask in time from the 2-dim new_mask
                    #
                    full_mask    = ma.masked_array(new_mask.astype(int),mask=new_mask)
                    time_mask    = np.sum(full_mask,axis=1).mask

                    # determine new mask by thresholding saturation
                    time_mask_velo = RFIL.threshold_data(data=scan_velo_vec,reference_data=scan_velo_vec,\
                                                             data_mask=time_mask,sigma=velo_fg_sigma,stats_type=stats_type_velocity,mask_true_flag=True)


                    fg_velo = 0
                    for i in range(len(time_mask_velo)):
                            if time_mask_velo[i] == True:
                                new_mask[i,:] = True
                                fg_velo += 1 
                     
                    if toutput:
                            print('\t- Scanning velocity FG in time')
                            print('\t\t flagged: ',fg_velo,'time steps')


                # ---------------------------------------------------------------------------------------------
                # flagging by relation of noise and strengh of the signal, averages the waterfall in time and 
                # determine the statistics per channel
                # ---------------------------------------------------------------------------------------------
                
                if noise_fg_sigma > 0:

                    # Here get the stats_type 
                    #
                    stats_type_noise         = para_code_input['stats_type_flagging']['stats_type_noise']
                    stats_type_threshold     = para_code_input['stats_type_flagging']['stats_type_threshold']
                    sigma_threshold          = para_code_input['stats_type_flagging']['sigma_noise_threshold']
                    #

                    lf_mask = RFIL.flag_waterfall_by_noise_variation_in_channels(waterfall_data,new_mask,\
                                                                                     sigma_threshold,noise_fg_sigma,\
                                                                                     stats_type_threshold,stats_type_noise,mask_true_flag=True)
                    
                    #
                    rfi_std_mean_fg_chan = 0
                    for i in range(len(lf_mask)):
                            if lf_mask[i] == 1:
                                new_mask[:,i] = True
                                rfi_std_mean_fg_chan += 1
                                
                    if toutput:
                            print('\t- Noise-std-stats FG in channels')
                            print('\t\t flagged: ',rfi_std_mean_fg_chan,'channels')

                #
                # =============================================================================================

                # =============================================================================================
                #

                # ---------------------------------------------------------------------------------------------
                # flag the waterfall spectrum (2d) by upper and lower boundary
                # ---------------------------------------------------------------------------------------------
                
                if wf_bound_fg_sigma > 0:

                  mask_select_org               = copy(new_mask)
                  #
                  stats_type_wt                 = para_code_input['stats_type_flagging']['stats_type_waterfall']                  
                  waterfall_data_masked         = ma.masked_array(waterfall_data,mask=mask_select_org,fill_value=np.nan)
                  #
                  mask_ul_bound                 = RFIL.threshold_data(data=waterfall_data,reference_data=waterfall_data_masked.compressed(),\
                                                                  data_mask=mask_select_org,sigma=wf_bound_fg_sigma,stats_type=stats_type_wt,mask_true_flag=True)
                                                                  
                  if toutput:
                      print('\t- Waterfall boundary threshold FG')
                      print('\t\t flags generated: ',np.count_nonzero(mask_ul_bound)-np.count_nonzero(new_mask))

                  new_mask                  = np.logical_or(new_mask,mask_ul_bound)

                            
                # ---------------------------------------------------------------------------------------------
                # flag the waterfall spectrum (2d) by threshold smoothing individual times
                # ---------------------------------------------------------------------------------------------

                
                if wtbysmoothingrow_fg_sigma > 0:
                    
                    # get the setting for the envelope statistics
                    #
                    envelop_bins                  = para_code_input['smoothing']['envelop_xy_spwd']
                    envelop_smooth_kernel         = para_code_input['smoothing']['envelop_xy_smooth_kernel']
                    envelop_smooth_kernel_size    = para_code_input['smoothing']['envelop_xy_smooth_kernel_size']
                    
                    # smoothing settings, type and kernel sizes, sequence
                    #
                    smooth_type_intensity         = para_code_input['smoothing']['smooth_type']
                    #
                    if flagprocessing == 'INPUT':
                        kernels            = para_code_input['smoothing']['sp_wt_kernels_size_INPUT']
                    else:
                        kernel_size_limit  = para_code_input['smoothing']['sp_wt_kernel_size_limit_SEQUENCE']
                        kernels            = RFIL.kernel_sequence(kernel_size_limit,kernel_sequence_type=flagprocessing)


                    if toutput:
                        print('\t- FG waterfall with smoothing and thresholds per timestep')
                        print('\t\t --PROCESSING_TYPE='+flagprocessing)
                        if len(kernels) > 10:
                            print('\t\t used ',len(kernels),' kernels.')
                        else:
                            print('\t\t use the following kernels: ',kernels)

                    wt_mask  = RFIL.flag_waterfall_by_thresholding_with_smoothing_for_each_timestep(waterfall_data,new_mask,\
                                                                                                        flagprocessing,smooth_type_intensity,kernels,\
                                                                                                        envelop_bins,wtbysmoothingrow_fg_sigma,\
                                                                                                        envelop_smooth_kernel,envelop_smooth_kernel_size,toutput)
                    if toutput:
                        print('\t\t flags generated: ',np.count_nonzero(wt_mask)-np.count_nonzero(new_mask))

                    new_mask = copy(np.logical_or(new_mask,wt_mask))


                # ---------------------------------------------------------------------------------------------
                # flag the waterfall spectrum (2d) by either filtering or convolution and thresholding  
                # ---------------------------------------------------------------------------------------------

                if wtbyfilter_fg_sigma > 0:
                    
                    # get the setting
                    #
                    stats_type_wtfilter           = para_code_input['stats_type_flagging']['stats_type_wtsmooth']

                    
                    # This sets the interpolation of maskes arrays
                    # Either use Gaussian (using smoothing) or interpolations via CloughTocher, LinearNDInterpolator
                    interpol_type        = para_code_input['filter']['wt_interpoltype']
                    gauss_sigma_fraction = para_code_input['filter']['wt_interpoltype_setting_gauss_smooth_sigma']
                    gauss_size_fraction  = para_code_input['filter']['wt_interpoltype_setting_gauss_smooth_size']

                    # Note most of the filter gerneate similar results
                    # x will filter RFI and y will filter in time
                    # ['sobelx','scharrx','prewittx','canny']
                    #
                    # or use a gaussian filtering with increasing sigma switched on if flagprocessing 

                    gauss_sigma = int(gauss_sigma_fraction * np.min(np.shape(waterfall_data)))
                    gauss_size  = int(gauss_size_fraction * np.min(np.shape(waterfall_data)))

                    
                    if flagprocessing == 'INPUT':
                        filter_seq, filter_seq_type    = RFIL.filter_sequence(para_code_input['filter']['wt_filter_type_INPUT'],filter_limit=None,sequence_type=flagprocessing)
                    else:
                        wtkernel_size_limit = para_code_input['filter']['wt_kernel_size_limit_SEQUENCE']
                        filter_seq, filter_seq_type    = RFIL.filter_sequence(filters=None,filter_limit=wtkernel_size_limit,sequence_type=flagprocessing,gauss_sigma=gauss_sigma,gauss_size=gauss_size)

                        add_on_filter_to_sequence = para_code_input['filter']['wt_kernel_ADD_ON_SEQUENCE']
                        #
                        if len(add_on_filter_to_sequence) > 0:
                            add_on,add_on_type = RFIL.filter_sequence(add_on_filter_to_sequence,filter_limit=None,sequence_type='INPUT')
                            filter_seq += add_on
                            filter_seq_type += add_on_type

                    
                    if toutput:
                        print('\t- FG waterfall with ',np.unique(filter_seq_type),'and thresholds entire spectrum')
                        print('\t\t CAUTION might be very slooooooooooooow')
                        print('\t\t --PROCESSING_TYPE='+flagprocessing)
                        if len(filter_seq) > 10:
                            print('\t\t used ',len(filter_seq),' filters.')
                        else:
                            print('\t\t use the following filter: ',filter_seq)


                    ft_mask         = RFIL.flag_waterfall_by_filtering_with_convolutional_smoothing(waterfall_data,new_mask,\
                                                                                                           flagprocessing,filter_seq,filter_seq_type,\
                                                                                                           wtbyfilter_fg_sigma,stats_type=stats_type_wtfilter,\
                                                                                                           interpol_type=interpol_type,gauss_sigma=gauss_sigma,gauss_size=gauss_size)
                    if toutput:
                        print('\t\t flags generated: ',np.count_nonzero(ft_mask)-np.count_nonzero(new_mask))

                    new_mask = copy(np.logical_or(new_mask,ft_mask))

                # =============================================================================================


                # =============================================================================================
                #


                # ---------------------------------------------------------------------------------------------
                # flagging on growthrate on the average 1d spectrum  
                # ---------------------------------------------------------------------------------------------

                if growthrate_fg_sigma > 0:

                    # Here get the stats_type [default is madmedian] to determine the mean and std
                    #
                    stats_type_growthrate   = para_code_input['stats_type_flagging']['stats_type_growthrate']
                    #
                    envelop_xy_bins               = para_code_input['smoothing']['envelop_growthrate_spwd']
                    envelop_xy_smooth_kernel      = para_code_input['smoothing']['envelop_xy_smooth_kernel']
                    envelop_xy_smooth_kernel_size = para_code_input['smoothing']['envelop_xy_smooth_kernel_size']

                    
                    # get the single averaged spectrum
                    #
                    spectrum_masked          = ma.masked_array(waterfall_data,mask=new_mask,fill_value=np.nan)
                    #
                    intensity                = spectrum_masked.mean(axis=0)[1:] # note exclude first channel to get base 2 number of channels
                    channels                 = np.arange(len(intensity))

                    
                    # 
                    #
                    gwr_select = RFIL.flag_spectrum_by_growthratechanges(channel=channels,intensity=intensity.data,mask=intensity.mask,\
                                                                             stats_type_threshold=stats_type_growthrate,sigma_threshold=growthrate_fg_sigma,\
                                                                             envelop_bins=envelop_xy_bins,sigma_envelop=growthrate_fg_sigma,\
                                                                                        smooth_type_envelop=envelop_xy_smooth_kernel,\
                                                                                        smooth_kernel_size_envelop=envelop_xy_smooth_kernel_size)
                    #
                    gwr_fg_chan = 0
                    for i in range(len(gwr_select)):
                            if gwr_select[i] == True:
                                new_mask[:,i+1] = True
                                gwr_fg_chan += 1

                    if toutput:
                        print('\t- FG channels on growth rate spectrum')
                        print('\t\t flagged: ',gwr_fg_chan,'channels ')

                        
                    
                # ---------------------------------------------------------------------------------------------
                # flagging by smoothing with various kernel sizes on an average 1d spectrum  
                # ---------------------------------------------------------------------------------------------

                if smooth_fg_sigma > 0:

                    # Here get the stats_type [default is madmedian] to determine the mean and std
                    #
                    smooth_type_intensity         = para_code_input['smoothing']['smooth_type']
                    envelop_xy_bins               = para_code_input['smoothing']['envelop_xy_spwd']
                    envelop_xy_smooth_kernel      = para_code_input['smoothing']['envelop_xy_smooth_kernel']
                    envelop_xy_smooth_kernel_size = para_code_input['smoothing']['envelop_xy_smooth_kernel_size']
                    #
                    kernel_size_limit             = para_code_input['smoothing']['sp_wt_kernel_size_limit_SEQUENCE']
                    
                    # get the single averaged spectrum
                    #
                    spectrum_masked          = ma.masked_array(waterfall_data,mask=new_mask,fill_value=np.nan)
                    #
                    intensity                = spectrum_masked.mean(axis=0)[1:] # note exclude first channel to get base 2 number of channels
                    channels                 = np.arange(len(intensity))

                    if flagprocessing != 'INPUT':
                        kernels     = RFIL.kernel_sequence(kernel_size_limit,kernel_sequence_type=flagprocessing)
                    else:
                        kernels     = para_code_input['smoothing']['sp_wt_kernels_size_INPUT']
                        
                    smoo_select = RFIL.flag_spectrum_by_thresholding_with_smoothing(channel=channels,intensity=intensity.data,mask=intensity.mask,\
                                                                                        smooth_type_intensity=smooth_type_intensity,smooth_kernel_sizes_intensity=kernels,\
                                                                               envelop_bins=envelop_xy_bins,sigma_envelop=smooth_fg_sigma,\
                                                                                        smooth_type_envelop=envelop_xy_smooth_kernel,\
                                                                                        smooth_kernel_size_envelop=envelop_xy_smooth_kernel_size)
                    #
                    smoo_fg_chan = 0
                    for i in range(len(smoo_select)):
                            if smoo_select[i] == True:
                                new_mask[:,i+1] = True
                                smoo_fg_chan += 1

                    if toutput:
                        print('\t- FG channels on smooth spectrum')
                        if len(kernels) > 10:
                            print('\t\t kernel: ',smooth_type_intensity,', number of kernels ',len(kernels),flagprocessing)
                        else:
                            print('\t\t kernel: ',smooth_type_intensity,', size ',kernels,' Type ',flagprocessing)
                        print('\t\t\t SWPD  : ',envelop_xy_bins)
                        print('\t\t\t boundary smoothing kernel: ',envelop_xy_smooth_kernel,', size ',envelop_xy_smooth_kernel_size)
                        print('\t\t flagged: ',smoo_fg_chan,'channels ')


                # ---------------------------------------------------------------------------------------------
                # flagging by smoothing with various kernel using pur thresholding on 1d spectrum  (based on Tobias Winchen functions) 
                # ---------------------------------------------------------------------------------------------

                if smooth_thresh_fg_sigma > 0:

                    # Here get the stats_type [default is madmedian] to determine the mean and std
                    #
                    window_size_start_thresholding         = para_code_input['smoothing']['window_size_start_thresholding']
                    
                    # get the single averaged spectrum
                    #
                    spectrum_masked          = ma.masked_array(waterfall_data,mask=new_mask,fill_value=np.nan)
                    #
                    intensity                = spectrum_masked.mean(axis=0)[1:] # note exclude first channel to get base 2 number of channels

                    window_size_start        = window_size_start_thresholding
                    smoo_select              = RFIL.flag_spectrum_thresholding(intensity.data, intensity.mask, window_size_start, smooth_thresh_fg_sigma, replace='minimum')

                    #
                    smoo_fg_chan = 0
                    for i in range(len(smoo_select)):
                            if smoo_select[i] == True:
                                new_mask[:,i+1] = True
                                smoo_fg_chan += 1

                    if toutput:
                        print('\t- FG channels on smoothed thresholding spectrum (Tobi)')
                        print('\t\t flagged: ',smoo_fg_chan,'channels ')



                # ---------------------------------------------------------------------------------------------
                # flagging based 1d spectrum baseline fit 
                # ---------------------------------------------------------------------------------------------
                
                if bslf_fg_sigma > 0:
                    
                    # get the setting
                    #
                    stats_type_baselinefit  = para_code_input['stats_type_flagging']['stats_type_baselinefit']
                    #
                    envelop_xy_bins               = para_code_input['smoothing']['envelop_xy_spwd']
                    envelop_xy_smooth_kernel      = para_code_input['smoothing']['envelop_xy_smooth_kernel']
                    envelop_xy_smooth_kernel_size = para_code_input['smoothing']['envelop_xy_smooth_kernel_size']
                    #
                    lam_factor                    = para_code_input['bsl_fit']['lam_factor']
                    set_bslf_func                 = para_code_input['bsl_fit']['set_bslf_func']

                    # get the single averaged spectrum
                    #
                    spectrum_masked          = ma.masked_array(waterfall_data,mask=new_mask,fill_value=np.nan)
                    #
                    intensity                = spectrum_masked.mean(axis=0)[1:] # note exclude first channel to get base 2 number of channels
                    channels                 = np.arange(len(intensity))

                    # determine the number of spwd
                    #
                    splitting_in_bins_base_2 = (np.log(len(channels)/envelop_xy_bins)/np.log(2))


                    bslf_mask,bsl_fit_info,baseline_fit_intensity  = RFIL.flag_spectrum_by_basline_fitting(channels,intensity.data,intensity.mask,stats_type_baselinefit,\
                                                                                        envelop_bins=splitting_in_bins_base_2,sigma_envelop=bslf_fg_sigma,\
                                                                                smooth_type_envelop=envelop_xy_smooth_kernel,\
                                                                                        smooth_kernel_size_envelop=envelop_xy_smooth_kernel_size,set_bslf_func=set_bslf_func,lam_factor=lam_factor)



                    if len(savebslfitspectrum) > 0:
                        baseline_fit_intensity = np.insert(baseline_fit_intensity,0,0)
                        
                        save_bslfit_spectra_data[d.replace('timestamp','')] = {}
                        save_bslfit_spectra_data[d.replace('timestamp','')]['baseline_fit_intensity'] = baseline_fit_intensity

                    
                    
                    # here do the flagging
                    #
                    bslf_fg_chan = 0
                    for i in range(len(bslf_mask)):
                            if bslf_mask[i] == True:
                                new_mask[:,i+1] = True
                                bslf_fg_chan += 1
                                
                    if toutput:
                        print('\t- FG channels on baseline fit ',bsl_fit_info)
                        print('\t\t flagged channels: ',bslf_fg_chan)

                #
                # =============================================================================================
                                    


                # ---------------------------------------------------------------------------------------------
                # flagging based on cleaning up 2d and 1d on channels and times
                # ---------------------------------------------------------------------------------------------

                if docleanup_mask:

                    # 2d cleanup input
                    #
                    fraction_time    = para_code_input['cleaning']['fraction_time']
                    fraction_channel = para_code_input['cleaning']['fraction_channel']
                    #
                    new_mask = RFIL.clean_up_2d_mask(new_mask,fraction_time=0.7,fraction_channel=0.7,mask_true_flag=True)

                    # 1d cleanup input
                    #
                    # clean up mask of some pattern to exclude single grouping of channels
                    # can define this in the settings file
                    #
                    clean_bins_freq = para_code_input['cleaning']['clean_bins_freq']
                    clean_bins_time = para_code_input['cleaning']['clean_bins_time']


                    # Clean up in time
                    #
                    mask_in_time         = np.prod(new_mask.astype(int),axis=1)
                    clean_in_time_select = RFIL.clean_up_1d_mask(mask_in_time,clean_bins_time,setvalue=1)
                    
                    # here do the flagging
                    #
                    clean_in_times = 0
                    for i in range(len(clean_in_time_select)):
                            if clean_in_time_select[i] == 1:
                                new_mask[i,:] = True
                                clean_in_times += 1

                                
                    # Clean up in frequency
                    #
                    mask_in_freq         = np.prod(new_mask.astype(int),axis=0)
                    clean_in_freq_select = RFIL.clean_up_1d_mask(mask_in_freq,clean_bins_freq,setvalue=1)

                    # here do the flagging
                    #
                    clean_in_freq = 0
                    for i in range(len(clean_in_freq_select)):
                            if clean_in_freq_select[i] == 1:
                                new_mask[:,i] = True
                                clean_in_freq += 1
                                
                    if toutput:
                            print('\t- Clean up final 2 d mask')
                            for p in range(len(clean_bins_time)):
                                    print('\t\t with clean_pattern',clean_bins_time[p])
                            print('\t\t flagged: ',clean_in_times,'times')
                            for p in range(len(clean_bins_freq)):
                                    print('\t\t with clean_pattern',clean_bins_freq[p])
                            print('\t\t flagged: ',clean_in_freq,'frequencies')

                #
                # =============================================================================================

                            

                # Here we store the updated new_mask                        
                #
                if len(eval(not_use_chan_range)) > 0:
                    # here we include the previous channels not to do the flagging on
                    #
                    blank_channels = eval(not_use_chan_range)
                    new_mask[:,blank_channels[0]+1:blank_channels[1]] = False                    
                    #
                    new_mask[:,0]    = True                    # exclude the DC term of the FFT spectrum in the full spectrum
                    new_mask[:,-1]   = True                    # exclude the DC term of the FFT spectrum in the full spectrum

                    
                full_new_mask[d.replace('timestamp','')]      = new_mask
                #
                # =============================================================================================

                
    # save the baseline fit for bandpass subtraction
    #
    if len(savebslfitspectrum) > 0:
        if toutput:
                #print('\n   === Save bslfit spectra into a numpy-z file ',savebslfitspectrum,' === \n')
                print('\n   === Save bslfit spectra into a pickle file ',savebslfitspectrum,' === \n')

        #np.savez(savebslfitspectrum,**save_bslfit_spectra_data)
        RFIL.save_data(savebslfitspectrum,save_bslfit_spectra_data)
        
    # determine the full fg time required 
    full_fg_elapsed_time = process_time() - full_fg_time
    if toutput:
        print('\n\tFull masking needed ',full_fg_elapsed_time,' [s]')

    # ---------------------------------------------------------------------------------------------
    # Merge the mask into a single one ONLY IF BOTH CHANNELS ARE EQUAL
    # ---------------------------------------------------------------------------------------------

    keys = full_new_mask.keys()

    check_shapes = 1
    for i,k in enumerate(keys):
        mask_shape = full_new_mask[k].shape
        if i == 0:
            t_shape    = mask_shape[0]
            f_shape    = mask_shape[1]
        else:
            if t_shape - mask_shape[0] != 0:
                check_shapes = -1
            if f_shape - mask_shape[1] != 0:
                check_shapes = -1

    if check_shapes == 1:
        final_maskcomb = copy(new_mask)

        for k in keys:
                final_maskcomb = np.logical_or(full_new_mask[k],final_maskcomb)

        final_mask = {}
        for d in timestamp_keys:
                final_mask[d.replace('timestamp','')] = final_maskcomb
    else:
        print('CAUTON both channels have different dimensions')
        print('Generate individual mask')
        
        final_mask = {}
        for d in timestamp_keys:
            if RFIL.str_in_strlist(d,use_data_fg) and RFIL.str_in_strlist(d,plot_type) and RFIL.str_in_strlist(d,scan_keys):
                final_mask[d.replace('timestamp','')] = full_new_mask[d.replace('timestamp','')]

    # ---------------------------------------------------------------------------------------------
    # Save the mask
    # ---------------------------------------------------------------------------------------------

    if len(savemask) > 0:
        if toutput:
            print('\n   === Save the mask into a numpy-z file ',savemask,' === \n')
 
        full_masks = {}
        full_masks['final_comb_mask'] = final_mask
        full_masks['final_mask']      = full_new_mask
        
        np.savez(savemask, **full_masks)


    # ---------------------------------------------------------------------------------------------
    # Load the mask
    # ---------------------------------------------------------------------------------------------

    if len(loadmask) > 0:
        if toutput:
            print('\n   === Load in mask from ',loadmask,' === \n')

        # Load the dictionary
        loaded_dict = np.load(loadmask+'.npz', allow_pickle=True)

        # retrive the mask 
        full_new_mask = loaded_dict['final_mask'].item()
        final_mask    = loaded_dict['final_comb_mask'].item()


    # ---------------------------------------------------------------------------------------------
    # Plot the spectrum of the data set
    # ---------------------------------------------------------------------------------------------
    #
    if doplot_final_spec:

        if toutput:
            print('\n   === Generate 1d Spectrum plot === \n')

        # Store the final spectra to be optional saved
        #
        plt_final_spectra_data = {}
        #
        for d in timestamp_keys:

            if RFIL.str_in_strlist(d,plot_type) and RFIL.str_in_strlist(d,use_data_fg) and RFIL.str_in_strlist(d,scan_keys):

                plt_waterfall_data  = obsfile[d.replace('timestamp','')+'spectrum'] 
                freq                = obsfile[d.replace('timestamp','frequency')][:]

                if doplot_with_invert_mask:
                    f_mask         = np.invert(final_mask[d.replace('timestamp','')])
                else:
                    f_mask         = final_mask[d.replace('timestamp','')]

                fg_info = int(100*np.sum(f_mask.astype(int))/np.prod(f_mask.shape))
                
                if toutput:
                    print('\tmask:               ',d.replace('timestamp',''),' ',fg_info,' %')
                    print('\tgenerate plot for : ',d.replace('timestamp',''))

                fullmask_data  = ma.masked_array(plt_waterfall_data,mask=f_mask,fill_value=np.nan)
                spectrum_mean  = fullmask_data.mean(axis=0)[:]
                spectrum_std   = fullmask_data.std(axis=0)[:]
                spectrum_mask  = spectrum_mean.mask[:]

                # safe the spectra in dic
                #
                if len(savefinalspectrum) > 0:
                    plt_final_spectra_data[d.replace('timestamp','')] = {}
                    plt_final_spectra_data[d.replace('timestamp','')]['spectrum_mean'] = spectrum_mean
                    plt_final_spectra_data[d.replace('timestamp','')]['spectrum_mask'] = spectrum_mask
                    plt_final_spectra_data[d.replace('timestamp','')]['spectrum_std']  = spectrum_std
                    plt_final_spectra_data[d.replace('timestamp','')]['freq']          = freq
                    plt_final_spectra_data[d.replace('timestamp','')]['obs_id']        = obs_id
                    plt_final_spectra_data[d.replace('timestamp','')]['time_data']     = time_data

                
                # print the spectrum
                #
                title = 'obsid: '+str(obs_id)+' '+d.replace('timestamp','')
                #
                plt_fname = data_file.replace('..','').replace('/','').replace('.hdf5','').replace('.HDF5','')+'_'+d.replace('timestamp','').replace('/','_')+'SPEC'                
                STP.plot_spectrum(freq,spectrum_mean,spectrum_std,title,fspec_yrange,pltsave=pltsave,plt_fname=plt_fname)


        if len(savefinalspectrum) > 0:
            if toutput:
                print('\n   === Save final spectra into a numpy-z file ',savefinalspectrum,' === \n')

            np.savez(savefinalspectrum,**plt_final_spectra_data)


    # ---------------------------------------------------------------------------------------------
    # Plot the waterfall spectrum of the data set
    # ---------------------------------------------------------------------------------------------
    #
    if doplot_final_full_data:

        if toutput:
            print('\n   === Generate waterfall plot === \n')

        #
        for d in timestamp_keys:
            
            if RFIL.str_in_strlist(d,plot_type) and RFIL.str_in_strlist(d,use_data_fg) and RFIL.str_in_strlist(d,scan_keys):

                plt_waterfall_data  = obsfile[d.replace('timestamp','')+'spectrum'][:]

                if doplot_with_invert_mask:
                    f_mask         = np.invert(final_mask[d.replace('timestamp','')])
                else:
                    f_mask         = final_mask[d.replace('timestamp','')]


                fg_info = int(100*np.sum(f_mask.astype(int))/np.prod(f_mask.shape))
                
                if toutput:
                    print('\tmask:               ',d.replace('timestamp',''),' ',fg_info,' %')
                    print('\tgenerate plot for : ',d.replace('timestamp',''))


                # print the waterfall plot
                #
                fullmask_data           = ma.masked_array(plt_waterfall_data,mask=f_mask,fill_value=np.nan)

                plt_fname = data_file.replace('..','').replace('/','').replace('.hdf5','').replace('.HDF5','')+'_'+d.replace('timestamp','').replace('/','_')+'WFPLT'
                title     = 'obsid: '+str(obs_id)+' '+d.replace('timestamp','')
                #
                STP.plot_waterfall_spectrum(fullmask_data,d,title,pltsave=pltsave,plt_fname=plt_fname)

    #
    #
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    # Plot the azimuth elevation of the data set
    # ---------------------------------------------------------------------------------------------
    #
    if doplotobs:

        if toutput:
            print('\n   === Generate obsscan plot === \n')


        data_x, data_y,data_c = [],[],[]
        for d in timestamp_keys:
            
            if RFIL.str_in_strlist(d,plot_type) and RFIL.str_in_strlist(d,use_data_fg) and RFIL.str_in_strlist(d,scan_keys):

                plt_waterfall_data  = obsfile[d.replace('timestamp','')+'spectrum'][:]

                
                if doplot_with_invert_mask:
                    f_mask         = np.invert(final_mask[d.replace('timestamp','')])
                else:
                    f_mask         = final_mask[d.replace('timestamp','')]

                fullmask_data  = ma.masked_array(plt_waterfall_data,mask=f_mask,fill_value=np.nan)
                spectrum_mean  = fullmask_data.mean(axis=1)[:]

                if rad_dec_scan == False:
                    az          = obsfile[d.replace('timestamp','azimuth')][:]
                    el          = obsfile[d.replace('timestamp','elevation')][:]
                else:
                    az          = obsfile[d.replace('timestamp','ra')][:]
                    el          = obsfile[d.replace('timestamp','dec')][:]
                
                sort_it        = np.argsort(spectrum_mean)
                
                data_x  = np.concatenate((data_x,az[sort_it][:]),axis=None)
                data_y  =  np.concatenate((data_y,el[sort_it][:]),axis=None)
                data_c  = np.concatenate((data_c,spectrum_mean[sort_it][:]),axis=None)

                
        plt_fname = data_file.replace('..','').replace('/','').replace('.hdf5','').replace('.HDF5','')+'_'+d.replace('timestamp','').replace('/','_')+'OBS'
        #
        STP.plot_observation(data_x,data_y,data_c,rad_dec_scan,'obsid: '+str(obs_id),pltsave,plt_fname)                
    #
    #
    # ---------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------
    # Update the mask in the original input file ! CAUTION CAUTION CAUTION 
    # ---------------------------------------------------------------------------------------------
    #
    #
    if change_flag:
        print('\nCAUTION CAUTION CAUTION FILE ',data_file,' WILL BE EDITED')
        obsfile.close()  # first close the open input file

        with h5py.File(data_file, 'r+') as infile:
            for d in timestamp_keys: 
                print('\tupdate mask: ',d.replace('timestamp',''))
                if RFIL.str_in_strlist(d,use_data_fg) and RFIL.str_in_strlist(d,scan_keys):
                    infile[d.replace('timestamp','')+'mask'][:] = final_mask[d.replace('timestamp','')].astype(bool)
        print('\n... old masks have been replaced!\n')
        sys.exit(-1)
    #
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    # erase the mask in the original input file ! CAUTION CAUTION CAUTION 
    # ---------------------------------------------------------------------------------------------
    #
    #
    if reset_flag:
        print('\n\tCAUTION CAUTION CAUTION FILE ',data_file,' WILL BE EDITED\n')
        obsfile.close()  # first close the open input file

        with h5py.File(data_file, 'r+') as infile:
            for d in timestamp_keys: 

                if RFIL.str_in_strlist(d,use_data_fg) and RFIL.str_in_strlist(d,scan_keys):

                    # get the mask
                    mask_data  = infile[d.replace('timestamp','mask')][:] 

                    print('\terase mask: ',d.replace('timestamp',''))
                    infile[d.replace('timestamp','')+'mask'][:] = np.ones(mask_data.shape).astype(bool)

        print('\n\t... masks have been erased!\n')
        sys.exit(-1)
    #
    # ---------------------------------------------------------------------------------------------


def new_argument_parser():

    #
    # some input for better playing around with the example
    #
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)


    parser.add_option('--DATA_FILE', dest='datafile', type=str,
                      help='DATA - HDF5 file of the Prototyp')

    parser.add_option('--USE_DATA', dest='usedata', type=str,default="[\'P0\',\'P1\',\'S0\']",
                      help='data to flag, default use all or select e.g. \"[\'P0\',\'P1\']\" or \"[\'S0\']\"')

    parser.add_option('--USE_NOISEDATA', dest='usenoisedata', type=str,default="[\'ND0\']",
                      help='use data noise diode on and off "[\'ND0\',\'ND1\']", default is \"[\'ND0\']\"')

    parser.add_option('--USE_SCAN', dest='usescan', type=str,default='',
                      help='select scan to flag, default are all scans, to choose scan 000 and 001 use e.g. \"[\'000\',\'001\']\"')

    parser.add_option('--NOTUSE_CHANELRANGE', dest='notusechanrange', type=str,default='[]',
                      help='exclude channel range from flagging procedure e.g. \'[0,2500]\' ')
       
    parser.add_option('--FG_BY_HAND', dest='hand_fg', type=str,default='[]',
                      help='Flag selection [[BLC [channel,time],TRC [channel, time]], ...] in the waterfallplot e.g [[[0,10],[5000,110]],[[10000,500],[15000,510]]]')

    parser.add_option('--FG_TIME_SIGMA', dest='time_fg_sigma', type=float,default=0,
                      help='determine bad time use threshold on amplitude. default = 0 is off use e.g. = 5')

    parser.add_option('--FG_SATURATION_SIGMA', dest='saturation_fg_sigma', type=float,default=0,
                      help='use the saturation information to flag. default = 0 is off use e.g. = 3')

    parser.add_option('--FG_VELO_SIGMA', dest='velo_fg_sigma', type=float, default=0,
                      help='determine flags based on the scanning velocity outlieres on sky. [default = 0 is off use e.g. = 6]')

    parser.add_option('--FG_NOISE_SIGMA', dest='noise_fg_sigma', type=float, default=0,
                      help='determine flags based on the linear relation of noise and power. [default = 0 is off use e.g. = 3]')

    parser.add_option('--FG_GROWTHRATE_SIGMA', dest='growthrate_fg_sigma', type=float, default=0,
                      help='determine flags based on the growthrate function outliers. [default = 0 is off use e.g. = 6]')

    parser.add_option('--FG_SMOOTH_SIGMA', dest='smooth_fg_sigma', type=float, default=0,
                      help='determine flags based on increase smooth kernel and thresholding on difference org smooth spectra, use also PROCESSING_TYPE see RFI_SETTINGS.json. [default = 0 is off use e.g. = 3]')

    parser.add_option('--FG_SMOOTH_THRESHOLDING_SIGMA', dest='smooth_thresholding_fg_sigma', type=float, default=0,
                      help='determine flags based on smooth thresholding spectrum (Tobi), see RFI_SETTINGS.json. [default = 0 is off use e.g. = 6]')

    parser.add_option('--FG_WT_SMOOTHING_SIGMA', dest='wtbysmoothingrow_fg_sigma', type=float, default=0,
                      help='determine flags based on smoothing and thresholding the waterfall spectrum in each time step, see RFI_SETTINGS.json. [VERY SLOW, default = 0 is off use e.g. = 6]')

    parser.add_option('--FG_WT_FILTERING_SIGMA', dest='wtbyfilter_fg_sigma', type=float, default=0,
                      help='determine flags based on filtering or smoothing and thresholding the entire waterfall spectrum, see RFI_SETTINGS.json. [VERY SLOW, default = 0 is off use e.g. = 3]')

    parser.add_option('--PROCESSING_TYPE', dest='flagprocessing', type=str,default='FAST',
                      help='setting how accurate/much time the flagging proceed. FAST (default), SLOW, INPUT uses the kernels of the RFI_SETTINGS.json file.')

    parser.add_option('--FG_WT_BOUND_SIGMA', dest='wf_bound_fg_sigma', type=float, default=0,
                      help='determine flags based on upper and lower boundary. [e.g. =4, Useful in combination with: --USE_BSLFIT=]')

    parser.add_option('--FG_BSLF_SIGMA', dest='bslf_fg_sigma', type=float, default=0,
                      help='determine flags based on spectral baseline fit. [default = 0 is off use e.g. = 10]')

    parser.add_option('--FG_CLEANUP_MASK', dest='docleanup_mask',action='store_true', default=False,
                      help='Clean up the processed mask, use specific pattern and on the percentage in time and channel, see RFI_SETTINGS.json. [default = False]')

    parser.add_option('--CHANGE_COORDS_TO_AZEL', dest='rad_dec_scan', action='store_false',
                      default=True,help='Switch to Azimuth-Elevation scan type to be used for velocity outlier flag and the plotting.')

    parser.add_option('--PLOT_SPEC', dest='doplot_final_spec', action='store_true',
                      default=False,help='Plot the mean averaged in time spectrum.')

    parser.add_option('--FINAL_SPEC_YRANGE', dest='fspec_yrange', type=str,default='[0,0]',
                      help='[ymin,ymax]')

    parser.add_option('--PLOT_WATERFALL', dest='doplot_final_full_data', action='store_true',
                      default=False,help='Plot the waterfall spectrum')

    parser.add_option('--PLOT_OBS', dest='doplotobs', action='store_true',default=False,
                      help='Plot the observation DO_AZEL_SCAN to switch from RADEC to AZEL')

    parser.add_option('--PLOT_WITH_INVERTED_MASK', dest='invert_mask', action='store_true',
                      default=False,help='Plot the final plots using an inverted mask')

    parser.add_option('--SAVE_PLOT', dest='pltsave', action='store_true',
                      default=False,help='Save the plot as png files.')

    parser.add_option('--EDIT_MASK', dest='change_flag', action='store_true',
                      default=False,help='Replace the original mask in the file with the new mask.')

    parser.add_option('--RESET_MASK', dest='reset_flag', action='store_true',
                      default=False,help='Clear the original mask in the file.')

    parser.add_option('--SAVE_MASK', dest='savemask',type=str,default='',
                      help='Save the mask into numpy npz file.')

    parser.add_option('--LOAD_MASK', dest='loadmask', type=str,default='',
                      help='Load the mask from the numpy npz file.')

    parser.add_option('--SAVE_FINALSPECTRUM', dest='savefinalspectrum', type=str,default='',
                      help='Safe the final 1d spectra as numpy npz file. [works only with --DOPLOT_FINAL_SPEC]')

    parser.add_option('--SAVE_BSLFIT', dest='savebslfitspectrum', type=str,default='',
                      help='Safe the 1d baseline fit spectra as numpy npz file. [works only with --FG_BSLF_SIGMA]')

    parser.add_option('--USE_BSLFIT', dest='usebslfitspectrum', type=str,default='',
                      help='Use the 1d baseline fit spectra as bandpass numpy npz file.')

    parser.add_option('--SILENCE', dest='toutput', action='store_false',
                      default=True,help='Switch off all output')

    parser.add_option('--OBSINFO', dest='getobsinfo', action='store_true',
                      default=False,help='Show observation info stored in the file')

    parser.add_option('--HELP', dest='help', action='store_true',
                      default=False,help='Show info on input')

    return parser


if __name__ == "__main__":
    main()
