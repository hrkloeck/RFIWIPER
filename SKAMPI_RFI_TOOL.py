#
#
# Hans-Rainer Kloeckner
#
# MPIfR 2026
#
# History:
# based on CHECK_SURVEY_SCANS.py
# after Claudia's commit we have add
#    RFI versus mean flagging
#    baseline flagging
#
#
#
import h5py
import numpy as np
import numpy.ma as ma
import multiprocessing
import sys
from optparse import OptionParser
#
import MPG_HDF5_libs as MPGHD 
#
from astropy.time import Time
from astropy import units as u
#
import copy
import pybaselines as pbsl
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import process_time
import RFI_LIB_SCANS as RFIL
# ###################################################
#
# Some plotting parameter 
#
cmap = copy.copy(mpl.cm.cubehelix)
cmap.set_bad(color='black')

im_size  = (8.27, 11.69)[::-1]  # A4 landscape
#im_size  = (8.27, 11.69)       # A4 portrait
#im_size  = (8, 4)       # A4 portrait
#
plt.rcParams['figure.figsize'] = im_size
#
#
# ###################################################


def main():

    import numpy as np

    # setting for the plotting
    DPI = 150


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
    donotflag                 = opts.donotflag
    flagprocessing            = opts.flagprocessing
    dofgbyhand                = opts.hand_time_fg
    time_sigma                = opts.auto_time_fg_sigma
    saturation_fg_sigma       = opts.saturation_fg_sigma
    bound_sigma_input         = opts.bound_sigma_input
    flag_RFI_std_mean_sigma   = opts.flag_RFI_std_mean_sigma
    flag_smooth_sigma         = opts.flag_smooth_sigma
    flag_bslf_sigma           = opts.flag_bslf_sigma
    doplot_final_spec         = opts.doplot_final_spec
    doplot_final_full_data    = opts.doplot_final_full_data
    doplot_with_invert_mask   = opts.invert_mask
    fspec_yrange              = opts.fspec_yrange
    pltsave                   = opts.pltsave
    savefinalspectrum         = opts.savefinalspectrum
    reset_flag                = opts.reset_flag
    change_flag               = opts.change_flag
    savemask                  = opts.savemask
    loadmask                  = opts.loadmask
    donotncpus                = opts.donotncpus
    toutput                   = opts.toutput
    usencpus                  = opts.usencpus
    do_rfi_report             = opts.do_rfi_report
    do_rfi_report_sigma       = opts.do_rfi_report_sigma
    scan_velo_fg_sigma        = opts.scan_velo_fg_sigma
    rad_dec_scan              = opts.rad_dec_scan
    getobsinfo                = opts.getobsinfo

    # define what to plot
    #
    use_data_fg               = RFIL.cleanup_strg_input(use_data)
    use_noise_data            = RFIL.cleanup_strg_input(use_noise_data)
    use_scan                  = RFIL.cleanup_strg_input(use_scan)

    plot_type                 = use_noise_data
    dofgbyhand                = eval(dofgbyhand)


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
        scan_keys  = np.unique(use_scan)
    #
    # #########  



    if getobsinfo:

        
        info_dics = {}

        # get info from attributes of the file
        for a in obsfile.attrs:
            info_dics[a] = obsfile.attrs[a]

        # enlarge info
        info_dics['SCAN'] = list(MPGHD.get_obs_info(timestamp_keys,info_idx=1))
        info_dics['TYPE'] = list(MPGHD.get_obs_info(timestamp_keys,info_idx=2))

        obs_pos = [info_dics['telescope_longitude'],info_dics['telescope_latitude'],info_dics['telescope_height']]


        print('\n======== DATA INFORMATION =========================\n')

        print('\t - Data file ', data_file)

        print('\n\t - General Information ')

        for i in info_dics:
            print('\t\t ',i,info_dics[i])

            
        for d in timestamp_keys:


            if RFIL.str_in_strlist(d,use_data_fg) and RFIL.str_in_strlist(d,plot_type) and RFIL.str_in_strlist(d,scan_keys):
                
                info_data = d.split('/')

                time_data         = obsfile[d][:]
                freq              = obsfile[d.replace('timestamp','frequency')][:][1:] # exclude the DC term
                #
                acu_times         = Time(obsfile[d][:],scale='utc',format='unix').iso
                #
                sc_data_az        = obsfile[d.replace('timestamp','azimuth')][:]
                sc_data_el        = obsfile[d.replace('timestamp','elevation')][:]
                sc_data_ra        = obsfile[d.replace('timestamp','ra')][:]
                sc_data_dec       = obsfile[d.replace('timestamp','dec')][:]
                
                gain              = obsfile[d.replace('timestamp','power')][:]/obsfile[d.replace('timestamp','power').replace(info_data[2],info_data[2].replace('ND0','ND1'))][:]

                stats_gain        = RFIL.data_stats(gain,stats_type='mean',accur=100)
                
                mask_data         = obsfile[d.replace('timestamp','')+'mask']
                masked_percentage = np.count_nonzero(mask_data)/np.cumprod(mask_data.shape)[-1]*100

                
                # determine velocities
                velo_dec = np.gradient(sc_data_dec)/np.gradient(time_data.flatten()) 
                velo_ra  = np.gradient(sc_data_ra)/np.gradient(time_data.flatten())
                velo_az  = np.gradient(sc_data_az)/np.gradient(time_data.flatten())
                velo_el  = np.gradient(sc_data_el)/np.gradient(time_data.flatten()) 
                #
                vstatstyp = 'madmedian'
                stats_ra  = RFIL.data_stats(velo_ra,stats_type=vstatstyp,accur=100)
                stats_dec = RFIL.data_stats(velo_dec,stats_type=vstatstyp,accur=100)
                stats_az  = RFIL.data_stats(velo_az,stats_type=vstatstyp,accur=100)
                stats_el  = RFIL.data_stats(velo_el,stats_type=vstatstyp,accur=100)


                print('\n\t - Scan Info')
                print('\t\tscan ',info_data[1],'type ',info_data[2])

                print('\t\t\t - time range:                 ', min(acu_times)[0],max(acu_times)[0])
                print('\t\t\t - total time:                 ', max(time_data)[0]- min(time_data)[0],' [s]')
                print('\t\t\t - percentage masked:          ', masked_percentage, '[%]')
                print('\t\t\t - gain un-masked:             ', *stats_gain[:2], '[mean, std]')
                
                print('\t\t\t - Azimuth [min, max, velo]:   ',min(sc_data_az),max(sc_data_az),stats_az[0], '[deg, deg, deg/s]')
                print('\t\t\t - Elevation [min, max, velo]: ',min(sc_data_el),max(sc_data_el),stats_el[0], '[deg, deg, deg/s]')
                print('\t\t\t - RA [min, max, velo]:        ',min(sc_data_ra),max(sc_data_ra),stats_ra[0], '[deg, deg, deg/s]')
                print('\t\t\t - DEC [min, max, velo]:       ',min(sc_data_dec),max(sc_data_dec),stats_dec[0], '[deg, deg, deg/s]')

                planets = ['sun    ','moon   ','jupiter']
                for p in planets:
                    planet_separation = MPGHD.source_plant_separation(sc_data_ra,sc_data_dec,p.replace(' ',''),time_data,obs_pos).flatten()
                    lowest_FOV        = MPGHD.fov_fwhm(freq[0],15,type='fov',outunit='deg')
                    print('\t\t\t - distance to ',p,'       ',min(planet_separation),', FoV ',lowest_FOV,'[deg]')
            

                    
        print('\n\n')
                    
        sys.exit(-1)
    
    # ---------------------------------------------------------------------------------------------
    # Define how many cpu are used 
    # ---------------------------------------------------------------------------------------------

    if donotncpus == False:
        if usencpus < 0:
            ncpus = multiprocessing.cpu_count() - 1
        else:
            ncpus = usencpus
    else:
        ncpus = 1



    # ---------------------------------------------------------------------------------------------
    # Do the spectrum flagging of the data set
    # ---------------------------------------------------------------------------------------------


    # Here we get some default setting for SKAMPI in S-Band
    #
    para_code_input      = RFIL.get_json('RFI_SETTINGS.json')

    # these parameters are used for the heavyflag option
    splitting            = para_code_input['data_handling']['splitting']
    usedbinning          = para_code_input['data_handling']['usedbinning']
    stats_type           = para_code_input['data_handling']['stats_type']
    smooth_bound_kernel  = para_code_input['data_handling']['smooth_bound_kernel']

    # there is a handflag option with the follwong parameter
    # that does a single step of the heavyflag
    smoo_kernel_type           = para_code_input['data_handling']['smooth_by_hand'][0]
    smoo_kernel                = para_code_input['data_handling']['smooth_by_hand'][1]
    smoo_boundary_kernel_type  = para_code_input['data_handling']['smooth_by_hand'][2]
    smoo_boundary_kernel       = para_code_input['data_handling']['smooth_by_hand'][3]
    splititin                  = para_code_input['data_handling']['smooth_by_hand'][4]
    

    # smoothing process of the masking function
    #
    smooth_type           = para_code_input['data_handling']['smooth_type']
    #
    kernel_sizes          = para_code_input['flagprocessing'][flagprocessing]['kernel_sizes']
    kernel_sequence_type  = para_code_input['flagprocessing'][flagprocessing]['kernel_sequence_type']

    # clean up mask of some pattern 
    #
    clean_bins = [[1,0,1],\
                      [1,0,0,1],\
                      [1,0,0,0,1],\
                      [1,0,0,0,0,1],\
                      [1,0,0,0,0,0,1],\
                      [1,0,0,0,1,0,0,0,1]]

    # Set some input settings 
    #
    bound_sigma          = [bound_sigma_input,bound_sigma_input]   # if edges of the spectrum to much eaten away increase # old setting: bound_sigma          = [3,3]
    flag_on              = eval(use_data)                          # only use noise diode off to generate flags ['ND0','ND1'] would do all

    #
    # --------------------------------------------


    
    # time the fg processing
    #
    full_fg_time   = process_time()

    full_new_mask = {}
    for d in timestamp_keys:

             #
             # select data base on input
             #
             if RFIL.str_in_strlist(d,use_data_fg) and RFIL.str_in_strlist(d,plot_type) and RFIL.str_in_strlist(d,scan_keys):

                time_data       = obsfile[d][:]
                freq            = obsfile[d.replace('timestamp','frequency')][:][1:] # exclude the DC term
                #
                spectrum_data   = obsfile[d.replace('timestamp','')+'spectrum']

                new_mask        = np.zeros(spectrum_data.shape).astype(bool)

                if toutput:
                    print('\n\tgenerate mask for : ',d.replace('timestamp',''),'\n')


                # ---------------------------------------------------------------------------------------------
                # Mask the first and last channel 
                # ---------------------------------------------------------------------------------------------

                new_mask[:,0]    = True                    # exclude the DC term of the FFT spectrum in the full spectrum
                new_mask[:,-1]   = True                    # exclude the DC term of the FFT spectrum in the full spectrum


                # ---------------------------------------------------------------------------------------------
                # spectrum flagging individual times (takes care of ampl) DO NOT USE THE NOISE DIODE DATA HERE
                #
                # (flagged times will not be used in the spectrum flagging and is present in the final mask)
                # ---------------------------------------------------------------------------------------------
                
                if time_sigma != 0:

                    if RFIL.str_in_strlist(d,flag_on):

                        time_sigma_mask_data      = ma.masked_array(spectrum_data,mask=new_mask,fill_value=np.nan)
                        
                        median_over_freq_per_time = np.median(time_sigma_mask_data,axis=1)
                        time_mask                 = RFIL.boundary_mask_data(median_over_freq_per_time.compressed(),median_over_freq_per_time,sigma=time_sigma,stats_type='madmean',do_info=False).astype(int)

                        fg_time = 0 
                        for i in range(len(time_mask)):
                            if time_mask[i] == 1:
                                new_mask[i,:] = True
                                fg_time +=1
                                
                        if toutput:
                            print('\t- FG in time')
                            print('\t\t flagged: ',fg_time,'time steps')
                             
                # ---------------------------------------------------------------------------------------------
                # spectrum flagging by input 
                #
                # (flagged times will not be used in the spectrum flagging and is present in the final mask)
                # ---------------------------------------------------------------------------------------------

                if len(dofgbyhand) > 0:

                    az = obsfile[d.replace('timestamp','azimuth')][:]
                    el = obsfile[d.replace('timestamp','elevation')][:]

                    for i in range(len(dofgbyhand)):
                        new_mask[dofgbyhand[i][0]:dofgbyhand[i][1],:] = True
                        
                        if toutput:
                            print('\t- Hand FG in time')
                            print('\t\t idx: ',dofgbyhand[i])
                            # time info
                            time_range  = Time([time_data[dofgbyhand[i][0]],time_data[dofgbyhand[i][1]]],format='unix')
                            print('\t\t timerange: ',time_range.to_value('isot'))
                            print('\t\t azimut range: ',az[dofgbyhand[i][0]],az[dofgbyhand[i][1]])
                            print('\t\t elevation range: ',el[dofgbyhand[i][0]],el[dofgbyhand[i][1]])




                # ---------------------------------------------------------------------------------------------
                # spectrum flagging by saturation information 
                #
                # (flagged times will not be used in the spectrum flagging and is present in the final mask)
                # ---------------------------------------------------------------------------------------------

                if saturation_fg_sigma > 0:

                    satur = obsfile[d.replace('timestamp','saturated_samples')][:]
                    satur = satur.flatten()

                    time_mask_sat = RFIL.boundary_mask_data(satur,satur,sigma=saturation_fg_sigma,stats_type='madmedian',do_info=False).astype(int)

                    fg_sat = 0 
                    for i in range(len(time_mask_sat)):
                            if time_mask_sat[i] == 1:
                                new_mask[i,:] = True
                                fg_sat += 1
                                
                    if toutput:
                            print('\t- Saturation FG in time')
                            print('\t\t flagged: ',fg_sat,'time steps')


                # ---------------------------------------------------------------------------------------------
                # flagging by scan velocity 
                #
                # (flagged times will not be used in the spectrum flagging and is present in the final mask)
                # ---------------------------------------------------------------------------------------------

                if scan_velo_fg_sigma > 0:

                    if rad_dec_scan == False:
                        sc_data_x = obsfile[d.replace('timestamp','azimuth')][:]
                        sc_data_y = obsfile[d.replace('timestamp','elevation')][:]
                    else:
                        sc_data_x = obsfile[d.replace('timestamp','dec')][:]
                        sc_data_y  = obsfile[d.replace('timestamp','ra')][:]

                    scan_velo_vec = np.sqrt(np.abs(np.gradient(sc_data_x)) + np.abs(np.gradient(sc_data_y)))

                    time_mask_s_velo = RFIL.boundary_mask_data(scan_velo_vec,scan_velo_vec,sigma=scan_velo_fg_sigma,stats_type='madmedian',do_info=False).astype(int)

                    fg_velo = 0
                    for i in range(len(time_mask_s_velo)):
                            if time_mask_s_velo[i] == 1:
                                new_mask[i,:] = True
                                fg_velo += 1 
                     
                    if toutput:
                            print('\t- Scan velocity FG in time')
                            print('\t\t flagged: ',fg_velo,'time steps')


                # ---------------------------------------------------------------------------------------------
                # flagging by relation of noise and strengh of the signal 
                #
                # ---------------------------------------------------------------------------------------------
                
                if flag_RFI_std_mean_sigma > 0:

                    flag_RFI_std_mean_stats_type = 'madmedian'
                    
                    RFI_std_mean_mask_data_1       = ma.masked_array(spectrum_data,mask=new_mask,fill_value=np.nan)

                    # flag channels on std outlieres axis=0 is time, axis=1 is channel
                    #
                    std_freq_sigma            = 9
                    std_over_time_per_freq    = np.std(RFI_std_mean_mask_data_1,axis=0)

                    chan_mask                 = RFIL.boundary_mask_data(std_over_time_per_freq.compressed(),std_over_time_per_freq,\
                                                                            sigma=std_freq_sigma,stats_type=flag_RFI_std_mean_stats_type,do_info=False).astype(int)

                    fg_chan = 0
                    for i in range(len(chan_mask)):
                            if chan_mask[i] == 1:
                                new_mask[:,i] = True
                                fg_chan += 1


                    # flag on a linear relation of std and mean
                    #
                    RFI_std_mean_mask_data_2  = ma.masked_array(spectrum_data,mask=new_mask,fill_value=np.nan)

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
                    lf_mask             = RFIL.boundary_mask_data(RFI_std_cor.compressed(),RFI_std_cor,sigma=flag_RFI_std_mean_sigma,\
                                                                      stats_type=flag_RFI_std_mean_stats_type,do_info=False).astype(int)
                    #
                    rfi_std_mean_fg_chan = 0
                    for i in range(len(lf_mask)):
                            if lf_mask[i] == 1:
                                new_mask[:,i] = True
                                rfi_std_mean_fg_chan += 1
                                
                    if toutput:
                            print('\t- RFI-std-mean-stats FG in channels')
                            print('\t\t flagged: ',rfi_std_mean_fg_chan,'channels')
                    

                                            
                # ---------------------------------------------------------------------------------------------
                # flagging based on single smoothed spectrum  
                # ---------------------------------------------------------------------------------------------

                if flag_smooth_sigma > 0:

                    flag_smoo_stats_type = 'madmadmedian'
                    
                    from scipy.interpolate import interp1d
                
                    # get the single averaged spectrum
                    smoo_data_mask_data = ma.masked_array(spectrum_data,mask=new_mask,fill_value=np.nan)
                    #
                    smoo_freq           = freq
                    smoo_chan           = np.arange(len(freq))
                    smoo_amp            = smoo_data_mask_data.mean(axis=0)[1:]

                    # generate a linear interpolated dataset at the flagged channels
                    #                    
                    cl_smoo_freq      = smoo_freq[np.invert(smoo_amp.mask)]
                    cl_smoo_chan      = smoo_chan[np.invert(smoo_amp.mask)]
                    cl_smoo_amp       = smoo_amp.data[np.invert(smoo_amp.mask)]
                    #
                    intepol_func      = interp1d(cl_smoo_freq,cl_smoo_amp,fill_value='extrapolate')
                    smoo_amp_interpol = intepol_func(smoo_freq)

                    # smooth the data and subtract from the original
                    #
                    #smoo_kernel_type = 'hamming'
                    #smoo_kernel      = 31
                    #
                    smoo_amp_interpol_bslcorr   = smoo_amp_interpol - RFIL.convolve_1d_data(smoo_amp_interpol,smooth_type=smoo_kernel_type,smooth_kernel=smoo_kernel)

                    
                    # Split the residual in spwd to obtain an envelope of the data distribution
                    # CAUTION splitting only works if you splitt the spectrum 
                    # into an integer number
                    #
                    #splititin                 = 12
                    #smoo_boundary_kernel_type = 'median'
                    #smoo_boundary_kernel      = int(2**(splititin/2.) + 1)
                    #
                    spwd_data                 = np.array_split(smoo_amp_interpol_bslcorr,2**splititin)
                    spwd_data_freq            = np.array_split(smoo_freq,2**splititin)
                    up_bound                  = flag_smooth_sigma * np.std(spwd_data,axis=1)
                    #

                    # Smooth the boundary
                    # 
                    smo_boundary            = RFIL.convolve_1d_data(up_bound,smooth_type=smoo_boundary_kernel_type,smooth_kernel=smoo_boundary_kernel)
                    #
                    # need to interpolate full channel range
                    #
                    intepol_bound_func      = interp1d(np.mean(spwd_data_freq,axis=1),smo_boundary,fill_value='extrapolate')
                    bound_up_interpoled     = intepol_bound_func(smoo_freq)
                    bound_low_interpoled    = -1 * bound_up_interpoled
                    #
                    select_up               = smoo_amp_interpol_bslcorr > bound_up_interpoled
                    select_low              = smoo_amp_interpol_bslcorr < bound_low_interpoled
                    #
                    smoo_select_unclead     = np.logical_or(select_up,select_low).astype(int)
                    smoo_select             = RFIL.clean_up_1d_mask(smoo_select_unclead,clean_bins,setvalue=1)

                    # Just for debugging in the future if something odd is happening
                    #
                    dotestplot = False
                    if dotestplot:
                        from matplotlib import pylab as plt
                        plt.plot(smoo_freq,bound_up_interpoled,color='red')
                        plt.plot(smoo_freq,bound_low_interpoled,color='red')
                        plt.scatter(smoo_freq,smoo_amp_interpol_bslcorr)
                        plt.show()
                    
                    # here do the flagging
                    #
                    smoo_fg_chan = 0
                    for i in range(len(smoo_select)):
                            if smoo_select[i] == 1:
                                new_mask[:,i] = True
                                smoo_fg_chan += 1

                    if toutput:
                        print('\t- FG channels on smooth spectrum')
                        print('\t\t kernel: ',smoo_kernel_type,' size ',smoo_kernel)
                        print('\t\t boundary averaging: ',smoo_boundary_kernel_type,' SWPD ',len(spwd_data_freq))
                        print('\t\t\t boundary smoothing kernel: ',smoo_boundary_kernel_type,' size ',smoo_boundary_kernel)
                        print('\t\t flagged: ',smoo_fg_chan,'channels ')

                                
                # ---------------------------------------------------------------------------------------------
                # flagging based on single spectrum baseline fit 
                # ---------------------------------------------------------------------------------------------

                
                if flag_bslf_sigma > 0:

                    flag_bslf_stats_type = 'madmadmedian'
                    
                    from scipy.interpolate import interp1d
                
                    # get the single averaged spectrum
                    bslf_data_mask_data = ma.masked_array(spectrum_data,mask=new_mask,fill_value=np.nan)
                    #
                    bslf_freq         = freq
                    bslf_amp          = bslf_data_mask_data.mean(axis=0)[1:]


                    # generate a linear interpolated dataset at the flagged channels
                    #                    
                    cl_bslf_freq      = bslf_freq[np.invert(bslf_amp.mask)]
                    cl_bslf_amp       = bslf_amp.data[np.invert(bslf_amp.mask)]
                    #
                    intepol_func      = interp1d(cl_bslf_freq,cl_bslf_amp,fill_value='extrapolate')
                    bslf_amp_interpol = intepol_func(bslf_freq)

                    # use pybasline and different methods to do a baseline fit
                    #
                    # https://pybaselines.readthedocs.io/en/v0.8.0/api/pybaselines/whittaker/index.html
                    #
                    regular_modpoly_0 = pbsl.whittaker.airpls(bslf_amp_interpol)[0]
                    regular_modpoly_1 = pbsl.whittaker.arpls(bslf_amp_interpol)[0]
                    regular_modpoly_2 = pbsl.whittaker.aspls(bslf_amp_interpol)[0]
                    regular_modpoly_3 = pbsl.whittaker.derpsalsa(bslf_amp_interpol)[0]   # this is really good
                    regular_modpoly_4 = pbsl.whittaker.drpls(bslf_amp_interpol)[0]
                    regular_modpoly_5 = pbsl.whittaker.iarpls(bslf_amp_interpol)[0]
                    regular_modpoly_6 = pbsl.whittaker.iasls(bslf_amp_interpol)[0]
                    regular_modpoly_7 = pbsl.whittaker.psalsa(bslf_amp_interpol)[0]
                    #
                    interpol_data_info = ['whittaker.airpls','whittaker.arpls','whittaker.aspls','whittaker.derpsalsa','whittaker.drpls',
                                              'whittaker.iarpls','whittaker.iasls','whittaker.psalsa']
                        
                    interpol_data      = np.array([regular_modpoly_0,regular_modpoly_1,regular_modpoly_2,regular_modpoly_3,regular_modpoly_4,\
                                                       regular_modpoly_5,regular_modpoly_6,regular_modpoly_7])


                    bsl_stats = []
                    for bsli in range(interpol_data.shape[0]):
                        bsl_stats_data_mean, bsl_stats_data_std, bsl_stats_stats_type = RFIL.data_stats(bslf_amp_interpol-interpol_data[bsli],stats_type=flag_bslf_stats_type,accur=100)
                        bsl_stats.append(bsl_stats_data_std/bsl_stats_data_mean)
                        
                    
                    std_mean_ratio_min = np.argmin(bsl_stats)
                    std_mean_ratio_max = np.argmax(bsl_stats)

                    docaution = False
                    if std_mean_ratio_min == std_mean_ratio_min:
                        std_mean_ratio_min = 3
                        std_mean_ratio_max = 1
                        docaution = True

                    # use the baseline with the minium std_mean ratio
                    #
                    bslf_amp_interpol_bslcorr   = bslf_amp_interpol - interpol_data[std_mean_ratio_min]
                    bslf_mask                   = RFIL.boundary_mask_data(bslf_amp_interpol_bslcorr,bslf_amp_interpol_bslcorr,sigma=flag_bslf_sigma,stats_type='madmedian',do_info=False).astype(int)


                    # Check if the min and max functions do differ substancially
                    #
                    checkbsl = interpol_data[std_mean_ratio_min] - interpol_data[std_mean_ratio_max]

                    
                    sel_bsl = checkbsl >= 0
                    if np.count_nonzero(sel_bsl) > 0:
                        set_lower_limit_to_fg = (np.arange(len(freq))[sel_bsl])[0]
                    else:
                        set_lower_limit_to_fg = 0

                    # here do the flagging
                    #
                    bslf_fg_chan = 0
                    for i in range(len(bslf_mask)):
                            if bslf_mask[i] == 1 and i >= set_lower_limit_to_fg:
                                new_mask[:,i] = True
                                bslf_fg_chan += 1
                                
                    if toutput:
                        if docaution:
                            print('Caution bslfit has been set!')
                            print('bslstats not determination ', bsl_stats)
                            print(interpol_data_info)
                        print('\t- FG channels on baseline fit ',interpol_data_info[std_mean_ratio_min])
                        print('\t\t flagged: ',bslf_fg_chan,'channels above ',freq[set_lower_limit_to_fg]/1E9,' [GHz]')

                # ---------------------------------------------------------------------------------------------
                # flag individual each spectrum by applying denoising with a lot of smoothing
                # ---------------------------------------------------------------------------------------------

                if donotflag:

                    if RFIL.str_in_strlist(d,flag_on):

                        if toutput:
                            print('\tgenerate flags for : ',d.replace('timestamp',''),'\n')


                        if ncpus > 1:
                                # Setting runs on mutiple cpu 
                                #
                                t_steps = spectrum_data.shape[0]
                                idx     = 0
                                step    = int(t_steps / ncpus) + 1
                                #
                                for i in range(step):
                                    idxq       = 0
                                    mmque      = []
                                    jobs       = []
                                    result_dic = {}
                                    for cps in range(ncpus):
                                        if idx < t_steps:
                                            mmque.append(multiprocessing.Queue())
                                            if toutput:
                                                print('Fan out jobs use ',ncpus,' CPU: ',idx,' ',d.replace('timestamp',''))

                                            fg_spec            = spectrum_data[idx,1:]      # exclude the DC term for the FG estimates

                                            # here check if time has been flagged
                                            check_time_fg = np.sum(new_mask[idx].astype(int))
                                            if check_time_fg != new_mask.shape[1]:                                        
                                                cleanup_spec_mask  = np.zeros(len(fg_spec)).astype(bool)
                                            else:
                                                cleanup_spec_mask  = np.ones(len(fg_spec)).astype(bool)

                                            # Here do the multiprocessing
                                            jo = multiprocessing.Process(target=RFIL.flag_spec_by_smoothing, args=(fg_spec,freq,cleanup_spec_mask,splitting,kernel_sizes,kernel_sequence_type,\
                                                                                     smooth_type,usedbinning,bound_sigma,stats_type,smooth_bound_kernel,clean_bins,idx,mmque[idxq],idxq))
                                            jobs.append(jo)
                                            jo.start()
                                            idxq += 1
                                        #
                                        idx += 1

                                    # update the directory
                                    for k in range(len(jobs)):
                                            result_dic.update(mmque[k].get())

                                    # wait until all the jobs are done
                                    for j in jobs:
                                            jo.join()

                                    # get the results into a final mask
                                    rs = result_dic.keys()
                                    for k in result_dic:
                                            new_mask[result_dic[k][0]][1:] = result_dic[k][1]

                        else:

                            # go through all the time stamps
                            #
                            for s in range(spectrum_data.shape[0]):

                                fg_spec            = spectrum_data[s,1:] # exclude the DC term for the FG estimates

                                # check if time has been flagged
                                #
                                check_time_fg = np.sum(new_mask[s].astype(int))
                                if check_time_fg != new_mask.shape[1]:                                        
                                    cleanup_spec_mask  = np.zeros(len(fg_spec)).astype(bool)
                                else:
                                    cleanup_spec_mask  = np.ones(len(fg_spec)).astype(bool)

                                # check timeing of process
                                fg_t               = process_time()

                                final_sp_mask      = RFIL.flag_spec_by_smoothing(fg_spec,freq,cleanup_spec_mask,splitting,kernel_sizes,kernel_sequence_type,\
                                                                                     smooth_type,usedbinning,bound_sigma,stats_type,\
                                                                                     smooth_bound_kernel,clean_bins)
                                new_mask[s][1:]    = final_sp_mask

                                if toutput:
                                    if s%10 == 0:
                                        # do some time measures
                                        elapsed_time = process_time() - fg_t
                                        print(int(100*s/spectrum_data.shape[0]),'% done, ',' time used ',elapsed_time,' ',d.replace('timestamp',''))


                full_new_mask[d.replace('timestamp','')]      = new_mask

    # determine the full fg time required 
    full_fg_elapsed_time = process_time() - full_fg_time
    if toutput:
        print(' Full FG time needed ',full_fg_elapsed_time,' ')

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
        final_maskcomb = copy.copy(new_mask)

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
            print('\n   === Generates 1d Spectrum plots === \n')

        # Store the final spectra to be optional saved
        #
        plt_final_spectra_data = {}

        # Here does the plotting of the data
        #
        import matplotlib.pyplot as plt
        import matplotlib
        import matplotlib.ticker as tck
        from matplotlib.offsetbox import AnchoredText
        #
        for d in timestamp_keys:

            if RFIL.str_in_strlist(d,plot_type) and RFIL.str_in_strlist(d,use_data_fg) and RFIL.str_in_strlist(d,scan_keys):

                spectrum_data  = obsfile[d.replace('timestamp','')+'spectrum'] 

                freq           = obsfile[d.replace('timestamp','frequency')][:]

                if doplot_with_invert_mask:
                    f_mask         = np.invert(final_mask[d.replace('timestamp','')])
                else:
                    f_mask         = final_mask[d.replace('timestamp','')]

                fg_info = int(100*np.sum(f_mask.astype(int))/np.prod(f_mask.shape))
                
                if toutput:
                    print('\tmask:               ',d.replace('timestamp',''),' ',fg_info,' %')
                    print('\tgenerate plot for : ',d.replace('timestamp',''))

                fullmask_data  = ma.masked_array(spectrum_data,mask=f_mask,fill_value=np.nan)
                spectrum_mean  = fullmask_data.mean(axis=0)
                spectrum_std   = fullmask_data.std(axis=0)


                # safe the spectra in dic
                #
                if len(savefinalspectrum) > 0:
                    plt_final_spectra_data[d.replace('timestamp','')] = {}
                    plt_final_spectra_data[d.replace('timestamp','')]['spectrum_mean'] = spectrum_mean
                    plt_final_spectra_data[d.replace('timestamp','')]['spectrum_std']  = spectrum_std
                    plt_final_spectra_data[d.replace('timestamp','')]['freq']          = freq
                    plt_final_spectra_data[d.replace('timestamp','')]['obs_id']        = obs_id
                    plt_final_spectra_data[d.replace('timestamp','')]['time_data']     = time_data


                plt_info_mean      = spectrum_mean.mean()    
                plt_info_std       = spectrum_mean.std() 

                # print the spectrum
                fig, ax = plt.subplots()
                plt.title('obsid: '+str(obs_id)+' '+d.replace('timestamp',''))
                ax.errorbar(freq,spectrum_mean,yerr=spectrum_std,marker='.',ecolor = 'r',alpha=0.3)
                ax.set_xlabel('frequency [Hz]')
                ax.set_ylabel('mean of data [Jy]')
                ax.xaxis.set_minor_locator(tck.AutoMinorLocator())

                anchored_text = AnchoredText('mean,std '+str('%3.2e'%plt_info_mean)+', '+str('%3.2e'%plt_info_std), loc=1)
                ax.add_artist(anchored_text)


                plt_fspec_yrange = eval(fspec_yrange)
                if max(plt_fspec_yrange) != 0 or min(plt_fspec_yrange) != 0:
                    ax.set_ylim(*plt_fspec_yrange)

                if pltsave:
                    plt_fname = data_file.replace('.hdf5','').replace('.HDF5','')+'_'+d.replace('timestamp','').replace('/','_')+'SPEC'
                    plt_fname = filenamecounter(plt_fname,extention='.png')
                    fig.savefig(plt_fname,dpi=DPI)
                else:
                    plt.show()

                plt.close()


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
            print('\n   === Generates waterfall  plots === \n')

        # Here does the plotting of the data
        #
        import matplotlib.pyplot as plt
        import matplotlib
        #
        for d in timestamp_keys:
            
            if RFIL.str_in_strlist(d,plot_type) and RFIL.str_in_strlist(d,use_data_fg) and RFIL.str_in_strlist(d,scan_keys):

                spectrum_data  = obsfile[d.replace('timestamp','')+'spectrum'][:]

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
                fullmask_data           = ma.masked_array(spectrum_data,mask=f_mask,fill_value=np.nan)


                fig, ax = plt.subplots()
                plt.title('obsid: '+str(obs_id)+' '+d.replace('timestamp',''))
                wfplt = ax.imshow(fullmask_data,interpolation='nearest',origin='lower',cmap=cmap,norm=mpl.colors.LogNorm(),aspect='auto')

                ax.set_xlabel('channels')
                ax.set_ylabel('time')

                if pltsave:
                    plt_fname = data_file.replace('.hdf5','').replace('.HDF5','')+'_'+d.replace('timestamp','').replace('/','_')+'WFPLT'
                    plt_fname = filenamecounter(plt_fname,extention='.png')
                    fig.savefig(plt_fname,dpi=DPI)
                else:
                    plt.show()

                plt.close()

    #
    #
    # ---------------------------------------------------------------------------------------------




    # ---------------------------------------------------------------------------------------------
    # Do RFI report 
    # ---------------------------------------------------------------------------------------------
    #
    if do_rfi_report > 0 :

        # define a range to plot (can be change in the input)
        #
        plt_report_sigma = do_rfi_report_sigma

        if toutput:
            print('\n   === Generates report Information and 1d Spectrum plots === \n')

        # Here does the plotting of the data
        #
        import matplotlib.pyplot as plt
        import matplotlib
        import matplotlib.ticker as tck
        from matplotlib.offsetbox import AnchoredText
        #
        for d in timestamp_keys:

            if RFIL.str_in_strlist(d,plot_type):

                
                print('\t = Observation information')
                print('\t\t DC-term excluded')
                
                print('\t\t - Obs id: ',obs_id)

                # Some general info 
                time_range  = Time([time_data[0],time_data[-1]],format='unix')
                az          = obsfile[d.replace('timestamp','azimuth')][:]
                el          = obsfile[d.replace('timestamp','elevation')][:]

                print('\t\t - timerange: ',time_range.to_value('isot'))
                print('\t\t - azimut range: ',min(az),max(az))
                print('\t\t - elevation range: ',min(el),max(el))


                spectrum_data  = obsfile[d.replace('timestamp','')+'spectrum'][:,1:]
                freq           = obsfile[d.replace('timestamp','frequency')][1:]

                if doplot_with_invert_mask:
                    f_mask         = np.invert(final_mask[d.replace('timestamp','')])[:,1:]
                else:
                    f_mask         = final_mask[d.replace('timestamp','')][:,1:]


                fg_info = int(100*np.sum(f_mask.astype(int))/np.prod(f_mask.shape))
                
                if toutput:
                    print('\tmask:               ',d.replace('timestamp',''),' ',fg_info,' %')
                    print('\tgenerate report for : ',d.replace('timestamp',''))


                if (len(freq)%do_rfi_report) != 0:
                    print('\nCAUTION \n\nThe setting produced sub-arrays with no equal sizes.')
                    print('Subrray size is and used binning parameter ',len(freq),do_rfi_report)
                    print('Change usedbinning parameter in main programme\n\n')
                    sys.exit(-1)


                # split the data and the mask into sub-sections
                #
                sp_spectrum_data   = np.array_split(spectrum_data,do_rfi_report,axis=1)
                sp_freq            = np.array_split(freq,do_rfi_report)
                sp_f_mask          = np.array_split(f_mask,do_rfi_report,axis=1)



                for sp in range(len(sp_freq)):


                    fullmask_data  = ma.masked_array(sp_spectrum_data[sp],mask=sp_f_mask[sp],fill_value=np.nan)

                    spectrum_mean  = fullmask_data.mean(axis=0)
                    spectrum_std   = fullmask_data.std(axis=0)

                    plt_mean       = spectrum_mean.mean()    
                    plt_std        = spectrum_mean.std()   

                     
                    # print the spectrum
                    fig, ax = plt.subplots()
                    plt.title('obsid: '+str(obs_id)+' SPWD['+str(sp)+'] '+d.replace('timestamp',''))
                    ax.errorbar(sp_freq[sp],spectrum_mean,yerr=spectrum_std,marker='.',ecolor = 'r',alpha=0.3)
                    ax.set_xlabel('frequency [Hz]')
                    ax.set_ylabel('mean of data [Jy]')
                    ax.xaxis.set_minor_locator(tck.AutoMinorLocator())


                    anchored_text = AnchoredText('mean,std '+str('%3.2e'%plt_mean)+', '+str('%3.2e'%plt_std), loc=1)
                    ax.add_artist(anchored_text)

                    plt_fspec_yrange = eval(fspec_yrange)
                    if max(plt_fspec_yrange) != 0 or min(plt_fspec_yrange) != 0:
                        ax.set_ylim(*plt_fspec_yrange)
                    else:
                        ax.set_ylim([plt_mean-plt_report_sigma*plt_std,plt_mean+plt_report_sigma*plt_std])


                    if pltsave:
                        plt_fname = data_file.replace('.hdf5','').replace('.HDF5','')+'_'+d.replace('timestamp','').replace('/','_')+'SPEC'+'_SPWD'+str(sp)
                        plt_fname = filenamecounter(plt_fname,extention='.png')
                        fig.savefig(plt_fname,dpi=DPI)
                    else:
                        plt.show()

                    plt.close()


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

    parser.add_option('--USEDATA', dest='usedata', type=str,default="['P0','P1']",
                      help='use data to flag default is "[\'P0\',\'P1\']", for Stokes use e.g. \"[\'S0\']\"')

    parser.add_option('--USENOISEDATA', dest='usenoisedata', type=str,default="['ND0']",
                      help='use data noise diode on and off "[\'ND0\',\'ND1\']", default is \"[\'ND0\']\"')

    parser.add_option('--USESCAN', dest='usescan', type=str,default="[]",
                      help='select scan to flag, default are all scans, to choose scan 000 and 001 use e.g. \"[\'000\',\'001\']\"')
    
    parser.add_option('--DONOTHEAVYFLAG', dest='donotflag', action='store_false',
                      default=True,help='Do not use the time heavy flag procedure.')
   
    parser.add_option('--PROCESSING_TYPE', dest='flagprocessing', type=str,default='SEMIFAST',
                      help='setting how accurate/much time the flagging proceed. FAST, SEMIFAST, SLOW, default is SEMIFAST')

    parser.add_option('--DO_FG_TIME_BY_HAND', dest='hand_time_fg', type=str,default='[]',
                      help='use the time index of the waterfall plot e.g. [[0,10],[100,110]]')

    parser.add_option('--DO_FG_TIME_AUTO_SIGMA', dest='auto_time_fg_sigma', type=float,default=0,
                      help='automatically determine bad time use threshold. default = 0 is off use e.g. = 5')

    parser.add_option('--DO_FG_SATURATION_SIGMA', dest='saturation_fg_sigma', type=float,default=0,
                      help='use the saturation information to flag. default = 0 is off use e.g. = 3')

    parser.add_option('--DO_FG_BOUNDARY_SIGMA', dest='bound_sigma_input', type=float, default=3,
                      help='if the spectrum is mask to much at the edges increase. [default = 3 sigma]')

    parser.add_option('--DO_FG_VELO_SCAN_SIGMA', dest='scan_velo_fg_sigma', type=float, default=0,
                      help='determine flags based on the scan velocity outliere on sky. [default = 0 is off use e.g. = 6]')

    parser.add_option('--DO_RFI_STD_MEAN_SIGMA', dest='flag_RFI_std_mean_sigma', type=float, default=0,
                      help='determine flags based on the linear relation of noise and power. [default = 0 is off use e.g. = 3]')

    parser.add_option('--DO_SMOOTH_SIGMA', dest='flag_smooth_sigma', type=float, default=0,
                      help='determine flags based on smooth spectrum, see RFI_SETTINGS.json. [default = 0 is off use e.g. = 3]')
    
    parser.add_option('--DO_BSLF_SIGMA', dest='flag_bslf_sigma', type=float, default=0,
                      help='determine flags based on spectral baseline fit. [default = 0 is off use e.g. = 10]')

    parser.add_option('--DO_AZEL_SCAN', dest='rad_dec_scan', action='store_false',
                      default=True,help='Switch to Azimuth-Elevation scan type to be used for velocity outliere flag.')

    parser.add_option('--DOPLOT_FINAL_SPEC', dest='doplot_final_spec', action='store_true',
                      default=False,help='Plot the final spectrum after Flagging')

    parser.add_option('--FINAL_SPEC_YRANGE', dest='fspec_yrange', type=str,default='[0,0]',
                      help='[ymin,ymax]')

    parser.add_option('--DOPLOT_FINAL_WATERFALL', dest='doplot_final_full_data', action='store_true',
                      default=False,help='Plot the final waterfall after Flagging')

    parser.add_option('--DOPLOT_WITH_INVERTED_MASK', dest='invert_mask', action='store_true',
                      default=False,help='Plot the final plots using an inverted mask')

    parser.add_option('--DOSAVEPLOT', dest='pltsave', action='store_true',
                      default=False,help='Save the plots as figures')

    parser.add_option('--EDIT_FLAG', dest='change_flag', action='store_true',
                      default=False,help='Switch to replace the old with the new mask')

    parser.add_option('--RESET_FLAG', dest='reset_flag', action='store_true',
                      default=False,help='Switch to clear all mask')

    parser.add_option('--DOSAVEMASK', dest='savemask',type=str,default='',
                      help='Save the mask into numpy npz file.')

    parser.add_option('--DOSAVEFINALSPECTRUM', dest='savefinalspectrum', type=str,default='',
                      help='Safe the final 1d spectra as numpy npz file. [works only with --DOPLOT_FINAL_SPEC]')

    parser.add_option('--DOLOADMASK', dest='loadmask', type=str,default='',
                      help='Upload the mask.')

    parser.add_option('--DONOTCPUS', dest='donotncpus', action='store_true',
                      default=False,help='Switch off using multiple CPUs on the maschine')

    parser.add_option('--USENCPUS', dest='usencpus', type=int,
                      default=-1,help='Define the number of CPUs to use')

    parser.add_option('--SILENCE', dest='toutput', action='store_false',
                      default=True,help='Switch off all output')

    parser.add_option('--DO_RFI_REPORT', dest='do_rfi_report', type=int, default=-1,
                      help='provides info and SPWD plots. Input is number of SPWD [default = -1, use e.g. 8]')

    parser.add_option('--DO_RFI_REPORT_SIGMA', dest='do_rfi_report_sigma', type=float, default=5,
                      help='set the y-range of the SPWDs plots of the RFI report [default = 5]')

    parser.add_option('--OBSINFO', dest='getobsinfo', action='store_true',
                      default=False,help='Show observation info stored in the file')

    parser.add_option('--HELP', dest='help', action='store_true',
                      default=False,help='Show info on input')

    return parser



if __name__ == "__main__":
    main()
