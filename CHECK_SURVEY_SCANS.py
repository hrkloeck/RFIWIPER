#
#
# Hans-Rainer Kloeckner
#
# MPIfR 2024
#
#
# This is an example to get you a head start using a HDF5 files
# of the SKAMPI telescope and get Ferdinands data flagged 
#
#
import h5py
import numpy as np
import numpy.ma as ma
import multiprocessing
import sys
from optparse import OptionParser
#
from MPG_HDF5_libs import *
#
from astropy.time import Time
from astropy import units as u
#
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import process_time
import RFI_LIB_SCANS as RFIL
# import heat from submodule
import sys
sys.path.append("heat/")
import heat as ht

# use graphviz to plot the flowchart
#import graphviz

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
    donotflag                 = opts.donotflag
    flagprocessing            = opts.flagprocessing
    dofgbyhand                = opts.hand_time_fg
    time_sigma                = opts.auto_time_fg_sigma
    saturation_fg_sigma       = opts.saturation_fg_sigma
    bound_sigma_input         = opts.bound_sigma_input
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
    # use Heat as backend
    heat_backend              = opts.heat_backend

    do_rfi_report             = opts.do_rfi_report
    do_rfi_report_sigma       = opts.do_rfi_report_sigma

    # define hat to plot
    #
    plot_type                 = eval(use_data)
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
    timestamp_keys       = findkeys(obsfile,keys=['scan','timestamp'],exactmatch=True)
    #
    spectrum_keys        = findkeys(obsfile,keys=['scan','spectrum'],exactmatch=True)
    # #########  
    #


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


    # Some hardcoded input
    #
    splitting            = [0,6000,-1]                        # split the spectrum into two sections (usefull for SKAMPI)
    usedbinning          = [100,61]                           # carefull these setting cost time  need to check in RFI_lib is np splitt can generate 
    stats_type           = ['madmean','madmean']              # define the type of statistic estimates used
    smooth_bound_kernel  = [31,31]                            # smooth kernel for overall boundary range
    
    # smoothing process of the masking function
    #
    smooth_type          = ['wiener','wiener']
    #
    # kernel_sizes and kernel_sequence_type for this is defined below


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

    if flagprocessing == 'SEMIFAST':
        kernel_sizes         = [7,30]                              # carefull these setting cost time 
        kernel_sequence_type = ['middle_fast','middle_fast']       # optional middle_fast, fast, slow
    elif flagprocessing == 'FAST':
        kernel_sizes         = [7,13]                              # carefull these setting cost time  
        kernel_sequence_type = ['fast','fast']                     # optional middle_fast, fast, slow
    else:
        kernel_sizes         = [100,500]                           # carefull these setting cost time  
        kernel_sequence_type = ['slow','slow']                     # optional middle_fast, fast, slow


    # --------------------------------------------

    # time the fg processing
    #
    full_fg_time   = process_time()

    full_new_mask = {}
    for d in timestamp_keys:


            time_data       = obsfile[d][:]
            freq            = obsfile[d.replace('timestamp','frequency')][:][1:] # exclude the DC term
            #
            spectrum_data   = obsfile[d.replace('timestamp','')+'spectrum']

            new_mask        = np.zeros(spectrum_data.shape).astype(bool)

            if toutput:
                print('\tgenerate mask for : ',d.replace('timestamp',''),'\n')


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

                    median_over_freq_per_time = np.median(spectrum_data,axis=1)
                    time_mask                 = RFIL.boundary_mask_data(median_over_freq_per_time,median_over_freq_per_time,sigma=time_sigma,stats_type='madmean',do_info=False).astype(int)

                    for i in range(len(time_mask)):
                        if time_mask[i] == 1:
                            new_mask[i,:] = True

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

                for i in range(len(time_mask_sat)):
                        if time_mask_sat[i] == 1:
                            new_mask[i,:] = True


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
                        if heat_backend:
                            print('Using Heat backend')
                            # Heat backend supports only 'hamming' for now
                            # TODO: implement support for other window functions
                            smooth_type = ['hamming','hamming']                 
                            new_mask = ht.array(new_mask, split=0)
                            fg_spectra = ht.array(spectrum_data[:,1:], split=0) # exclude the DC term for the FG estimates
                            # check what time has been flagged
                            check_time_fg = ht.sum(new_mask.astype(ht.int), axis=1)
                            cleanup_spectra_mask = ht.ones(fg_spectra.shape, split=0).astype(ht.bool)
                            time_is_flagged = check_time_fg != new_mask.shape[1]
                            cleanup_spectra_mask[time_is_flagged] = False

                            # time the flagging
                            print('Heat backend: starting flagging')
                            fg_t = process_time()
                            final_mask = RFIL.flag_spec_by_smoothing_ht(fg_spectra, freq, cleanup_spectra_mask, splitting, kernel_sizes, kernel_sequence_type, smooth_type, usedbinning, bound_sigma, stats_type, smooth_bound_kernel, clean_bins)
                            new_mask = final_mask # TODO: this is probably unnecessary  
                            if toutput:
                                elapsed_time = process_time() - fg_t
                                print('Heat backend: time uses ', elapsed_time, ' ', d.replace('timestamp', ''))
                        else:
                            # time entire procedure
                            fg_t               = process_time()
                            for s in range(spectrum_data.shape[0]):
                                if toutput and s % 100 == 0:
                                    print(f"{s}/{spectrum_data.shape[0]}")
                                fg_spec            = spectrum_data[s,1:] # exclude the DC term for the FG estimates
                                
                                # check if time has been flagged
                                #
                                check_time_fg = np.sum(new_mask[s].astype(int))
                                if check_time_fg != new_mask.shape[1]:                                        
                                    cleanup_spec_mask  = np.zeros(len(fg_spec)).astype(bool)
                                else:
                                    cleanup_spec_mask  = np.ones(len(fg_spec)).astype(bool)

                                # check timeing of process
                                #fg_t               = process_time()

                                final_sp_mask      = RFIL.flag_spec_by_smoothing(fg_spec,freq,cleanup_spec_mask,splitting,kernel_sizes,kernel_sequence_type,\
                                                                                    smooth_type,usedbinning,bound_sigma,stats_type,\
                                                                                    smooth_bound_kernel,clean_bins)
                                new_mask[s][1:]    = final_sp_mask

                            if toutput:
                                # do some time measures
                                elapsed_time = process_time() - fg_t
                                print(s,' time uses ',elapsed_time,' ',d.replace('timestamp',''))


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
        final_mask = {}
        for d in timestamp_keys:
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

            if RFIL.str_in_strlist(d,plot_type):

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

            if RFIL.str_in_strlist(d,plot_type):


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

    parser.add_option('--USEDATA', dest='usedata', type=str,default="['ND0']",
                      help='use data noise diode off and on "[\'ND0\',\'ND1\']", default is [\'ND0\']')

    parser.add_option('--DONOTFLAG', dest='donotflag', action='store_false',
                      default=True,help='Do not flag the data.')
   
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

    parser.add_option('--HELP', dest='help', action='store_true',
                      default=False,help='Show info on input')
    # if this option is passed, `heat_backend` is set to True and the heat package is used as backend for parallelism
    parser.add_option('--HEAT_BACKEND', dest='heat_backend', action='store_true',
                      default=False, help='Use the Heat backend for parallelization and GPU support')
    return parser



if __name__ == "__main__":
    main()
