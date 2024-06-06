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
    dofgbyhand                = opts.hand_time_fg
    time_sigma                = opts.auto_time_fg_sigma
    doplot_final_spec         = opts.doplot_final_spec
    doplot_final_full_data    = opts.doplot_final_full_data
    fspec_yrange              = opts.fspec_yrange
    pltsave                   = opts.pltsave
    reset_flag                = opts.reset_flag
    change_flag               = opts.change_flag
    savemask                  = opts.savemask
    loadmask                  = opts.loadmask
    donotncpus                = opts.donotncpus
    toutput                   = opts.toutput
    usencpus                  = opts.usencpus



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


    splitting            = [0,6000,-1]           # split the spectrum into two sections
    #kernel_sizes         = [100,500]            # setting for continous kernel carefull these setting cost time 
    kernel_sizes         = [7,30]                # carefull these setting cost time 
    smooth_type          = ['wiener','wiener']
    usedbinning          = [1,60]                # carefull these setting cost time 
    bound_sigma          = [3,3]
    stats_type           = ['madmean','madmean'] 
    smooth_bound_kernel  = [31,31]
    flag_on              = eval(use_data)        # only use noise diode off to generate flags ['ND0','ND1'] would do all


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

            new_mask[:,0]   = True                    # exclude the DC term of the FFT spectrum in the full spectrum
            new_mask[:,-1]   = True                    # exclude the DC term of the FFT spectrum in the full spectrum
 

            # ---------------------------------------------------------------------------------------------
            # spectrum flagging individual times (takes care of ampl) DO NOT USE THE NOISE DIODE DATA HERE
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
            # (this will be ignored in the spectrum flagging, but is present in the final mask)
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
                                            print('Fan out jobs use ',ncpus,' CPU: ',idx,' ')
                                        fg_spec            = spectrum_data[idx,1:]      # exclude the DC term for the FG estimates

                                        # here check if time has been flagged
                                        check_time_fg = np.sum(new_mask[idx].astype(int))
                                        if check_time_fg != new_mask.shape[1]:                                        
                                            cleanup_spec_mask  = np.zeros(len(fg_spec)).astype(bool)
                                        else:
                                            cleanup_spec_mask  = np.ones(len(fg_spec)).astype(bool)

                                        # Here do the multiprocessing
                                        jo = multiprocessing.Process(target=RFIL.flag_spec_by_smoothing, args=(fg_spec,freq,cleanup_spec_mask,splitting,kernel_sizes,\
                                                                                 smooth_type,usedbinning,bound_sigma,stats_type,smooth_bound_kernel,idx,mmque[idxq],idxq))
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

                            fg_t               = process_time()
                            final_sp_mask      = RFIL.flag_spec_by_smoothing(fg_spec,freq,cleanup_spec_mask,splitting,kernel_sizes,\
                                                                                 smooth_type,usedbinning,bound_sigma,stats_type,smooth_bound_kernel)
                            new_mask[s][1:]    = final_sp_mask

                            if toutput:
                                #do some stuff
                                elapsed_time = process_time() - fg_t
                                print('time uses',elapsed_time)


            full_new_mask[d.replace('timestamp','')]      = new_mask


    # ---------------------------------------------------------------------------------------------
    # Merge the mask into a single one
    # ---------------------------------------------------------------------------------------------

    keys = full_new_mask.keys()

    final_maskcomb = copy.copy(new_mask)

    for k in keys:
            final_maskcomb = np.logical_or(full_new_mask[k],final_maskcomb)

    final_mask = {}
    for d in timestamp_keys:
            final_mask[d.replace('timestamp','')] = final_maskcomb


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

        # Here does the plotting of the data
        #
        import matplotlib.pyplot as plt
        import matplotlib
        #
        for d in timestamp_keys:

            if RFIL.str_in_strlist(d,plot_type):


                if toutput:
                    print('\tgenerate plot for : ',d.replace('timestamp',''))

                spectrum_data  = obsfile[d.replace('timestamp','')+'spectrum'] 

                freq           = obsfile[d.replace('timestamp','frequency')][:]

                fullmask_data  = ma.masked_array(spectrum_data,mask=final_mask[d.replace('timestamp','')],fill_value=np.nan)
                spectrum_mean  = fullmask_data.mean(axis=0)
                spectrum_std   = fullmask_data.std(axis=0)


                # print the spectrum
                fig, ax = plt.subplots()
                plt.title('obsid: '+str(obs_id)+' '+d.replace('timestamp',''))
                #ax.plot(freq,spectrum_mean)
                ax.errorbar(freq,spectrum_mean,yerr=spectrum_std,marker='.',ecolor = 'r',alpha=0.3)
                ax.set_xlabel('frequency [Hz]')
                ax.set_ylabel('mean of data [Jy]')

                plt_fspec_yrange = eval(fspec_yrange)
                if max(plt_fspec_yrange) != 0 or min(pl_fspec_yrange) != 0:
                    ax.set_ylim(*plt_fspec_yrange)

                if pltsave:
                    plt_fname = data_file.replace('.hdf5','').replace('.HDF5','')+'_'+d.replace('timestamp','').replace('/','_')+'SPEC'
                    plt_fname = filenamecounter(plt_fname,extention='.png')
                    fig.savefig(plt_fname,dpi=DPI)
                else:
                    plt.show()

                plt.close()



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

                if toutput:
                    print('\tgenerate plot for : ',d.replace('timestamp',''))

                spectrum_data  = obsfile[d.replace('timestamp','')+'spectrum'][:]

                #spectrum_data = spectrum_data[0:1000,:]


                # print the waterfall plot
                #
                fullmask_data           = ma.masked_array(spectrum_data,mask=final_mask[d.replace('timestamp','')],fill_value=np.nan)
                #fullmask_data           = ma.masked_array(spectrum_data,mask=final_mask[d.replace('timestamp','')][0:1000,:],fill_value=np.nan)


                fig, ax = plt.subplots()
                plt.title('obsid: '+str(obs_id)+' '+d.replace('timestamp',''))
                #wfplt = ax.imshow(fullmask_data,interpolation='nearest',origin='lower',vmin=stats[0]-3*stats[1],vmax=stats[0]+3*stats[1])
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
   
    parser.add_option('--DO_FG_TIME_BY_HAND_', dest='hand_time_fg', type=str,default='[]',
                      help='use the time index of the waterfall plot e.g. [[0,10],[100,110]]')

    parser.add_option('--DO_FG_TIME_AUTO_SIGMA', dest='auto_time_fg_sigma', type=float,default=0,
                      help='automatically determine bad time use threshold. default = 0 is off use e.g. = 5')

    parser.add_option('--DOPLOT_FINAL_SPEC', dest='doplot_final_spec', action='store_true',
                      default=False,help='Plot the final spectrum after Flagging')

    parser.add_option('--FINAL_SPEC_YRANGE', dest='fspec_yrange', type=str,default='[0,0]',
                      help='[ymin,ymax]')

    parser.add_option('--DOPLOT_FINAL_WATERFALL', dest='doplot_final_full_data', action='store_true',
                      default=False,help='Plot the final waterfall after Flagging')

    parser.add_option('--DOSAVEPLOT', dest='pltsave', action='store_true',
                      default=False,help='Save the plots as figures')

    parser.add_option('--EDIT_FLAG', dest='change_flag', action='store_true',
                      default=False,help='Switch to replace the old with the new mask')

    parser.add_option('--RESET_FLAG', dest='reset_flag', action='store_true',
                      default=False,help='Switch to clear all mask')

    parser.add_option('--DOSAVEMASK', dest='savemask',type=str,default='',
                      help='Save the mask into numpy npz file.')

    parser.add_option('--DOLOADMASK', dest='loadmask', type=str,default='',
                      help='Upload the mask.')

    parser.add_option('--DONOTCPUS', dest='donotncpus', action='store_true',
                      default=False,help='Switch off using multiple CPUs on the maschine')

    parser.add_option('--USENCPUS', dest='usencpus', type=int,
                      default=-1,help='Define the number of CPUs to use')

    parser.add_option('--SILENCE', dest='toutput', action='store_false',
                      default=True,help='Switch off all output')

    parser.add_option('--HELP', dest='help', action='store_true',
                      default=False,help='Show info on input')

    return parser



if __name__ == "__main__":
    main()
