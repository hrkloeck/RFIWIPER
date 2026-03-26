#
#
# Hans-Rainer Kloeckner
#
# MPIfR 2025
#
# History:
#
# Part of the RFIWIPER 
#
#
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as tck
from matplotlib.offsetbox import AnchoredText

from MPG_HDF5_libs import filenamecounter

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
DPI = 150
#
# ###################################################


def plot_histogram(data,stats_type='mean',sigma=6,doshow=True):
        """
        plot a histogram
        useful to check the data distribution e.g. of the scanning velocity 
        """
        #
        import matplotlib.pyplot as plt
        #
        # Just as a tool to help debugging
        #
        plt_data_mean, plt_data_std, plt_stats_type = data_stats(data,stats_type)
        #
        #
        cuts1  = plt_data_mean - sigma * plt_data_std
        cuts2  = plt_data_mean + sigma * plt_data_std
        
        print('%e'%cuts1,'--','%e'%cuts2)
        
        fig, ax            = plt.subplots()
        yvalues,bins,patch = ax.hist(data,density=True,bins=accur)
        #
        plt.plot([cuts1,cuts1],[min(yvalues),max(yvalues)],'-,','b')
        plt.plot([cuts2,cuts2],[min(yvalues),max(yvalues)],'-,','b')
        #
        if doshow:
                plt.show()
            

def plot_spectrum(freq,spectrum,spectrum_std,title,fspec_yrange,pltsave=False,plt_fname=None):
    """
    """

    plt_info_mean      = spectrum.mean()    
    plt_info_std       = spectrum.std()
    
    fig, ax = plt.subplots()
    plt.title(title)
    ax.errorbar(freq,spectrum,yerr=spectrum_std,marker='.',ecolor = 'r',alpha=0.3)
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('mean of data [Jy]')
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
        
    anchored_text = AnchoredText('mean,std '+str('%3.2e'%plt_info_mean)+', '+str('%3.2e'%plt_info_std), loc=1)
    ax.add_artist(anchored_text)


    plt_fspec_yrange = eval(fspec_yrange)
    if max(plt_fspec_yrange) != 0 or min(plt_fspec_yrange) != 0:
        ax.set_ylim(*plt_fspec_yrange)

    if pltsave:
        plt_fname = filenamecounter(plt_fname,extention='.png')
        fig.savefig(plt_fname,dpi=DPI)
    else:
        plt.show()
        
    plt.close()

    
def plot_waterfall_spectrum(data,d,title,pltsave=False,plt_fname=None):
    """
    """
    fig, ax = plt.subplots()
    plt.title(title)
    wfplt = ax.imshow(data,interpolation='nearest',origin='lower',cmap=cmap,norm=mpl.colors.LogNorm(),aspect='auto')
    plt.colorbar(wfplt)
    
    ax.set_xlabel('channels')
    ax.set_ylabel('time')
    
    if pltsave:
        plt_fname = filenamecounter(plt_fname,extention='.png')
        fig.savefig(plt_fname,dpi=DPI)
    else:
        plt.show()
        
    plt.close()


def plot_observation_scan(data_x,data_y,data_c,data_m,data_info,rad_dec_scan,title,pltsave=False,plt_fname=None,mask_true_flag=True):
    """
    """

    # For plotting need to sort
    #
    sort_data_c    = np.argsort(np.concatenate(data_c))
    #
    data_x_s       = np.concatenate(data_x)[sort_data_c]
    data_y_s       = np.concatenate(data_y)[sort_data_c]
    data_c_s       = np.concatenate(data_c)[sort_data_c]
    data_m_s       = np.concatenate(data_m)[sort_data_c]
    #
    sel_color      = data_m_s == mask_true_flag
    #
    data_x_s_plt     = data_x_s[sel_color]
    data_y_s_plt     = data_y_s[sel_color]
    data_c_s_plt     = data_c_s[sel_color]


    fig, ax = plt.subplots()
    plt.title(title)

    if np.log(np.max(data_c_s_plt)-np.min(data_c_s_plt)) > 19:
        sc = ax.scatter(data_x_s_plt,data_y_s_plt,c=data_c_s_plt,norm='log',linewidths=0)
    else:
        sc = ax.scatter(data_x_s_plt,data_y_s_plt,c=data_c_s_plt,linewidths=0)

    plt.colorbar(sc)
    
    if rad_dec_scan == False:
            ax.set_xlabel('azimuth [deg]')
            ax.set_ylabel('elevation [deg]')
            x_lab = 'AZ'
            y_lab = 'EL'
    else:
            ax.set_xlabel('right ascension  [deg]')
            ax.set_ylabel('declination [deg]')
            x_lab = 'RA'
            y_lab = 'DEC'

    max_position = '('+x_lab+', '+y_lab+')= ('+str(np.round(data_x_s[np.argmax(data_c_s)],3))+', '+str(np.round(data_y_s[np.argmax(data_c_s)],2))+')'

    plt.text(data_x_s_plt[np.argmax(data_c_s_plt)]*1.0015,data_y_s_plt[np.argmax(data_c_s_plt)]*0.9995,max_position)

    for sa in range(len(data_info)):
        sa_text = 'scan '+str(data_info[sa])
        sel_sc_color = data_m[sa] == mask_true_flag
        plt.text(data_x[sa][sel_sc_color][0],data_y[sa][sel_sc_color][0],sa_text)

    if pltsave:
            plt_fname = filenamecounter(plt_fname,extention='.png')
            fig.savefig(plt_fname,dpi=DPI)
    else:
            plt.show()

    plt.close()


def plot_observation_colouring(data_x,data_y,data_c,data_m,data_t,data_info,data_idx,colouringis,title,pltsave=False,plt_fname=None,mask_true_flag=True):
    """
    """

    multiple_input = len(data_x)

    
    # For plotting need to sort
    #
    sort_data_c    = np.argsort(np.concatenate(data_c))
    #
    data_x_s       = np.concatenate(data_x)[sort_data_c]
    data_y_s       = np.concatenate(data_y)[sort_data_c]
    data_c_s       = np.concatenate(data_c)[sort_data_c]
    data_m_s       = np.concatenate(data_m)[sort_data_c]
    data_t_s       = np.concatenate(data_t)[sort_data_c]
    data_idx_s     = np.concatenate(data_idx)[sort_data_c]

    sel_color      = data_m_s == mask_true_flag

    if multiple_input == 1:
        data_x_plt     = data_idx_s[sel_color]
    else:
        data_x_plt     = data_t_s[sel_color]
                
    data_y_plt     = data_c_s[sel_color]
    data_c_plt     = data_c_s[sel_color]
    data_idx_plt   = data_idx_s[sel_color]

    
    fig, ax = plt.subplots()

    plt.title(title)
    
    if np.log(np.max(data_c_s)-np.min(data_c_s)) > 19:
        sc = ax.scatter(data_x_plt,data_y_plt,c=data_c_plt,norm='log',linewidths=0)
    else:
        sc = ax.scatter(data_x_plt,data_y_plt,c=data_c_plt,linewidths=0)

    plt.colorbar(sc)

    ax.set_xlabel('time [UTC]')
    ax.set_ylabel(colouringis)
    #
    x_lab = 'TIME'
    y_lab = colouringis[:3]
            
    if multiple_input == 1:

        max_position = '('+x_lab+', '+y_lab+')= ('+str('%e'%np.round(data_x_plt[np.argmax(data_c_plt)],3))+', '+str('%e'%np.round(data_y_plt[np.argmax(data_c_plt)],2))+')'
        plt.text(data_x_plt[np.argmax(data_c_plt)],data_y_plt[np.argmax(data_c_plt)],max_position)


        max_colour  = data_c_plt[np.argmax(data_c_plt)]
        min_colour  = data_c_plt[np.argmin(data_c_plt)]
        s_10percent = min_colour + (max_colour-min_colour) * 0.1    
        t_min       = data_x_plt[np.argmin(data_x_plt)]
        t_max       = data_x_plt[np.argmax(data_x_plt)]

        sel_10pc    = data_c_plt > s_10percent

        
        plt.plot([t_min,t_max],[s_10percent,s_10percent])
        plt.text(t_min,s_10percent,'10% of maximum')

        plt.text(min(data_idx_plt[sel_10pc]),s_10percent,str(min(data_idx_plt[sel_10pc])))
        plt.text(max(data_idx_plt[sel_10pc]),s_10percent,str(max(data_idx_plt[sel_10pc])))
        
        ax.set_xlabel('index of the time axis')


    for sa in range(len(data_info)):
        sa_text = 'scan '+str(data_info[sa])
        sel_sc_color = data_m[sa] == mask_true_flag
        plt.text(data_t[sa][sel_sc_color][0],data_c[sa][sel_sc_color][0],sa_text)


    if pltsave:
            plt_fname = filenamecounter(plt_fname,extention='.png')
            fig.savefig(plt_fname,dpi=DPI)
    else:
            plt.show()

    plt.close()
