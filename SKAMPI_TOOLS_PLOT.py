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


def plot_observation(data_x,data_y,data_c,rad_dec_scan,title,pltsave=False,plt_fname=None):
    """
    """

    fig, ax = plt.subplots()
    plt.title(title)

    sc = ax.scatter(data_x,data_y,c=data_c,alpha=0.2,norm='log')
    plt.colorbar(sc)
    
    if rad_dec_scan == False:
            ax.set_xlabel('azimuth [deg]')
            ax.set_ylabel('elevation [deg]')
    else:
            ax.set_xlabel('right ascension  [deg]')
            ax.set_ylabel('declination [deg]')
            
    if pltsave:
            plt_fname = filenamecounter(plt_fname,extention='.png')
            fig.savefig(plt_fname,dpi=DPI)
    else:
            plt.show()

    plt.close()

