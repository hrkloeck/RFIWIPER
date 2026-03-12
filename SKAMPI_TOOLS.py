#
#
# Hans-Rainer Kloeckner
#
# MPIfR 2026
#
# Provide some handy tools to optain information
# of the data set
#
#

import numpy as np
import numpy.ma as ma
import json

import MPG_HDF5_libs as MPGHD
import RFI_WIPER_TOOLS_STATISTIC as STS
import RFI_WIPER_LIBRARY as RFIL

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u


def observation_info(obsfile,timestamp_keys,spectrum_keys,use_data_fg=[],plot_type=[],scan_keys=[],prt_info=True):
    """
    Provide observation information 
    stored in the SKAMPI HDF file 
    """
    
    info_dics = {}

    # get info from attributes of the file
    for a in obsfile.attrs:
        info_dics[a] = obsfile.attrs[a]

    #scan_keys   = MPGHD.get_obs_info(timestamp_keys,info_idx=1)
    #data_keys   = MPGHD.get_obs_info(MPGHD.get_obs_info(spectrum_keys,info_idx=2),info_idx=0,splittype='_')
    #noise_keys  = MPGHD.get_obs_info(MPGHD.get_obs_info(spectrum_keys,info_idx=2),info_idx=1,splittype='_')

    info_dics['SCAN']  = list(scan_keys)
    info_dics['TYPE']  = list(use_data_fg)
    info_dics['NOISE'] = list(plot_type)

    obs_pos = [info_dics['telescope_longitude'],info_dics['telescope_latitude'],info_dics['telescope_height']]


    print('\n======== DATA INFORMATION =========================\n')

    #print('\t - Data file ', data_file)

    print('\n - General Information ')

    for i in info_dics:
        print('\t',i,info_dics[i])


    for d in timestamp_keys:
        
        if RFIL.str_in_strlist(d,use_data_fg) and RFIL.str_in_strlist(d,plot_type) and RFIL.str_in_strlist(d,scan_keys):

            info_data         = d.split('/')

            time_data         = obsfile[d][:]
            freq              = obsfile[d.replace('timestamp','frequency')][:][1:] # exclude the DC term
            #
            acu_times         = Time(obsfile[d][:],scale='utc',format='unix').iso
            #
            sc_data_az        = obsfile[d.replace('timestamp','azimuth')][:]
            sc_data_el        = obsfile[d.replace('timestamp','elevation')][:]
            sc_data_ra        = obsfile[d.replace('timestamp','ra')][:]
            sc_data_dec       = obsfile[d.replace('timestamp','dec')][:]
            #
            c_min             = SkyCoord(ra=min(sc_data_ra)*u.degree, dec=min(sc_data_dec)*u.degree)
            c_max             = SkyCoord(ra=max(sc_data_ra)*u.degree, dec=max(sc_data_dec)*u.degree)
            #
            gain              = obsfile[d.replace('timestamp','power')][:]/obsfile[d.replace('timestamp','power').replace(info_data[2],info_data[2].replace('ND0','ND1'))][:]
            stats_gain        = STS.data_stats(gain,stats_type='mean')
            #
            satur             = obsfile[d.replace('timestamp','saturated_samples')][:].flatten()
            stats_satur       = STS.data_stats(satur,stats_type='mean')

            mask_data         = obsfile[d.replace('timestamp','')+'mask']
            masked_percentage = np.count_nonzero(mask_data)/np.cumprod(mask_data.shape)[-1]*100


            # determine velocities and acceleration
            #
            velo_dec,acce_dec =  generate_velo_acceleration(sc_data_dec,time_data.flatten())
            velo_ra,acce_ra   =  generate_velo_acceleration(sc_data_ra,time_data.flatten())
            #
            velo_az,acce_az =  generate_velo_acceleration(sc_data_az,time_data.flatten())
            velo_el,acce_el =  generate_velo_acceleration(sc_data_el,time_data.flatten())

            # determine statistics for velocities and acceleration
            #
            vstatstyp = 'madmedian'
            #
            velo_stats_ra  = STS.data_stats(velo_ra,stats_type=vstatstyp)
            velo_stats_dec = STS.data_stats(velo_dec,stats_type=vstatstyp)
            velo_stats_az  = STS.data_stats(velo_az,stats_type=vstatstyp)
            velo_stats_el  = STS.data_stats(velo_el,stats_type=vstatstyp)
            #
            acce_stats_ra  = STS.data_stats(acce_ra,stats_type=vstatstyp)
            acce_stats_dec = STS.data_stats(acce_dec,stats_type=vstatstyp)
            acce_stats_az  = STS.data_stats(acce_az,stats_type=vstatstyp)
            acce_stats_el  = STS.data_stats(acce_el,stats_type=vstatstyp)
            


            print('\n - Scan Info: ',info_data[1],'type ',info_data[2],'\n')

            print('\t - time range:                 ', min(acu_times)[0],max(acu_times)[0])
            print('\t - total time:                 ', max(time_data)[0]- min(time_data)[0],' [s]')
            print('\t - percentage masked:          ', masked_percentage, '[%]')
            print('\t - gain un-masked:             ', *stats_gain[:2], '[mean, std]')
            print('\t - saturation un-masked:       ', *stats_satur[:2], '[units]')
            #
            print('\t - Azimuth [min, max]:         ',min(sc_data_az),max(sc_data_az), '[deg, deg]')
            print('\t\t - Azimuth velo [min, max,',vstatstyp,', error]: ',min(velo_az),max(velo_az),velo_stats_az[0], velo_stats_az[1], '[deg/s, Delta deg/s]')
            print('\t\t - Azimuth acceleration [min, max,',vstatstyp,', error]: ',min(acce_az),max(acce_az),acce_stats_az[0], acce_stats_az[1], '[deg/s^2, Delta deg/s^2]')
            #
            print('\t - Elevation [min, max]:         ',min(sc_data_el),max(sc_data_el), '[deg, deg]')
            print('\t\t - Elevation velo [min, max,',vstatstyp,', error]: ',min(velo_el),max(velo_el),velo_stats_el[0], velo_stats_el[1], '[deg/s, Delta deg/s]')
            print('\t\t - Elevation acceleration [min, max,',vstatstyp,', error]: ',min(acce_el),max(acce_el),acce_stats_el[0], acce_stats_el[1], '[deg/s^2, Delta deg/s^2]')
            #
            print('\t - RA [min, max]:                ',min(sc_data_ra),max(sc_data_ra), '[deg, deg]')
            print('\t\t - RA velo [min, max,',vstatstyp,', error]: ',min(velo_ra),max(velo_ra),velo_stats_ra[0], velo_stats_ra[1], '[deg/s, Delta deg/s]')
            print('\t\t - RA acceleration [min, max,',vstatstyp,', error]: ',min(acce_ra),max(acce_ra),acce_stats_ra[0], acce_stats_ra[1], '[deg/s^2, Delta deg/s^2]')
            #
            print('\t - DEC [min, max]:               ',min(sc_data_dec),max(sc_data_dec), '[deg, deg]')
            print('\t\t - DEC velo [min, max,',vstatstyp,', error]: ',min(velo_dec),max(velo_dec),velo_stats_dec[0], velo_stats_dec[1], '[deg/s, Delta deg/s]')
            print('\t\t - DEC acceleration [min, max,',vstatstyp,', error]: ',min(acce_dec),max(acce_dec),acce_stats_dec[0], acce_stats_dec[1], '[deg/s^2, Delta deg/s^2]')
            #
            print('\t - RA, DEC [min | max]:        ',c_min.to_string('hmsdms'), ' | ',c_max.to_string('hmsdms'))                


            planets = ['sun    ','moon   ','jupiter']
            for p in planets:
                planet_separation = MPGHD.source_plant_separation(sc_data_ra,sc_data_dec,p.replace(' ',''),time_data,obs_pos).flatten()
                lowest_FOV        = MPGHD.fov_fwhm(freq[0],15,type='fov',outunit='deg')
                print('\t\t - distance to ',p,'       ',min(planet_separation),', FoV ',lowest_FOV,'[deg]')


def generate_velo_acceleration(coordinate_data,time_data):
    """
    generate velocity and acceleration array from input data
    """

    velo_array         = np.gradient(coordinate_data)/np.gradient(time_data) 
    acceleration_array = np.gradient(velo_array)/np.gradient(time_data) 

    return velo_array, acceleration_array

def get_json(filename,homedir=''):
    """
    get json info

    """
    with open(homedir+filename) as f:
        jsondata = json.load(f)

    return jsondata
