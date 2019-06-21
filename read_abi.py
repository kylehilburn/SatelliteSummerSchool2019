# Reader for GOES ABI data
# Kyle Hilburn, CIRA/CSU
# June 20, 2019

# Example usage, command line:
#    python read_abi.py abi_nc_filename

# Example usage, within program:
#    from read_abi import read_abi
#    data = read_abi(abi_nc_file_name)

################################################################

from datetime import datetime
from netCDF4 import Dataset
import numpy as np
import re
import sys
import warnings

################################################################

# regex groups: domain, mode, channel, satellite, start date
filename_pattern='Rad([0-9A-Z]+)-M([0-9]{1})C([0-9]{2})_G([0-9]{2})_s([0-9]{13})'
filename_time_format = '%Y%j%H%M%S'

proj_params = []
proj_params.append('perspective_point_height')
proj_params.append('semi_major_axis')
proj_params.append('semi_minor_axis')
proj_params.append('longitude_of_projection_origin')

################################################################

def get_abi_latlon(x,y,proj):

#   Reference:
#   Product Definition and User's Guide (PUG)
#   Volume 5: Level 2+ Products
#   DCN 7035538, Revision E
#   Equations from Section 4.2.8.1

#   Input:
#   x = x-coordinate (array, rad)
#   y = y-coordinate (array, rad)
#   proj = goes imager projection parameters (dictionary)

#   Output:
#   lon = geodetic longitude (array, deg)
#   lat = geodetic latitude (array, deg)

    requ = proj['semi_major_axis']
    rpol = proj['semi_minor_axis']
    hgt = proj['perspective_point_height']
    lon0 = proj['longitude_of_projection_origin']

    bigh = requ + hgt
    rrat = (requ*requ)/(rpol*rpol)
    lam0 = np.radians(lon0)

    a = np.sin(x)**2 + (np.cos(x)**2)*( np.cos(y)**2 + rrat*(np.sin(y)**2) )
    b = -2*bigh*np.cos(x)*np.cos(y)
    c = bigh**2 - requ**2
    with warnings.catch_warnings():
        # this is to catch: RuntimeWarning: invalid value encountered in sqrt
        # that occurs when sensor is viewing off Earth's surface
        warnings.simplefilter('ignore')
        rs = (-b - np.sqrt(b*b - 4.*a*c))/(2.*a)
    sx = rs * np.cos(x) * np.cos(y)
    sy = -rs * np.sin(x)
    sz = rs * np.cos(x) * np.sin(y)

    lat = np.degrees(np.arctan(rrat*sz/np.sqrt((bigh-sx)**2 + sy**2)))
    lon = np.degrees(lam0 - np.arctan(sy/(bigh-sx)))

    return lat,lon

################################################################

def read_abi(filename,verbose=False,getlatlon=True):

#   Input:
#   filename = ABI netcdf file name string
#   verbose = default False; if True, print additional information
#   getlatlon = default True; if False, skip getting lat,lon info (speeds reading)

#   Output:
#   data = dictionary of ABI data

    data = {}

    amatch = re.search(filename_pattern,filename)
    if not amatch:
        sys.exit('filename regex error in read_abi')
    adomain = 'Rad'+amatch.group(1)
    amode = 'M'+amatch.group(2)
    achan = 'C'+amatch.group(3)
    asat = 'G'+amatch.group(4)
    adate = amatch.group(5)
    adatetime = datetime.strptime(adate,filename_time_format)
    if verbose:
        print("reading GOES ABI for domain, mode, channel, satellite, date = ",adomain,amode,achan,asat,adate)
    data['domain'] = adomain
    data['mode'] = amode
    data['chan'] = achan
    data['sat'] = asat
    data['datestring'] = adate
    data['datetime'] = adatetime

    ds = Dataset(filename,'r')

    proj = {}
    for aparam in proj_params:
        proj[aparam] = getattr(ds.variables['goes_imager_projection'],aparam)
    data['proj'] = proj

    data['llcrnrlon'] = ds.variables['geospatial_lat_lon_extent'].geospatial_westbound_longitude
    data['urcrnrlon'] = ds.variables['geospatial_lat_lon_extent'].geospatial_eastbound_longitude
    data['llcrnrlat'] = ds.variables['geospatial_lat_lon_extent'].geospatial_southbound_latitude
    data['urcrnrlat'] = ds.variables['geospatial_lat_lon_extent'].geospatial_northbound_latitude
    data['lon_center'] = ds.variables['geospatial_lat_lon_extent'].geospatial_lon_center
    data['lat_center'] = ds.variables['geospatial_lat_lon_extent'].geospatial_lat_center
    data['lon_nadir'] = ds.variables['geospatial_lat_lon_extent'].geospatial_lon_nadir
    data['lat_nadir'] = ds.variables['geospatial_lat_lon_extent'].geospatial_lat_nadir

    x = np.array(ds.variables['x'])
    y = np.array(ds.variables['y'])
    data['x'] = x
    data['y'] = y
    if getlatlon:
        xx,yy = np.meshgrid(x,y)
        lat,lon = get_abi_latlon(xx,yy,proj)
        data['lat'] = lat
        data['lon'] = lon

    band = ds.variables['band_id'][0]
    rad = ds.variables['Rad'][...]
    if band < 7:
        kappa0 = ds.variables['kappa0'][0]
        amap = rad * kappa0
    else:
        c1 = ds.variables['planck_fk1'][0]
        c2 = ds.variables['planck_fk2'][0]
        bc1 = ds.variables['planck_bc1'][0]
        bc2 = ds.variables['planck_bc2'][0]
        bt = c2 / np.log((c1/rad)+1.)
        amap = (bt - bc1)/bc2

    if verbose:
        isgood = np.isfinite(amap)
        try:
            print('dtype,shape,nbad,min,max=',\
                amap.dtype,amap.shape,np.sum(~isgood),\
                    np.min(amap[isgood]),np.max(amap[isgood]) )
        except ValueError:
            print('min,max=',np.min(amap), np.max(amap))
            print('warning, whole array is bad/missing: ',filename)

    data['data'] = amap
    data['band'] = band

    ds.close()

    return data

################################################################

if __name__=='__main__':

    try:
        abi_nc_file_name = sys.argv[1]
    except IndexError:
        sys.exit('You must supply an ABI netcdf file name')

    data = read_abi(abi_nc_file_name,verbose=True)
