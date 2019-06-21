# Plotter for GOES ABI data
# Kyle Hilburn, CIRA/CSU
# June 21, 2019

# Example usage, command line:
#    python plot_abi.py abi_nc_filename           # will show figure on screen
#    python plot_abi.py abi_nc_filename figname   # will save figname

# Example usage, within program:
#     from read_abi import read_abi
#     from plot_abi import plot_abi
#     data = read_abi(abi_nc_file_name)
#     plot_abi(data)                              # will show figure on screen
# see comments in plot_abi() for information on optional arguments

################################################################

from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from read_abi import read_abi
import sys
import warnings

################################################################

def plot_abi(data, \
    anoprops = {}, \
    basemap = None, \
    cblabel = None, \
    cbprops = {}, \
    cmap = None, \
    coords = ('lat','lon'), \
    dpi = 100, \
    fieldtoplot = 'data', \
    figname = None, \
    figsize = (8,6), \
    norm = None, \
    overplot = None, \
    use_pcolor = False, \
    vmax = None, \
    vmin = None, \
    ):

#   required input:
#   data = data dictionary containing keys:
#          lat, lon, data, domain, mode, chan, sat, datetime (required)
#          band (if cmap not set)
#          llcrnrlon,urcrnrlon,llcrnrlon,urcrnrlon (if basemap not set)
#          the 'data' key is not necessary if 'fieldtoplot' is set to something else

#   optional input:
#   anoprops = anotation properties: kwargs for coastlines, countries, states
#   basemap = your own basemap, default: cylindrical with bounds from data
#   cblabel = colorbar label
#   cbprops = kwargs for colorbar
#   cmap = your own colormap, default defined below based on band number
#   coords = names of coordinate variables, default: ('lat','lon')
#   dpi = dots per inch, default: 100
#   fieldtoplot = the field to plot, default: data['data']
#                 note that tuples are treated as a dictionary within a dictionary
#   figname = your figure same (if you want to save the figure), default = None = do not save
#   figsize = your figure size (width,height) in inches, default: (8,6)
#   norm = norm for your colormap, default = None
#   overplot = a user defined function that plots over the image, default = None
#              function must have arguments: overplot(data,basemap)
#   use_pcolor = if True use pcolor instead of pcolormesh, default = False
#   vmax = your maximum value, only used if norm=None
#   vmin = your minimum value, only used if norm=None

#   output behavior:
#   if figname not set, output is showing figure on screen
#   if figname is set, then saved figure is output

    lat = data[coords[0]]
    lon = data[coords[1]]
    if 'tuple' not in str(type(fieldtoplot)):
        amap = data[fieldtoplot]
    else:
        amap = data[fieldtoplot[0]][fieldtoplot[1]]

    off_earth = ~np.isfinite(lon) | ~np.isfinite(lat)

    # workaround because matplotlib 3.0
    # does not support non-finite/masked values
    # in coordinate arrays in pcolormesh;
    # best to use pcolor, but is very slow!
    if not use_pcolor:
        if np.sum(off_earth) > 0:
            print("Note: masking coordinate locations off Earth's surface with 1.E30")
            lon[off_earth] = 1.E30
            lat[off_earth] = 1.E30
    else:
        lon = np.ma.masked_where(off_earth,lon)
        lat = np.ma.masked_where(off_earth,lat)

    if basemap == None:
        basemap = {}
        basemap['projection'] = 'cyl'
        basemap['resolution'] = 'l'
        basemap['llcrnrlon'] = data['llcrnrlon']
        basemap['urcrnrlon'] = data['urcrnrlon']
        basemap['llcrnrlat'] = data['llcrnrlat']
        basemap['urcrnrlat'] = data['urcrnrlat']

    basemap = Basemap(**basemap)
    with warnings.catch_warnings():
        # this is to catch: RuntimeWarning: invalid value encountered in greater
        # due to non-finite/masked values in coordinate arrays
        warnings.simplefilter('ignore')
        x,y = basemap(lon,lat)

    amap = np.ma.masked_where(off_earth,amap)
    amap = np.ma.masked_invalid(amap)

    if cmap == None:
        if data['band'] < 7:
            cmap = plt.get_cmap('binary_r')
        else:
            cmap = plt.get_cmap('jet')
        cmap.set_bad('black')

    fig = plt.figure(figsize=figsize)

    if not use_pcolor:
        if norm == None:
            pcm = basemap.pcolormesh(x,y,amap,cmap=cmap,vmin=vmin,vmax=vmax)
        else:
            pcm = basemap.pcolormesh(x,y,amap,cmap=cmap,norm=norm)
    else:
        stime = datetime.now()
        if norm == None:
            pcm = basemap.pcolor(x,y,amap,cmap=cmap,vmin=vmin,vmax=vmax)
        else:
            pcm = basemap.pcolor(x,y,amap,cmap=cmap,norm=norm)
        etime = datetime.now()
        dtime = (etime-stime).total_seconds()
        print('pcolor took '+str(dtime)+' seconds to render image')

    cb = plt.colorbar(pcm,**cbprops)
    if cblabel != None:
        cb.set_label(cblabel)

    basemap.drawcoastlines(**anoprops)
    basemap.drawcountries(**anoprops)
    basemap.drawstates(**anoprops)

    if overplot != None:
        overplot(data,basemap)

    data['isodatestring'] = data['datetime'].isoformat()+'Z'
    title = ', '.join([data[anitem] for anitem in ['domain','mode','chan','sat','isodatestring']])
    plt.title(title)

    if figname == None:
        plt.show()
    else:
        fig.savefig(figname,dpi=dpi)
        print('saving: '+figname)

################################################################

if __name__=='__main__':

    try:
        abi_nc_file_name = sys.argv[1]
    except IndexError:
        sys.exit('You must supply an ABI netcdf file name')

    try:
        figname = sys.argv[2]
    except IndexError:
        figname = None

    data = read_abi(abi_nc_file_name,verbose=True)

    plot_abi(data,figname=figname)
