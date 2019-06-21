# Utility functions for abi_exercises.py
# Kyle Hilburn, CIRA/CSU
# June 21, 2019

import copy
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import warnings

# available functions:
# compass()
# define_xsec()
# find_anvils()
# find_eye()
# find_ots()
# get_ij_bounds()
# parallax()
# plot_anvils()
# plot_ots()
# plot_reports()
# plot_xsec()
# plot_xsec_path()
# read_reports()
# sandwich()
# smooth_image()

#---------------------------------------------------------------

def compass(lat0,lon0,lat1,lon1):
#   input:
#   lat0,lon0 = start point
#   lat1,lon1 = end point
#   output:
#   rng    great circle distances (km) from (lat0,lon0) to (lat1,lon1)
#   azm    azimuth angles (degrees), azimuth is measured clockwise
#          from due north (compass heading)
    erad = 6371.2
    t0=np.radians(90.-lat0)
    t1=np.radians(90.-lat1)
    p0=np.radians(lon0)
    p1=np.radians(lon1)
    zz=np.cos(t0)*np.cos(t1)+np.sin(t0)*np.sin(t1)*np.cos(p1-p0)
    try:
        zz[zz<-1.] = -1
        zz[zz>1] = 1
    except TypeError:
        if zz < -1: zz=-1
        if zz > 1: zz=1
    xx=np.sin(t1)*np.sin(p1-p0)
    yy=np.sin(t0)*np.cos(t1)-np.cos(t0)*np.sin(t1)*np.cos(p1-p0)
    rng=erad*np.arccos(zz)
    azm = np.degrees(np.arctan2(xx,yy))
    return rng,azm

#---------------------------------------------------------------

def define_xsec(lat,lon,vals,pt1,pt2):
#   this defines a cross-sectional cut across an image
#   input:
#   lat = latitude array
#   lon = longitude array
#   vals = array of data values
#   pt1 = start point xsec (lat,lon) 
#   pt2 = end point xsec (lat,lon) 
#   output:
#   returns dictionary describing xsec, with keys:
#   npts = number points in xsec
#   type = type of xsec ('lat' or 'lon')
#   ilats, ilons = index arrays along xsec
#   lats,lons = lat,lon arrays along xsec
#   values = values along xsec
    lat1,lon1 = pt1
    lat2,lon2 = pt2
    print('creating cross section for user defined lat1,lon1,lat2,lon2=',lat1,lon1,lat2,lon2)
    imin1 = np.argmin(np.abs(lat-lat1)+np.abs(lon-lon1))
    ilat1,ilon1 = np.unravel_index(imin1,lat.shape)
    imin2 = np.argmin(np.abs(lat-lat2)+np.abs(lon-lon2))
    ilat2,ilon2 = np.unravel_index(imin2,lat.shape)
    nlat = np.abs(ilat2-ilat1)+1
    nlon = np.abs(ilon2-ilon1)+1
    if nlat > nlon:
        nindx = nlat
        atype = 'lat'
    else:
        nindx = nlon
        atype = 'lon'
    ilats = np.linspace(ilat1,ilat2,nindx,dtype=np.int32)
    ilons = np.linspace(ilon1,ilon2,nindx,dtype=np.int32)
    lats = []
    lons = []
    values = []
    for i in range(nindx):
        lats.append(lat[ilats[i],ilons[i]])
        lons.append(lon[ilats[i],ilons[i]])
        values.append(vals[ilats[i],ilons[i]])
    lats = np.array(lats,dtype=np.float32)
    lons = np.array(lons,dtype=np.float32)
    values = np.array(values,dtype=np.float32)
    return {'npts':nindx,'type':atype,'ilats':ilats,'ilons':ilons,'lats':lats,'lons':lons,'values':values}

#---------------------------------------------------------------

def find_anvils(data,threshold,fill_holes=True,areathr=40000,field='enhanced'):
#   input:
#   data = data dictionary from read_abi.py
#   threshold = threshold on data for identifying objects
#   fill_holes = fill holes in objects, default: True
#   areathr = area threshold for objects
#             default: 40,000 pixels ~ 10,000 km^2 = "1 Weld County" for Channel 02
#   field = field to analyze, default: 'enhanced'
#   output:
#   anvil dictionary, with keys:
#   bmask = binary mask of anvil candidates
#   nfeat = number of objects (meeting area threshold)
#   rank = array with object area rank (for objects meeting area threshold)
#   ots_cands = overshooting top candidates
    label_structure = np.ones((3,3))  #include diagonal connectivity
    img = data[field]
    oarray = np.ones(img.shape)
    bmask = np.zeros(img.shape)
    print('create binary mask using threshold of',str(threshold))
    bmask[img>threshold] = 1
    nofill = copy.deepcopy(bmask)
    if fill_holes:
        bmask = ndimage.morphology.binary_fill_holes(bmask)
    labels,numfeat = ndimage.measurements.label(bmask,structure=label_structure)
    print(str(numfeat)+' objects found')
    if numfeat == 0:
        return None
    objects = []
    nlarge = 0
    for ifeat in range(numfeat):
        anarea = np.sum(oarray[labels==ifeat+1])
        if anarea < areathr: continue
        objects.append((ifeat+1,anarea))
        nlarge += 1
    print(str(nlarge)+' objects found meeting area threshold of '+str(areathr)+' pixels')
    objects = sorted(objects,key=lambda tup:tup[1],reverse=True)  #sort by area
    rank = np.zeros(img.shape,dtype=np.int32)
    for itup,atup in enumerate(objects):
        ilabel,anarea = atup
        anobj = labels==ilabel
        rank[anobj] = itup+1
        print(itup+1,anarea)
    ots = np.zeros(img.shape)
    ots[(rank > 0) & (nofill == 0)] = 1
    return {'bmask':bmask, 'nfeat':nlarge, 'rank':rank, 'ots_cands':ots}
        
#---------------------------------------------------------------

def find_eye(data,basemap):
#   to be called as "overplot" function in plot_abi.py
#   this function finds the hurricane eye,
#   prints the location and tb value,
#   and plots the location on an image
    imax = np.argmax(data['diff'])
    iy,ix = np.unravel_index(imax,data['diff'].shape)
    elat = data['lat'][iy,ix]
    elon = data['lon'][iy,ix]
    print('eye iy,ix=',iy,ix)
    print('eye lat,lon=',elat,elon)
    print('eye tb=',data['data'][iy,ix])
    print('max tb in +/- 50 pixel area=',np.max(data['data'][iy-50:iy+50,ix-50:ix+50]))
    ex,ey = basemap(elon,elat)
    plt.text(ex,ey,'E',ha='center',va='center',fontsize='large',color='black')

#---------------------------------------------------------------

def find_ots(data,ptile=99.5,field='enhanced'):
#   find overshooting tops
#   input:
#   data = data dictionary from read_abi.py, with output from find_anvils() in data['anvils']
#   ptile = texture percentile for overshooting tops (default=99.5%)
#   field = field to analyze
#   output:
#   mask of overshooting tops
    amap = data[field]
    dx = ndimage.filters.sobel(amap,axis=0,mode='mirror')
    dy = ndimage.filters.sobel(amap,axis=1,mode='mirror')
    texture = np.sqrt(dx*dx+dy*dy)
    texture = ndimage.filters.uniform_filter(texture,(5,5),mode='mirror',origin=0)
    texture[data['anvils']['rank']==0] = 0.
    thr = np.percentile(texture,ptile)
    print(str(ptile)+' % texture threshold=',str(thr))
    ots = np.zeros(amap.shape)
    ots[(texture > thr) & (data['anvils']['ots_cands']==1)] = 1
    return ots

#---------------------------------------------------------------

def get_ij_bounds(lat,lon,bounds):
#   the purpose of this function is to get indices for a bounding region
#   this is to speed up plotting of small parts of large sectors (e.g., RadF)
#   and to avoid problems with non-finite/masked coordinate arrays
#   input:
#   lat, lon = latitude, longitude arrays
#   bounds = corners of region of interest (llcrnrlon,urcrnrlon,llcrnrlat,urcrnrlat)
#   output:
#   returns indices for subsetting region [iy1:iy2,ix1:ix2]
    llcrnrlon,urcrnrlon,llcrnrlat,urcrnrlat = bounds
    with warnings.catch_warnings():
        # this is to catch: RuntimeWarning: invalid value encountered in greater/less
        warnings.simplefilter('ignore')
        good = (lon > llcrnrlon) & (lon < urcrnrlon) & \
            (lat > llcrnrlat) & (lat < urcrnrlat)
    ny,nx = lat.shape
    iy = np.array(range(ny))
    ix = np.array(range(nx))
    ix,iy = np.meshgrid(ix,iy)
    ix = np.ma.masked_where(~good,ix)
    iy = np.ma.masked_where(~good,iy)
    iy1 = np.min(iy)
    iy2 = np.max(iy)
    ix1 = np.min(ix)
    ix2 = np.max(ix)
    return (iy1,iy2,ix1,ix2)

#---------------------------------------------------------------

def parallax(sublat,sublon,clat,clon,chgt):
#   correct parallax displacement
#   input:
#   sublat = sub-satellite latitude (0 deg for GEO)
#   sublon = sub-satellite longitude
#   clat = apparent cloud latitude
#   clon = apparent cloud longitude
#   chgt = height of cloud (km)
#   output:
#   alat = actual latitude
#   alon = actual longitude
    # parameters:
    req = 6378.1   #equatorial radius (km)
    rpo = 6356.6   #polar radius (km)
    rob = req/rpo  #oblateness
    rsat = 35786.0 + req  #radius of geo-satellite
    # convert to radians
    sublatr = np.radians(sublat)
    sublonr = np.radians(sublon)
    clatr = np.radians(clat)
    clonr = np.radians(clon)
    # equivalent Earth's radius
    aax = np.cos(clatr)**2
    bax = np.sin(clatr)**2
    eer = req / np.sqrt( aax + bax*(rob**2) )
    # apparent cloud location in Cartesian coordinates
    xc = eer * np.cos(clatr) * np.sin(clonr)
    yc = eer * np.sin(clatr)
    zc = eer * np.cos(clatr) * np.cos(clonr)
    # satellite location in Cartesian coordinates
    xs = rsat * np.cos(sublatr) * np.sin(sublonr)
    ys = rsat * np.sin(sublatr)
    zs = rsat * np.cos(sublatr) * np.cos(sublonr)
    # quadratic equation parameters
    bpar = ((req + chgt)/(rpo + chgt))**2
    cpar = (xs-xc)**2 + bpar*(ys-yc)**2 + (zs-zc)**2
    dpar = 2.0*( xc*(xs-xc) + bpar*yc*(ys-yc) + zc*(zs-zc) )
    epar = xc**2 + bpar*(yc**2) + zc**2 - (req+chgt)**2
    apar = ((-1.0*dpar) + np.sqrt(dpar**2 - 4.0*cpar*epar)) / (2.0*cpar)
    # actual location of cloud in Cartesian coordinates
    xa = xc + apar*(xs-xc)
    ya = yc + apar*(ys-yc)
    za = zc + apar*(zs-zc)
    # actual latitude
    alat = np.degrees(np.arctan(ya/np.sqrt(xa**2 + za**2)))
    # actual longitude
    alon = np.zeros(alat.shape)
    isgt = za > 0.0
    islt = za < 0.0
    iseqn = (za == 0.0) & (xa >= 0.0)
    iseqs = (za == 0.0) & (xa < 0.0)
    alon[isgt] = np.degrees(np.arctan(xa/za))[isgt]
    alon[islt] = (np.degrees(np.arctan(xa/za)) - 180.0)[islt]
    alon[iseqn] = 90.
    alon[iseqs] = -90.
    return alat,alon

#---------------------------------------------------------------

def plot_anvils(data,basemap,plotlabels=False):
#   data dictionary should have output from find_anvils() in data['anvils']
#   to be called as "overplot" function in plot_abi.py
#   optional argument: plotlabels = include rank labels on image, default: False
    rank = data['anvils']['rank']
    bmask = data['anvils']['bmask']
    nfeat = data['anvils']['nfeat']
    lat = data['lat']
    lon = data['lon']
    x,y = basemap(lon,lat)
    with warnings.catch_warnings():
        #This is to catch: UserWarning: No contour levels were found within the data range.
        warnings.simplefilter('ignore')
        basemap.contour(x,y,rank,[0],colors='blue',linestyles='solid',linewidths=2)
    if plotlabels:
        for ifeat in range(nfeat):
            com = ndimage.measurements.center_of_mass(bmask,rank,index=ifeat+1)
            ilat,ilon = int(com[0]),int(com[1])
            xf,yf = basemap(lon[ilat,ilon],lat[ilat,ilon])
            plt.text(xf,yf,str(ifeat+1),color='blue',\
                va='center',ha='center')

#---------------------------------------------------------------

def plot_ots(data,basemap):
#   data dictionary should have output from find_ots() in data['ots']
#   to be called as "overplot" function in plot_abi.py
#   overplot the anvil boundaries and overshooting tops
    rank = data['anvils']['rank']
    ots = data['ots']
    lat = data['lat']
    lon = data['lon']
    x,y = basemap(lon,lat)
    with warnings.catch_warnings():
        #This is to catch: UserWarning: No contour levels were found within the data range.
        warnings.simplefilter('ignore')
        basemap.contour(x,y,rank,[0],colors='blue',linestyles='solid',linewidths=2)
        basemap.contour(x,y,ots,[0],colors='red',linestyles='solid',linewidths=2)

#---------------------------------------------------------------

def plot_reports(data,basemap,ms=20):
#   data dictionary should have output from read_reports() in data['reports']
#   to be called as "overplot" function in plot_abi.py
#   overplot storm reports
#   optional arguement: ms = marker size, default 20
    reports = data['reports']
    treps = []
    hreps = []
    wreps = []
    for arep in reports:
        ax,ay = basemap(arep['lon'],arep['lat'])
        ec = 'none'
        if arep['type'] == 'wind':
            if arep['mag'] == None: continue
            if arep['mag'] >= 75: ec='black'
        if arep['type'] == 'hail':
            if arep['mag'] == None: continue
            if arep['mag'] >= 2.0: ec='black'
        if arep['type'] == 'tornado': treps.append((ax,ay,ec))
        if arep['type'] == 'hail': hreps.append((ax,ay,ec))
        if arep['type'] == 'wind': wreps.append((ax,ay,ec))
    # plot order bottom to top: wind, hail, tornado
    for arep in wreps:
        ax,ay,ec = arep
        basemap.scatter(ax,ay,marker='o',c='blue',edgecolors=ec,s=ms)
    for arep in hreps:
        ax,ay,ec = arep
        basemap.scatter(ax,ay,marker='o',c='green',edgecolors=ec,s=ms)
    for arep in treps:
        ax,ay,ec = arep
        basemap.scatter(ax,ay,marker='o',c='red',edgecolors=ec,s=ms)

#---------------------------------------------------------------

def plot_xsec(data,eye=(0,0,0),tbthr=200.):
#   produce a figure showing the cross section
#   input:
#   data = data dictionary from read_abi.py, with output from define_xsec() in data['xsec']
#   eye = (latitude,longitude,brightness temperature) tuple for eye
#   tbthr = brightness temperature threshold for edge of eye, default = 200 K
#   output: show plot on screen
    print('find edge of the eye using a '+str(tbthr)+' K threshold')
    atype = data['xsec']['type']
    npts = data['xsec']['npts']
    z = data['xsec']['values']
    if atype == 'lat':
        x = data['xsec']['lats']
        y = data['xsec']['lons']
        xlabel = 'Latitude'
        ex = eye[0]
    else:
        x = data['xsec']['lons']
        y = data['xsec']['lats']
        xlabel = 'Longitude'
        ex = eye[1]
    fig = plt.figure()
    plt.plot(x,z)
    plt.text(ex,eye[2],'E',ha='center',va='center',fontsize='large',color='black')
    ieye = np.argmin(np.abs(x-ex))
    print('eye tb, xsec eye tb=',eye[2],z[ieye])
    for i in np.arange(ieye,npts+1):
        if z[i] <= tbthr:
            xright = x[i]
            yright = y[i]
            zright = z[i]
            break
    for i in np.arange(ieye,-1,-1):
        if z[i] <= tbthr:
            xleft = x[i]
            yleft = y[i]
            zleft = z[i]
            break
    if atype == 'lat':
        print('south,north eye latitudes=',xleft,xright)
        rng,azm = compass(xleft,yleft,xright,yright)
        print('eye width (km)=',rng)
        plt.scatter(x[ieye],z[ieye],color='red')
        plt.scatter(xleft,zleft,color='blue')
        plt.scatter(xright,zright,color='blue')
        plt.plot([x[0],x[-1]],[tbthr,tbthr],color='blue')
    else:
        print('west,east eye longitudes=',xleft,xright)
        rng,azm = compass(yleft,xleft,yright,xright)
        print('eye width (km)=',rng)
        plt.scatter(x[ieye],z[ieye],color='red')
        plt.scatter(xleft,zleft,color='blue')
        plt.scatter(xright,zright,color='blue')
        plt.plot([x[0],x[-1]],[tbthr,tbthr],color='blue')
    plt.xlabel(xlabel)
    plt.ylabel('Brightness Temperature (K)')
    plt.grid()
    plt.show()

#---------------------------------------------------------------

def plot_xsec_path(data,basemap):
#   data dictionary should have output from define_xsec() in data['xsec']
#   to be called as "overplot" function in plot_abi.py
#   overplot the path of the cross-section
    lats = data['xsec']['lats']
    lons = data['xsec']['lons']
    xs,ys = basemap(lons,lats)
    basemap.plot(xs,ys,color='black')

#---------------------------------------------------------------

def read_reports(datafile):
#   read storm reports
#   input: datafile = file name of storm reports
#   output: reports list
#   each list element is dictionary with keys:
#   type = 'wind', 'hail', or 'tornado'
#   time = time of report
#   lat = latitude of report
#   lon = longitude of report
#   mag = magnitude of report, or None
    reports = []
    f = open(datafile,'r')
    for aline in f:
        items = [anitem.strip() for anitem in aline.split()]
        data = {}
        data['type'] = items[0]
        data['time'] = datetime.strptime(items[1],'%Y%m%d%H%MZ')
        data['lat'] = float(items[2])
        data['lon'] = float(items[3])
        if items[4] == 'None':
            data['mag'] = None
        else:
            data['mag'] = float(items[4])
        reports.append(data)
    f.close()
    return reports

#---------------------------------------------------------------

def sandwich(data,basemap,vmin=200,vmax=240,alpha=0.3):
#   overplot C13 on C02 to make "sandwich" product
#   data dictionary should have C13 data in data['overplot']
#   to be called as "overplot" function in plot_abi.py
    lat = data['overplot']['lat']
    lon = data['overplot']['lon']
    amap = data['overplot']['data']
    off_earth = ~np.isfinite(lon) | ~np.isfinite(lat)
    if np.sum(off_earth) > 0:
        print("Note: masking coordinate locations off Earth's surface with 1.E30")
        lon[off_earth] = 1.E30
        lat[off_earth] = 1.E30
    with warnings.catch_warnings():
        # this is to catch: RuntimeWarning: invalid value encountered in greater
        # due to non-finite/masked values in coordinate arrays
        warnings.simplefilter('ignore')
        x,y = basemap(lon,lat)
    amap = np.ma.masked_where(off_earth,amap)
    amap = np.ma.masked_invalid(amap)
    amap = np.ma.masked_where(amap>vmax,amap)
    cmap = plt.get_cmap('jet_r')
    pcm = basemap.pcolormesh(x,y,amap,cmap=cmap,vmin=vmin,vmax=vmax,alpha=alpha,\
        antialiased=True,linewidth=0)  #last two args to reduce lines in pcolormesh with transparency
    cb = plt.colorbar(pcm)
    cb.set_label('Brightness Temperature (K)')
    cb.set_alpha(1)  #to remove lines in colorbar with transparency
    cb.draw_all()  #to remove lines in colorbar with transparency

#---------------------------------------------------------------

def smooth_image(animage,sigma,**kwargs):
#   Gaussian smooth an image
#   input:
#   animage = the image to smooth
#   sigma = standard deviation for Gaussian kernel
#   kwargs = optional keyword arguments
#   output: smoothed image
    return ndimage.gaussian_filter(animage,sigma,**kwargs)

#---------------------------------------------------------------

