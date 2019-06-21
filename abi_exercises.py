# ABI Data Manipulation Exercises
# Kyle Hilburn, CIRA/CSU
# June 21, 2019

import numpy as np
from plot_abi import plot_abi
from read_abi import read_abi
import sys
import utilities as util

try:
    exercise = sys.argv[1]
except IndexError:
    sys.exit('you must supply exercise name as argument')

# Valid exercise names:
# 1a, 1b, 1c, 1d, 1e
# 2a, 2b, 2c, 2d, 2e
# runall

runall = False
if exercise=='runall': runall = True

datafiles = []
datafiles.append('OR_ABI-L1b-RadF-M3C13_G16_s20182951600373_e20182951611151_c20182951611211.nc')
datafiles.append('OR_ABI-L1b-RadM1-M6C02_G16_s20191252300365_e20191252300422_c20191252300459.nc')
datafiles.append('OR_ABI-L1b-RadM1-M6C13_G16_s20191252300365_e20191252300434_c20191252300468.nc')

#---------------------------------------------------------------

if exercise.startswith('1') or runall:

    print()
    print('Exercise 1, Hurricane Willa off the west coast of Mexico')

    data = read_abi(datafiles[0],verbose=True)

    #iy1,iy2,ix1,ix2 = util.get_ij_bounds(data['lat'],data['lon'],(-120,-80,10,35))
    iy1,iy2,ix1,ix2 = util.get_ij_bounds(data['lat'],data['lon'],(-115,-100,15,25))
    data['lat'] = data['lat'][iy1:iy2,ix1:ix2]
    data['lon'] = data['lon'][iy1:iy2,ix1:ix2]
    data['data'] = data['data'][iy1:iy2,ix1:ix2]

    basemap = {}
    basemap['projection'] = 'cyl'
    basemap['resolution'] = 'l'
    basemap['llcrnrlon'] = -115
    basemap['urcrnrlon'] = -100
    basemap['llcrnrlat'] = 15
    basemap['urcrnrlat'] = 25
    #basemap['fix_aspect'] = False

    if exercise == '1a' or runall:
        print('Exercise 1a: Plot the image')
        print('Comment: The first step in any satellite analysis is defining the region of interest.')
        print('indices for region of interst, iy1,iy2,ix1,ix2 =',iy1,iy2,ix1,ix2)
        print('Question: How many ways can you think of to identify hurricane eyes in satellite imagery, and what are their strengths and weaknesses?')
        plot_abi(data,basemap=basemap,vmin=200,vmax=300,cblabel='Brightness Temperature (K)')
        if not runall: sys.exit()

    data['smoothed'] = util.smooth_image(data['data'],10)
    if exercise == '1b' or runall:
        print('Exercise 1b: Plot smoothed image')
        plot_abi(data,basemap=basemap,vmin=200,vmax=300,fieldtoplot='smoothed',cblabel='Brightness Temperature (K)')
        if not runall: sys.exit()

    data['diff'] = data['data'] - data['smoothed']
    if exercise == '1c' or runall:
        print('Exercise 1c: Plot difference of raw minus smoothed image, locate eye')
        plot_abi(data,basemap=basemap,overplot=util.find_eye,fieldtoplot='diff',cblabel='Brightness Temperature (K)')
        if not runall: sys.exit()

    #eye lat,lon= 19.069155 -107.28587
    #eye tb= 276.49625
    if exercise == '1d' or runall:
        print('Exercise 1d: Create Cross Section in Longitude')
        print('Comment: Using the eye coordinates, define cross-sections.')
        data['xsec'] = util.define_xsec(data['lat'],data['lon'],data['data'],(19.11,-115),(19.11,-100))
        plot_abi(data,basemap=basemap,overplot=util.plot_xsec_path,cblabel='Brightness Temperature (K)')
        print('Comment: The edge of the eye shown by blue dots, with brightness temperature threshold as blue line, the actual eye shown by "E", and the eye in the cross-section with a red dot.  If the cross-section cuts across the middle of the eye, the red dot and "E" should be overlapping.')
        util.plot_xsec(data,eye=(19.069155,-107.28587,276.49625))
        if not runall: sys.exit()

    if exercise == '1e' or runall:
        print('Exercise 1e: Create Cross Section in Latitude')
        print('Comment: You can increase the accuracy of your eye width estimate by zooming in more and taking more cross-sections.')
        data['xsec'] = util.define_xsec(data['lat'],data['lon'],data['data'],(15,-107.45),(25,-107.45))
        plot_abi(data,basemap=basemap,overplot=util.plot_xsec_path,cblabel='Brightness Temperature (K)')
        util.plot_xsec(data,eye=(19.069155,-107.28587,276.49625))
        if not runall: sys.exit()
  
#---------------------------------------------------------------

if exercise.startswith('2') or runall:

    print()
    print('Exercise 2, Lubbock TX supercell')

    data = read_abi(datafiles[1],verbose=True)

    basemap = {}
    basemap['projection'] = 'cyl'
    basemap['resolution'] = 'l'
    basemap['llcrnrlon'] = -103.5
    basemap['urcrnrlon'] = -98.5
    basemap['llcrnrlat'] = 32
    basemap['urcrnrlat'] = 37

    if exercise == '2a' or runall:
        print('Exercise 2a: Plot the image, no enhancement')
        plot_abi(data,basemap=basemap,vmin=0,vmax=1,cblabel='Visible Reflectance Factor')
        print('Comment: Now show image with square root enhancement.')
        data['enhanced'] = np.sqrt(data['data'])
        plot_abi(data,basemap=basemap,vmin=0,vmax=1,fieldtoplot='enhanced',cblabel='Visible Reflectance Factor')
        print('Comment: Now show image with squared enhancement.')
        data['enhanced'] = np.power(data['data'],2)
        plot_abi(data,basemap=basemap,vmin=0,vmax=1,fieldtoplot='enhanced',cblabel='Visible Reflectance Factor')
        if not runall: sys.exit()

    data['enhanced'] = np.sqrt(data['data'])
    if exercise == '2b' or runall:
        print('Exercise 2b: Locate thunderstorm anvils')
        print('Comment: We will use the square root enhancement.')
        plot_abi(data,basemap=basemap,vmin=0,vmax=1,fieldtoplot='enhanced',cblabel='Visible Reflectance Factor')
        data['anvils'] = util.find_anvils(data,0.5,fill_holes=False)
        print('Comment: Here is the binary mask showing the anvils. Note the presence of many small objects that are not anvils, and note the "holes" in the anvil objects due to shadows cast by overshooting tops.')
        plot_abi(data,basemap=basemap,vmin=0,vmax=1,fieldtoplot=('anvils','bmask'),cblabel='Visible Reflectance Factor')
        print('Comment: Applying an area threshold removes the small clouds that we are not interested in.')
        plot_abi(data,basemap=basemap,vmin=0,vmax=1,fieldtoplot=('anvils','rank'),cblabel='Visible Reflectance Factor')
        print('Comment: Now apply morphological hole filling.')
        data['anvils'] = util.find_anvils(data,0.5)
        print('Comment: Note that this removes the dark shadow areas inside the anvil objects.')
        plot_abi(data,basemap=basemap,vmin=0,vmax=1,fieldtoplot=('anvils','rank'),cblabel='Visible Reflectance Factor') 
        print('Comment: Plot the anvil boundaries on the visible image as blue contours.')
        plot_abi(data,basemap=basemap,vmin=0,vmax=1,fieldtoplot='enhanced',overplot=util.plot_anvils,cblabel='Visible Reflectance Factor')

    if exercise == '2c' or runall:
        print('Exercise 2c: Locate thunderstorm overshooting tops')
        print('Comment: Identify overshooting tops (red contours) by the shadows they cast, the presence of strong image texture, and must be located within anvil bounds.')
        print('Question: Can you see why it is difficult to identify overshooting tops when they occur near the edge of the anvil? Can you think of ways to solve this issue?')
        data['ots'] = util.find_ots(data)
        plot_abi(data,basemap=basemap,vmin=0,vmax=1,fieldtoplot='enhanced',overplot=util.plot_ots,cblabel='Visible Reflectance Factor')
        if not runall: sys.exit()

    if exercise == '2d' or runall:
        print('Exercise 2d: Overplot storm reports')
        print('Comments: Plotting SPC filtered reports within +/- 15 minutes of satellite image.')
        print('Markers show reports for tornadoes(red), hail(green), and wind(blue).')
        print('Note the association of storm reports with overshooting tops.')
        print('The parallax effect causes objects in the imagery to appear to be displaced away from the true location along the path from the sub-satellite point and the true location. For GOES-16 this means most objects over CONUS appear displaced northwest of their true location.')
        data['reports'] = util.read_reports('storm_reports.txt')
        plot_abi(data,basemap=basemap,vmin=0,vmax=1,fieldtoplot='enhanced',overplot=util.plot_reports,cblabel='Visible Reflectance Factor')
        print('GOES-16 longitude=',str(data['proj']['longitude_of_projection_origin']))
        print('Adjust for GOES-16 parallax assuming cloud height of 12.0 km')
        print('Region bounds before parallax correction =',np.nanmin(data['lat']),np.nanmax(data['lat']),np.nanmin(data['lon']),np.nanmax(data['lon']))
        data['plat'],data['plon'] = util.parallax(0.0,-75.0,data['lat'],data['lon'],12.0)
        print('Region bounds after parallax correction =',np.nanmin(data['plat']),np.nanmax(data['plat']),np.nanmin(data['plon']),np.nanmax(data['plon']))
        rng,azm = util.compass(data['lat'],data['lon'],data['plat'],data['plon'])  #from apparent to true location
        print('Min,Mean,Max Shift (km)=',np.min(rng),np.mean(rng),np.max(rng))
        print('Min,Mean,Max Azim (deg)=',np.min(azm),np.mean(azm),np.max(azm))
        print('Question: Would you expect this to overcorrect low cloud displacements?')
        plot_abi(data,basemap=basemap,vmin=0,vmax=1,fieldtoplot='enhanced',overplot=util.plot_reports,cblabel='Visible Reflectance Factor',coords=('plat','plon'))
        if not runall: sys.exit()

    if exercise == '2e' or runall:
        print('Exercise 2e: Overplot infrared (Sandwich product)')
        print('Comment: Overshooting tops can also be identified combining texture information from visible imagery with cloud top temperature information from infrared imagery.')
        data['overplot'] = read_abi(datafiles[2],verbose=True)
        plot_abi(data,basemap=basemap,vmin=0,vmax=1,fieldtoplot='enhanced',overplot=util.sandwich,cblabel='Visible Reflectance Factor')
        if not runall: sys.exit()

#---------------------------------------------------------------

