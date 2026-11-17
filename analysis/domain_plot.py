#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:04:07 2025

@author: michael.hosek
"""

#Plot the domain
#bigbang4
#gpd_plot

import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
from scipy.signal import detrend

from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from matplotlib.path import Path
from shapely.geometry.polygon import orient

import shapely
from scipy.stats import mannwhitneyu

from scipy import stats
from scipy import signal

from metpy.calc import geopotential_to_height
from metpy.units import units

from pyproj import Geod
geod = Geod(ellps='WGS84')

import glob
import argparse

import paramiko
import os

import palettable
import random
starttime = datetime.utcnow()

# llcrnrlat=28
# llcrnrlon=360-107
# urcrnrlat=47
# urcrnrlon=360-90
# lat_0=35
# lon_0=360-98
# latin1=25
# latin2=25
# m = Basemap(projection='lcc', llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
#             urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,
#             lat_1=latin1, lat_2=latin2, lat_0=lat_0, lon_0=lon_0,
#             resolution='h', area_thresh=10000, fix_aspect=True)

llcrnrlat=22.5
llcrnrlon=360-112.5
urcrnrlat=52.5
urcrnrlon=360-72.5
lat_0=35
lon_0=360-98
latin1=25
latin2=25
m = Basemap(projection='lcc', llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
            urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,
            lat_1=latin1, lat_2=latin2, lat_0=lat_0, lon_0=lon_0,
            resolution='h', area_thresh=10000, fix_aspect=True)

#%% read in the terrain map for masking
hght_array = xr.open_dataset("/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/ERA5_terrain_high.nc") #geopotential height
eralons = hght_array.longitude.values
eralats = hght_array.latitude.values
#restrict the latitude and longitude
reslon = []
for j in range(len(eralons)):
    if eralons[j] >= 240 and eralons[j] <= 300:
        reslon.append(eralons[j])

reslat = []
for j in range(len(eralats)):
    if eralats[j] >= 20 and eralats[j] <= 55:
        reslat.append(eralats[j])


hght_array = hght_array.sel(longitude=reslon)
hght_array = hght_array.sel(latitude=reslat)

eralons = hght_array.longitude.values
eralats = hght_array.latitude.values
x,y = np.meshgrid(eralons, eralats)
x,y = m(x,y)

hght = hght_array.z.values[0] / 9.80665 #convert geopotential height to height (m)

#%%

#define SGP region
sgp = shapely.Polygon(((-105,35), (-105, 40), (-100,40), (-95, 40), (-95, 30), (-100, 30), (-105, 30)))
sgpx,sgpy = sgp.exterior.xy
sgpx,sgpy =m(sgpx,sgpy)

#define SGP buffer zone
sgp_buffer = sgp.buffer(1.0)  # buffer of 1 degree
sgp_buffer_x, sgp_buffer_y = sgp_buffer.exterior.xy
sgp_buffer_x, sgp_buffer_y = m(sgp_buffer_x, sgp_buffer_y)

#define the lee-cyclone region
# lc = shapely.Polygon(((-105,32), (-105, 45), (-98, 45), (-98, 32), (-105,32)))
# lc = shapely.Polygon(((-106,32), (-106, 41), (-109, 43), (-109, 47.5), (-106, 47.5), (-100, 47.5), (-98, 47.5), (-95, 40), (-95, 32), (-106,32)))
lc = shapely.Polygon(((-106,32), (-106, 41), (-109, 43), (-109, 47.5), (-106, 47.5), (-100, 47.5), (-100, 47.5), (-98, 41), (-98, 32), (-106,32)))
lcx, lcy = lc.exterior.xy
lcx, lcy = m(lcx, lcy)

min_lon, min_lat, max_lon, max_lat = lc.bounds
latitudes = np.arange(min_lat, max_lat + 0.25, 0.5)

rows = []

for latitude in latitudes:
    transect = shapely.LineString([
        (min_lon - 1, latitude),
        (max_lon + 1, latitude),
    ])
    intersection = lc.intersection(transect)

    if intersection.is_empty:
        continue

    west_lon, _, east_lon, _ = intersection.bounds

    sample_lons = np.linspace(west_lon, east_lon, 101)
    sample_lats = np.full_like(sample_lons, latitude)
    distance_m = geod.line_length(sample_lons, sample_lats)

    rows.append({
        "latitude": latitude,
        "west_longitude": west_lon,
        "east_longitude": east_lon,
        "width_km": distance_m / 1000,
    })

widths = pd.DataFrame(rows)

print(widths.round(2).to_string(index=False))

min_lon, min_lat, max_lon, max_lat = lc.bounds
latitudes = np.arange(min_lat, max_lat + 0.25, 0.5)

terrain_height = xr.DataArray(
    hght,
    coords={"latitude": eralats, "longitude": eralons},
    dims=("latitude", "longitude"),
)
terrain_lons = ((np.asarray(eralons) + 180) % 360) - 180

rows = []

for latitude in latitudes:
    transect = shapely.LineString([
        (min_lon - 1, latitude),
        (max_lon + 1, latitude),
    ])
    intersection = lc.intersection(transect)

    if intersection.is_empty:
        continue

    west_lon, _, east_lon, _ = intersection.bounds

    heights = terrain_height.interp(latitude=latitude).values
    masked_inside = (
        np.isfinite(heights)
        & (heights >= 1750)
        & (terrain_lons >= west_lon)
        & (terrain_lons <= east_lon)
    )

    terrain_east_lon = np.nan
    distance_km = np.nan

    if masked_inside.any():
        terrain_east_lon = terrain_lons[masked_inside].max()
        sample_lons = np.linspace(terrain_east_lon, east_lon, 101)
        sample_lats = np.full_like(sample_lons, latitude)
        distance_km = geod.line_length(sample_lons, sample_lats) / 1000

    rows.append({
        "latitude": latitude,
        "terrain_east_longitude": terrain_east_lon,
        "domain_east_longitude": east_lon,
        "distance_km": distance_km,
    })

distances = pd.DataFrame(rows)
print(distances.round(2).to_string(index=False))

print(np.nanmean(distances['distance_km']), np.nanmedian(distances['distance_km']))

#define cyclone tracking region
cy = shapely.Polygon(((-80, 25), (-80, 52.5), (-85,52.5), (-90, 52.5), (-95,52.5), (-100, 52.5), (-112.5, 52.5), (-112.5, 25), (-100, 25), (-95,25), (-90,25), (-85,25), (-80, 25)))
cyx, cyy = cy.exterior.xy
cyx, cyy = m(cyx, cyy)


#set up anomaly clevs
clevs = np.arange(-2.125, 2.25, 0.25)
clevs_cbar = np.arange(-2, 2.5, 0.5)
#set cmap
nlevs = len(clevs) - 1
cmap = plt.get_cmap(name = 'bwr', lut=nlevs)

######################################## the plot
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12, 12),frameon=True)

m.drawcoastlines(linewidth=1,color='k', ax=ax)
m.drawcountries(linewidth=1,color='k', ax=ax)
m.drawstates(linewidth=1, color='k', ax=ax)
m.drawmeridians(range(-180, 180, 5), labels = [0,0,0,1], linewidth=0.25, ax=ax, fontsize=12)
m.drawparallels(range(-10, 90, 5), labels = [1,0,0,0], linewidth=0.25, ax=ax, fontsize=12)

#shade in the terrain mask
terrain_mask = np.where(hght>=1750, 1.1, 0)
m.contourf(x,y, terrain_mask, [1,2], colors=['gray'])

#shade the buffer zone
buffer_band = orient(sgp_buffer.difference(sgp), sign=1.0)
ring_paths = []
for ring in [buffer_band.exterior, *buffer_band.interiors]:
    ring_lon, ring_lat = ring.xy
    ring_x, ring_y = m(ring_lon, ring_lat)
    ring_paths.append(Path(np.column_stack((ring_x, ring_y)), closed=True))

band_path = Path.make_compound_path(*ring_paths)
band_patch = mpatches.PathPatch(band_path, facecolor='#0072B2', edgecolor='black', linewidth=0, alpha=0.5, hatch="//", zorder=12)
ax.add_patch(band_patch)

#plot the domains
m.plot(sgpx, sgpy, color='#0072B2', linewidth=7, ax=ax, zorder=13)
m.plot(lcx, lcy, color='#D55E00', linewidth=5, ax=ax, zorder=40)
m.plot(cyx, cyy, color='#CC79A7', linewidth=4, ax=ax, zorder=50)

    
#will need to make a legend
handles = [mpatches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='#0072B2', linewidth=4, label='Southern Great Plains'),
            mpatches.Patch(facecolor="#0072B2",edgecolor='black',hatch="//", alpha=0.5, label="SGP Buffer Zone"), 
            mpatches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='#D55E00', linewidth=4, label='Lee Domain'),
            mpatches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='#CC79A7', linewidth=4, label='Cyclone Tracking'),
            mpatches.Patch(facecolor='gray', label='Terrain Mask')]
leg = ax.legend(handles=handles, title_fontsize=14, fontsize=14, loc=4, framealpha=1.0)
leg.set_zorder(50)

plt.savefig("domain.png", format='png', dpi=150, bbox_inches='tight')   
    
    
#%%
print('Time to run: ' + str(datetime.utcnow() - starttime))