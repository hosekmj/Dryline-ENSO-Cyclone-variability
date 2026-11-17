#make a 4-panel plot showing the front prediction and cyclone identification
#gpd_plot, bigbang4

import pandas as pd
from datetime import datetime, timedelta
#import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from mpl_toolkits.basemap import Basemap
#import matplotlib.colors as colors
import matplotlib
import math
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.legend_handler import HandlerBase

#from shapely.geometry import LineString
#import shapely

#import random
from scipy.ndimage import label

import xarray as xr
import metpy
from metpy.units import units
from metpy.calc import height_to_pressure_std, geopotential_to_height

import paramiko
import os

#import multiprocessing
#from itertools import repeat

#import imageio
#import glob

starttime = datetime.utcnow()

#Kim's map projection
# llcrnrlat=20
# llcrnrlon=360-123
# urcrnrlat=48
# urcrnrlon=360-60
# lat_0=35
# lon_0=360-98
# latin1=25
# latin2=25
# mlarge = Basemap(projection='lcc', llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
#             urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,
#             lat_1=latin1, lat_2=latin2, lat_0=lat_0, lon_0=lon_0,
#             resolution='h', area_thresh=10000, fix_aspect=True)

llcrnrlat=25
llcrnrlon=360-112
urcrnrlat=48
urcrnrlon=360-87
lat_0=35
lon_0=360-98
latin1=25
latin2=25
mlarge = Basemap(projection='lcc', llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
            urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,
            lat_1=latin1, lat_2=latin2, lat_0=lat_0, lon_0=lon_0,
            resolution='h', area_thresh=10000, fix_aspect=True)

llcrnrlat=26
llcrnrlon=360-105
urcrnrlat=45
urcrnrlon=360-92
lat_0=35
lon_0=360-98
latin1=25
latin2=25
msmall = Basemap(projection='lcc', llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
            urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,
            lat_1=latin1, lat_2=latin2, lat_0=lat_0, lon_0=lon_0,
            resolution='h', area_thresh=10000, fix_aspect=True)

#pick out a good day

all_drylines = pd.read_pickle('/raid/student/mhosek/era5_drylines/pkl/dryline_envi_1940_2020.pkl')
spring_drylines = all_drylines[(all_drylines['month'] >= 2) & (all_drylines['month'] <= 6)].copy()

#restrict all drylines to just the SGP (use median lat/lon to define whether it's "in" the SGP)
sgp_n = 40
sgp_s = 30
sgp_w = -105
sgp_e = -95
spring_drylines = spring_drylines[(spring_drylines['median_lon'] >= -105) & (spring_drylines['median_lon'] <= -95)]
spring_drylines = spring_drylines[(spring_drylines['median_lat'] >= 30) & (spring_drylines['median_lat'] <= 40)]

spring_drylines = spring_drylines[spring_drylines['year']==1945]
print(spring_drylines)


#possible dates
testdate = datetime(1945, 5, 14, 0)
# testdate = datetime(1945, 6, 1, 0)
# testdate = datetime(1945, 3, 24, 0)

analysis_time_str = testdate.strftime('%Y-%m-%d %H')

#read in fronts
fronts = xr.open_dataset(f'/raid/mhosek/era5_drylines/all_fronts_1940_2006.nc', engine='netcdf4')
#restrict to the test date
fronts = fronts.sel(time='1945-05-14T00:00:00')
# fronts = fronts.sel(time='1945-06-01T00:00:00')
# fronts = fronts.sel(time='1945-03-24T00:00:00')

prob_dry = fronts.prob_dry_mask.values
prob_cold = fronts.prob_cold_mask.values
prob_warm = fronts.prob_warm_mask.values
prob_occluded = fronts.prob_occluded_mask.values
prob_stationary = fronts.prob_stationary_mask.values

#read in cyclones; restrict to test date
cyclones_df = pd.read_pickle(f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/all_cyclones_df_t=1750.pkl")

#keep only specific times (00z)
def keep_time(row, testdate):
    cols = ['time', 'lat', 'lon', 'mslp', 'maxwind', 'outermost_mslp', 'size', 'in_lee_domain', 'outermost_manual_fill',
            'coords', 'approx_area', 'any_within_SGP', 'dryline_within']
    keep_positions = [position for position, time in enumerate(row['time']) if time == testdate]

    orig_len = len(row['time'])

    for col in cols:
        values = row[col]
        if len(values) != orig_len:
            raise ValueError(f"Column {col} does not match length of time column")
        row[col] = [values[i] for i in keep_positions]
    return row

cyclones_df = cyclones_df.apply(keep_time, axis=1, testdate=testdate)
#drop empty rows
cyclones_df = cyclones_df[cyclones_df['time'].apply(len) > 0]

#NOTE: MUST CHCECK/CHANGE IF CHANGING DATE
cenlat, cenlon = cyclones_df.iloc[1]['lat'][0], cyclones_df.iloc[1]['lon'][0]
outermost_mslp = cyclones_df.iloc[1]['outermost_mslp'][0]
# cenlat, cenlon = cyclones_df.iloc[0]['lat'][0], cyclones_df.iloc[0]['lon'][0]
# outermost_mslp = cyclones_df.iloc[0]['outermost_mslp'][0]


############################### read in ERA5 data
era5_surface = xr.open_dataset(f'/raid/mhosek/era5_drylines/era5_data/ERA5_1940-2025_surface.nc')
era5_plev = xr.open_dataset(f'/raid/mhosek/era5_drylines/era5_data/ERA5_upperair_1940_2020.nc')
# era5_mslp = xr.open_dataset(f'/raid/mhosek/era5_drylines/era5_data/ERA5_1940-2025_mslp.nc')

#NOTE: MUST CHANGE IF CHANGE IN DATE
era5_mslp = xr.open_dataset("/raid/mhosek/cytrack/ERA5_data/mod_uvmslp_ERA5_19450514_00.nc")
# era5_mslp = xr.open_dataset("/raid/mhosek/cytrack/ERA5_data/mod_uvmslp_ERA5_19450324_00.nc")

era5_surface = era5_surface.sel(time='1945-05-14T00:00:00')
era5_plev = era5_plev.sel(time='1945-05-14T00:00:00')
# era5_surface = era5_surface.sel(time='1945-03-24T00:00:00')
# era5_plev = era5_plev.sel(time='1945-03-24T00:00:00')
# era5_mslp = era5_mslp.sel(time='1945-05-14T00:00:00')

temp_s = era5_surface.t.values #2m temperature (K)
specific_hum_s = era5_surface.q.values #2m specific humidity (g/kg)
# u10 = era5_surface.u.values #10m wind (m/s)
# v10 = era5_surface.v.values #10m wind (m/s)

mslp = era5_mslp.msl.values[0] / 100 #Pa -> hPa  #mean sea level pressure (hPa)
u10 = era5_mslp.u10.values[0] * 1.943 #10m wind (kts)
v10 = era5_mslp.v10.values[0] * 1.943 #10m wind (kts)

z500 = era5_plev['GH'].sel(pressure_level=500).values #500 hPa geopotential height (gpm)
u500, v500 = era5_plev['u'].sel(pressure_level=500).values, era5_plev['v'].sel(pressure_level=500).values

surf_lat, surf_lon = era5_surface.latitude.values, era5_surface.longitude.values
z500_lat, z500_lon = era5_plev.lat.values, era5_plev.lon.values
mslp_lat, mslp_lon = era5_mslp.latitude.values, era5_mslp.longitude.values


#lat/lon
grid_lat = fronts.lat.values 
grid_lon = fronts.lon.values #lons are poasitive

surf_x, surf_y = np.meshgrid(surf_lon, surf_lat)
surf_xsmall, surf_ysmall = msmall(surf_x, surf_y)
surf_xlarge, surf_ylarge = mlarge(surf_x, surf_y)

mslp_x, mslp_y = np.meshgrid(mslp_lon, mslp_lat)
mslp_xsmall, mslp_ysmall = msmall(mslp_x, mslp_y)
mslp_xlarge, mslp_ylarge = mlarge(mslp_x, mslp_y)

z500_x, z500_y = np.meshgrid(z500_lon, z500_lat)
z500_x, z500_y = mlarge(z500_x, z500_y)

x,y = np.meshgrid(grid_lon, grid_lat)
xsmall,ysmall = msmall(x,y)
xlarge, ylarge = mlarge(x, y)

############################################## 4-panel plot, each with cyclone and front prediction
#panel 1: MSLP
#panel 2: specific humidity
#panel 3: surface temperature
#panel 4: 500mb height and wind

# Find the grid point nearest the cyclone centroid.
ilat = np.argmin(np.abs(mslp_lat - cenlat))
lon_distance = np.abs((mslp_lon - cenlon + 180) % 360 - 180)
ilon = np.argmin(lon_distance)

# Label disconnected regions below the outermost isobar.
below_isobar = np.isfinite(mslp) & (mslp <= outermost_mslp)
region_labels, _ = label(below_isobar)

cyclone_label = region_labels[ilat, ilon]
if cyclone_label == 0:
    raise ValueError(
        "The grid point nearest the cyclone centroid is not within "
        "the outermost MSLP threshold."
    )

mslp_cyclone = (region_labels == cyclone_label).astype(float)

fig,((ax1,ax2), (ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(24, 24),frameon=True)
axes = [ax1,ax2,ax3,ax4]
panellabels = ['a)', 'b)', 'c)', 'd)']
for i, ax in enumerate(axes):
    if ax != ax1 and ax != ax2:
        m=msmall
        x=xsmall
        y=ysmall
        surf_x = surf_xsmall
        surf_y = surf_ysmall
        mslp_x = mslp_xsmall
        mslp_y = mslp_ysmall
    else:
        m=mlarge
        x=xlarge
        y=ylarge
        surf_x = surf_xlarge
        surf_y = surf_ylarge
        mslp_x = mslp_xlarge
        mslp_y = mslp_ylarge

    m.drawcoastlines(linewidth=0.5,color='k', ax=ax)
    m.drawcountries(linewidth=0.5,color='k', ax=ax)
    m.drawstates(linewidth=0.25, color='k', ax=ax)
    m.drawmeridians(range(-180, 180, 5), labels = [0,0,0,1], linewidth=0.25, ax=ax, fontsize=14)
    if ax == ax3:
        m.drawparallels(range(-10, 90, 5), labels = [0,1,0,0], linewidth=0.25, ax=ax, fontsize=14)
    else:
        m.drawparallels(range(-10, 90, 5), labels = [1,0,0,0], linewidth=0.25, ax=ax, fontsize=14)

    if ax == ax1:
        ax.barbs(mslp_x[::5, ::5],mslp_y[::5, ::5],u10[::5, ::5],v10[::5, ::5], length=5, barbcolor='dimgrey', zorder=10)
    elif ax == ax2:
        ax.barbs(z500_x[::5, ::5],z500_y[::5, ::5],u500[::5, ::5],v500[::5, ::5], length=5, barbcolor='dimgrey', zorder=10)
    else:
        ax.barbs(mslp_x[::4, ::4],mslp_y[::4, ::4],u10[::4, ::4],v10[::4, ::4], length=6, barbcolor='dimgrey', zorder=10)



    if ax == ax1 or ax == ax2:
        ax.text(0.02, 1.055, panellabels[i], transform=ax.transAxes, fontsize=28, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=3), zorder=500)
        ###########Plot all of the front probabilities    
        m.contourf(x,y,prob_dry.T, [0.4, 1.0], colors="sienna", extend='max', ax=ax)
        dry_plot = m.contour(x,y,prob_dry.T, levels=[0.4], colors='saddlebrown', linewidths=1.5, ax=ax)
        
        
        cold_plot = m.contourf(x,y,prob_cold.T, [0.5,1.0], colors='royalblue', ax=ax)
        cold_plot = m.contour(x,y,prob_cold.T, [0.5], colors='navy', linewidths=0.75, ax=ax, label='Cold Front')
        
        warm_plot = m.contourf(x,y,prob_warm.T, [0.5,1.0], colors='indianred', ax=ax)
        warm_plot = m.contour(x,y,prob_warm.T, [0.5], colors='red', linewidths=0.75, ax=ax, label='Warm Front')
        
        occ_plot = m.contourf(x,y,prob_occluded.T, [0.5,1.0], colors='plum', ax=ax)
        occ_plot = m.contour(x,y,prob_occluded.T, [0.5], colors='purple', linewidths=0.75, ax=ax, label='Occluded Front')
        
        stat_plot = m.contourf(x,y,prob_stationary.T, [0.5,1.0], colors='mediumseagreen', ax=ax)
        stat_plot = m.contour(x,y,prob_stationary.T, [0.5], colors='darkgreen', linewidths=0.75, ax=ax, label='Stationary Front')
        

        ############# plot the cyclone
        cyx, cyy = m(cenlon, cenlat)
        ax.plot(cyx, cyy, 'o', color='black', markersize=28, label='Cyclone Centroid')

        #outline the fill the area within the outermost closed isobar
        mslp_plot = m.contour(mslp_x, mslp_y, mslp_cyclone, levels=[0.5, 1], colors='cyan', linewidths=3, ax=ax)
        mslp_plot = m.contourf(mslp_x, mslp_y, mslp_cyclone, levels=[0.5, 1], colors='paleturquoise', ax=ax, zorder=0)

        # #set up legend
        # custom_lines = [Line2D([0], [0], color='peru', lw=10), Line2D([0], [0], color='lightcyan', lw=10), Line2D([0], [0], color='cornflowerblue', lw=10), Line2D([0], [0], color='lightcoral', lw=10), Line2D([0], [0], color='plum', lw=10), Line2D([0], [0], color='lightgreen', lw=10)]
        # legend = ax.legend(custom_lines, ['Dryline', 'SGP Cyclone', 'Cold Front', 'Warm Front', "Occluded Front", "Stationary Front"], framealpha=1, loc=3)
        # legend.set_zorder(11)
    
    else:
        ax.text(0.02, 1.055, panellabels[i], transform=ax.transAxes, fontsize=28, fontweight='bold', va='top', bbox=dict(facecolor='white', edgecolor='black', pad=3), zorder=500)
        dry_plot = m.contour(x,y,prob_dry.T, levels=[0.4], colors='saddlebrown', linewidths=5, ax=ax)

        ############# plot the cyclone
        cyx, cyy = m(cenlon, cenlat)
        ax.plot(cyx, cyy, 'o', color='black', markersize=28, label='Cyclone Centroid')

        #outline the fill the area within the outermost closed isobar
        mslp_plot = m.contour(mslp_x, mslp_y, mslp_cyclone, levels=[0.5, 1], colors='cyan', linewidths=5, ax=ax)


        # #set up legend
        # custom_lines = [Line2D([0], [0], color='saddlebrown', lw=2), Line2D([0], [0], color='cyan', lw=2)]
        # legend = ax.legend(custom_lines, ['Dryline', 'SGP Cyclone'], framealpha=1, loc=3)
        # legend.set_zorder(11)


#panel 1: MSLP
mslp_clev = np.arange(960, 1040, 4)
mslp_plot = mlarge.contour(mslp_xlarge, mslp_ylarge, mslp, levels=mslp_clev, colors='black', linewidths=1.5, ax=ax1)
ax1.clabel(mslp_plot, fontsize=12)

#panel 2: plot hte 500mb height
z500_clev = np.arange(4800, 6000, 60)
z500_plot = mlarge.contour(z500_x, z500_y, z500, levels=z500_clev, colors='black', linewidths=1.5, ax=ax2)
ax2.clabel(z500_plot, fontsize=12)

#panel 3: plot specific humidity
qv_clev = np.arange(0, 24, 2)
nlevs=len(qv_clev)
cmap_name='YlGn'
cmap = plt.get_cmap(name=cmap_name, lut=nlevs)
qv_plot = msmall.contourf(surf_xsmall, surf_ysmall, specific_hum_s, levels=qv_clev, cmap=cmap, ax=ax3)
cbar = plt.colorbar(qv_plot, ax=ax3, orientation='vertical', location='left', pad=0.02)
cbar.set_label(r"Specific Humidity ($g kg^{-1}$)", fontsize=18)
cbar.ax.tick_params(labelsize=14)


#panel 4: plot temperature
t_clev = np.arange(270, 320, 0.5)
nlevs=len(t_clev)
cmap_name='RdYlBu_r'
cmap = plt.get_cmap(name=cmap_name, lut=nlevs)
t_plot = msmall.contourf(surf_xsmall, surf_ysmall, temp_s, levels=t_clev, cmap=cmap, ax=ax4)
cbar = plt.colorbar(t_plot, ax=ax4, orientation='vertical', pad=0.02)
cbar.set_label("Temperature (K)", fontsize=18)
cbar.ax.tick_params(labelsize=14)

# Center a shared legend in the gap between the lower panels
p3 = ax3.get_position()
p4 = ax4.get_position()
gap_x = (p3.x1 + p4.x0) / 2
gap_y = (p3.y0 + p3.y1) / 2

class HandlerPatchLine(HandlerBase):
    def create_artists(
        self, legend, handle, xdescent, ydescent, width, height, fontsize, transform
    ):
        patch_proto, line_proto = handle

        patch = mpatches.Rectangle(
            (xdescent, ydescent),
            width,
            height,
            facecolor=patch_proto.get_facecolor(),
            edgecolor="none",
            transform=transform,
        )

        extension = width * 0.2  # Extend 20% of the handle width on each side
        y = ydescent + height / 2
        line = Line2D(
            [xdescent - extension, xdescent + width + extension],
            [y, y],
            color=line_proto.get_color(),
            linewidth=line_proto.get_linewidth(),
            transform=transform,
        )

        return [patch, line]

dryline_handle = (mpatches.Patch(facecolor="sienna", edgecolor="saddlebrown"),
    Line2D([0], [0], color="saddlebrown", lw=2))
cyclone_handle = (mpatches.Patch(facecolor="paleturquoise", edgecolor="cyan", label="SGP Cyclone"),
    Line2D([0], [0], color="cyan", lw=2))
centroid_handle = Line2D(
    [], [],
    marker="o",
    linestyle="None",
    markerfacecolor="black",
    markeredgecolor="black",
    markersize=18,
)

legend_handles = [dryline_handle, cyclone_handle, centroid_handle,
    mpatches.Patch(facecolor="royalblue", edgecolor="navy", label="Cold Front"),
    mpatches.Patch(facecolor="indianred", edgecolor="red", label="Warm Front"),
    mpatches.Patch(facecolor="plum", edgecolor="purple", label="Occluded Front"),
    mpatches.Patch(facecolor="mediumseagreen", edgecolor="darkgreen", label="Stationary Front"),
]


fig.legend(
    legend_handles,
    ["Dryline", "SGP Cyclone", "Cyclone Centroid", "Cold Front", "Warm Front", "Occluded Front", "Stationary Front"],
    loc="center",
    handler_map={tuple: HandlerPatchLine()},
    fontsize=25,
    bbox_to_anchor=(gap_x, gap_y),
    bbox_transform=fig.transFigure,
    framealpha=1,
)


#draw in the bounds of the smaller domain
# Small-map boundary in its projected coordinates.
xs = np.array([
    msmall.xmin, msmall.xmax, msmall.xmax,
    msmall.xmin, msmall.xmin,
])
ys = np.array([
    msmall.ymin, msmall.ymin, msmall.ymax,
    msmall.ymax, msmall.ymin,
])

# Convert small-map coordinates -> lon/lat -> large-map coordinates.
lons, lats = msmall(xs, ys, inverse=True)
xb, yb = mlarge(lons, lats)

for ax in (ax1, ax2):
    ax.plot(
        xb, yb,
        color="black",
        linestyle="--",
        linewidth=2,
        zorder=20,
    )


ax1.set_title("MSLP", fontsize=28)
ax2.set_title("500mb Height", fontsize=28)
ax3.set_title("Specific Humidity", fontsize=28)
ax4.set_title("Temperature", fontsize=28)
plt.suptitle(f"{analysis_time_str} UTC", fontsize=34, y=0.92)

plt.subplots_adjust(hspace=0.1, wspace=0.0)
plt.savefig(f'example_4panel.png', format='png', dpi=150, bbox_inches='tight')


print('Time to run: ' + str(datetime.utcnow() - starttime))