#Determine the ENSO phase for detrended SSTa using operational threshold (and indiv month)
#gpd_plot, bigbang4

import pandas as pd
from datetime import datetime, timedelta
#import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib.lines import Line2D
#import matplotlib.colors as colors
import matplotlib
from scipy import stats
from scipy.signal import detrend
import matplotlib.patheffects as pe

#from shapely.geometry import LineString
#import shapely

#import random

import xarray as xr

#import multiprocessing
#from itertools import repeat

#import imageio
#import glob


from netCDF4 import Dataset
starttime = datetime.utcnow()

#enso_xr = xr.open_dataset("/raid/mhosek/era5_drylines/ENSO_1940_2022.nc") #this is area-average SSTa of 3.4 region, monthly
enso_xr = xr.open_dataset("/home/michael.hosek/Documents/ERA5_drylines/enso.nc") #this is area-average SSTa of 3.4 region, monthly

#restrict to 1939-2020
enso_xr = enso_xr.sel(time=slice("1939-01-01", "2021-12-31"))


#since it's already SSTa, don't need to remove the seasonal cycle (anomalies are calculated relative to 1981-2010 climatology)
#Just need to remove the linear trend over time -- across all months together (no need to do individual)

ssta = enso_xr['value']
fit = ssta.polyfit(dim='time', deg=1)
trend = xr.polyval(ssta['time'], fit.polyfit_coefficients)

ssta_detrended = ssta - trend + ssta.mean(dim="time")
enso_xr['ssta_detrended'] = ssta_detrended


# #sanity plot
# fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,12))
# ax.plot(enso_xr.time, enso_xr['ENSO_index'].values, label='SSTa', color='red')
# ax.plot(enso_xr.time, enso_xr['ENSO_index_detrended'].values, label='SSTa Detrended', color='black')
# ax.set_xlabel('Time')
# ax.set_ylabel('SSTa (°C)')
# ax.set_title('ENSO Region 3.4 SSTa')
# ax.legend()
# plt.savefig("enso_sanity_check.png", dpi=100, bbox_inches='tight')


################### calculate operational ENSO phase
# 3-month index average is +/-0.5 for 5 consecutive months

#calculate the centered 3 month average
oni_detrended = ssta_detrended.rolling(time=3, min_periods=3, center=True).mean()
enso_xr['oni_detrended'] = oni_detrended

time = enso_xr['time'].values

enso = []
with open("enso_phase.txt", "a") as f:
    for t in range(len(time)):
        if pd.to_datetime(time[t]) >= datetime(1940, 1, 1) and pd.to_datetime(time[t]) <= datetime(2020, 12, 31):

            #to be an ENSO phase, you need to be part of a 5-consecutive-month stretch with ONI exceeding threshold

            #select the period
            oni_9month = oni_detrended.sel(time=slice(time[t-4], time[t+4]))
            #determine the ENSO phase based on the operational threshold
            e = any((oni_9month.isel(time=slice(i, i+5)) >= 0.5).all().item() for i in range(5))
            l = any((oni_9month.isel(time=slice(i, i+5)) <= -0.5).all().item() for i in range(5))

            if e:
                enso.append(1) #El Nino
            elif l:
                enso.append(-1) #La Nina
            else:
                enso.append(0) #Neutral

            #write out to txt file
            f.write(f"{pd.to_datetime(time[t]).strftime('%Y-%m')}: {oni_detrended.values[t]} {oni_9month.values} {enso[-1]}\n")


enso_xr = enso_xr.sel(time=slice("1940-01-01", "2020-12-31"))
time = enso_xr['time'].values


enso_xr['ENSO_phase'] = ('time', enso)

# print(enso_xr)

#print out how many March are in each phase
march_phases = enso_xr['ENSO_phase'].sel(time=enso_xr['time.month'] == 2)
print('Number of February in each phase:')
print('El Nino:', (march_phases == 1).sum().item())
print('La Nina:', (march_phases == -1).sum().item())
print('Neutral:', (march_phases == 0).sum().item())

march_phases = enso_xr['ENSO_phase'].sel(time=enso_xr['time.month'] == 3)
print('Number of March in each phase:')
print('El Nino:', (march_phases == 1).sum().item())
print('La Nina:', (march_phases == -1).sum().item())
print('Neutral:', (march_phases == 0).sum().item())

march_phases = enso_xr['ENSO_phase'].sel(time=enso_xr['time.month'] == 4)
print('Number of April in each phase:')
print('El Nino:', (march_phases == 1).sum().item())
print('La Nina:', (march_phases == -1).sum().item())
print('Neutral:', (march_phases == 0).sum().item())

march_phases = enso_xr['ENSO_phase'].sel(time=enso_xr['time.month'] == 5)
print('Number of May in each phase:')
print('El Nino:', (march_phases == 1).sum().item())
print('La Nina:', (march_phases == -1).sum().item())
print('Neutral:', (march_phases == 0).sum().item())

march_phases = enso_xr['ENSO_phase'].sel(time=enso_xr['time.month'] == 6)
print('Number of June in each phase:')
print('El Nino:', (march_phases == 1).sum().item())
print('La Nina:', (march_phases == -1).sum().item())
print('Neutral:', (march_phases == 0).sum().item())

#sanity plot
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,12))
ax.plot(enso_xr.time, enso_xr['ssta_detrended'].values, label='SSTa Detrended', color='black', alpha=0.75)
ax.plot(enso_xr.time, enso_xr['oni_detrended'].values, label='ONI Detrended', color='black', linestyle='--', alpha=0.75)

ymin, ymax = ax.get_ylim()
ax.fill_between(time, ymin, ymax, where=np.array(enso) == 1, facecolor='red', alpha=0.3, label='El Nino')
ax.fill_between(time, ymin, ymax, where=np.array(enso) == -1, facecolor='blue', alpha=0.3, label='La Nina')
ax.set_ylim(ymin, ymax)

ax.set_xlabel('Time')
ax.set_ylabel('SSTa (°C)')
ax.set_title('ENSO Region 3.4 SSTa')
ax.legend()
plt.savefig("enso_sanity_check.png", dpi=100, bbox_inches='tight')


#TO DO: SAVE OUT THE FILE
enso_xr = enso_xr.rename_vars({"value": "SSTa"})
enso_xr = enso_xr.rename_vars({"ssta_detrended": "SSTa_detrended"})
enso_xr.attrs['description'] = 'ENSO Region 3.4 SSTa and ONI detrended values with ENSO phase classification from PSL HadISST1.1 (https://psl.noaa.gov/data/timeseries/month/DS/Nino34/)'
enso_xr.attrs['SSTa'] = "deg C; monthly area-average sea surface temperature anomaly relative to 1981-2010 climatology"
enso_xr.attrs['SSTa_detrended'] = "deg C; linearly detrended SSTa across years"
enso_xr.attrs['oni_detrended'] = "deg C; centered 3-month avg detrended SSTa"
enso_xr.attrs['ENSO_phase'] = "ENSO phase classification (5-consecutive months +/-0.5 ONI): 1 = El Nino, -1 = La Nina, 0 = Neutral"

enso_xr.to_netcdf("/raid/mhosek/era5_drylines/ENSO_1940_2020_ONI.nc")

print('Time to run: ' + str(datetime.utcnow() - starttime))