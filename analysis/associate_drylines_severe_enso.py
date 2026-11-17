
#As in associate_dryline_severe.py, but for all drylines and also separate out ENSO phases
# Take 80km gridded reports and warning, along with 80km gridded drylines
# apply search box to all drylines and determine which SGP warnings/reports are associated with drylines and which aren't
#run in bigbang4
#use gpd_plot

import geopandas
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
from scipy.signal import detrend

import scipy.spatial

from metpy.calc import geopotential_to_height
from metpy.units import units

import glob
import argparse

import paramiko
import os

import palettable

start = datetime.utcnow()

# llcrnrlat=25
# llcrnrlon=360-107
# urcrnrlat=42
# urcrnrlon=360-93
# lat_0=35
# lon_0=360-98
# latin1=25
# latin2=25
# m = Basemap(projection='lcc', llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
#             urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,
#             lat_1=latin1, lat_2=latin2, lat_0=lat_0, lon_0=lon_0,
#             resolution='h', area_thresh=10000, fix_aspect=True)

#Kim's map projection
llcrnrlat=20
llcrnrlon=360-123
urcrnrlat=48
urcrnrlon=360-60
lat_0=35
lon_0=360-98
latin1=25
latin2=25
m = Basemap(projection='lcc', llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
            urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,
            lat_1=latin1, lat_2=latin2, lat_0=lat_0, lon_0=lon_0,
            resolution='h', area_thresh=10000, fix_aspect=True)

#%% read in data


#read in reports, warnings, and drylines
reports = xr.load_dataset('/home/michael.hosek/Documents/ERA5_drylines/gridded_6hrly_severe_80km_1995_2020.nc') #I think xr.load will be better here than open
warnings = xr.load_dataset('/home/michael.hosek/Documents/ERA5_drylines/gridded_6hrly_warnings_80km_1995_2020.nc')
drylines = xr.open_dataset('/raid/student/mhosek/era5_drylines/gridded_drylines_80km_1995_2020.nc')


# restrict to spring (Feb -- May); restrict warnings/reports to 21z (21z-03z)
reptimes = reports.time.values
keeptimes = []
for i in range(len(reptimes)):
    if pd.to_datetime(reptimes[i]).month >= 2 and pd.to_datetime(reptimes[i]).month <= 5 and pd.to_datetime(reptimes[i]).hour == 21:
        keeptimes.append(reptimes[i])
reports = reports.sel(time=keeptimes)

warntimes = warnings.time.values
keeptimes = []
for i in range(len(warntimes)):
    if pd.to_datetime(warntimes[i]).month >= 2 and pd.to_datetime(warntimes[i]).month <= 5 and pd.to_datetime(warntimes[i]).hour == 21:
        keeptimes.append(warntimes[i])
warnings = warnings.sel(time=keeptimes)

#Note: just like with warnings and reports, will have blindspot for Jan 31/Feb 1 and June 30/July 1
drytimes = drylines.time.values
keeptimes = []
for i in range(len(drytimes)):
    if pd.to_datetime(drytimes[i]).month >= 2 and pd.to_datetime(drytimes[i]).month <= 5:
        keeptimes.append(drytimes[i])
drylines = drylines.sel(time=keeptimes)
dryline_times = drylines.time.values

lons, lats = drylines.lons.values, drylines.lats.values

#read in monthly enso values
enso_xr = xr.open_dataset("/raid/student/mhosek/era5_drylines/ENSO_1940_2022.nc")
times = enso_xr.time.values
keeptimes = []
for i in range(len(times)):
    if pd.to_datetime(times[i]).month >= 2 and pd.to_datetime(times[i]).month <= 5:
        keeptimes.append(times[i])
        
enso_xr = enso_xr.sel(time=keeptimes)

#put in dataframe for easier searching
enso = pd.DataFrame()

times = enso_xr.time.values
year = []
month = []
for i in range(len(times)):
    year.append(pd.to_datetime(times[i]).year)
    month.append(pd.to_datetime(times[i]).month)

enso['time'] = enso_xr.time.values
enso['year'] = year
enso['month'] = month
enso['index'] = enso_xr.ENSO_index.values


print("read in all data")



#%%

#so, we can use the dates of every dryline (from spring_drylines) to look through the gridded drylines, warnings, and reports to associate reports/warnings with drylines
#Associatae over 6 hour window (21z -- 03z)
#search box:
    #for WPC drylines, was 150km east, 50km west, 25km north/south
    #for this: 240km east (3 grids, to account for storm motion over 3 hours), 80km west (1 grids, to account for 3 hours prior), 80km north/south

#ultimately, want to return an array of dryline associated severe for each report/warn category
#in a separate script, can then comapre the dryline-associated severe to the total severe in the reports/warnings datasets read in here

#output netcdf:
    #time x lat x lon (time is 00z only --> all dryline times)
    #at each time, it's the count by grid box of how many of that type of report/warning is associated with a dryline

#%%

# Create searchbox 
searchbox = np.zeros(drylines.dryline.values.shape)
dry = drylines.dryline.values


#the original searchbox
#For every dryline point, searchbox = 1 at that point, as well as 3 grids east, 1 grids west, 1 grid north and south
for t in range(len(dryline_times)):
    for i in range(len(dry[t])): #lats
        for j in range(len(dry[t][i])): #lons
            if dry[t][i][j] == 1:
                searchbox[t][i][j] = 1
                
                #north 1 grid
                if i != len(lats):
                    searchbox[t][i+1][j] = 1
                    
                    #west 1 grids = 1
                    if j != 0:
                        searchbox[t][i+1][j-1] = 1
                    
                    #east 3 grids = 1
                    if j != len(lons):
                        searchbox[t][i+1][j+1] = 1
                    if j < len(lons) - 1:
                        searchbox[t][i+1][j+2] = 1
                    if j < len(lons) - 2:
                        searchbox[t][i+1][j+3] = 1
                
                #south 1 grid
                if i != 0:
                    searchbox[t][i-1][j] = 1
                    
                    #west 1 grids = 1
                    if j != 0:
                        searchbox[t][i-1][j-1] = 1
                    
                    #east 3 grids = 1
                    if j != len(lons):
                        searchbox[t][i-1][j+1] = 1
                    if j < len(lons) - 1:
                        searchbox[t][i-1][j+2] = 1
                    if j < len(lons) - 2:
                        searchbox[t][i-1][j+3] = 1
                
                #west 1 grids = 1
                if j != 0:
                    searchbox[t][i][j-1] = 1
                
                #east 3 grids = 1
                if j != len(lons):
                    searchbox[t][i][j+1] = 1
                if j < len(lons) - 1:
                    searchbox[t][i][j+2] = 1
                
                if j < len(lons) - 2:
                    searchbox[t][i][j+3] = 1

'''      
#TEST A SMALLER SEARCHBOX
print("running a smaller searchbox")        
#For every dryline point, searchbox = 1 at that point, as well as 2 grids east, 1 grids west, 1 grid north and south
for t in range(len(dryline_times)):
    for i in range(len(dry[t])): #lats
        for j in range(len(dry[t][i])): #lons
            if dry[t][i][j] == 1:
                searchbox[t][i][j] = 1
                
                #north 1 grid
                if i != len(lats):
                    searchbox[t][i+1][j] = 1
                    
                    #west 1 grids = 1
                    if j != 0:
                        searchbox[t][i+1][j-1] = 1
                    
                    #east 2 grids = 1
                    if j != len(lons):
                        searchbox[t][i+1][j+1] = 1
                    if j < len(lons) - 1:
                        searchbox[t][i+1][j+2] = 1
                
                #south 1 grid
                if i != 0:
                    searchbox[t][i-1][j] = 1
                    
                    #west 1 grids = 1
                    if j != 0:
                        searchbox[t][i-1][j-1] = 1
                    
                    #east 2 grids = 1
                    if j != len(lons):
                        searchbox[t][i-1][j+1] = 1
                    if j < len(lons) - 1:
                        searchbox[t][i-1][j+2] = 1
                
                #west 1 grids = 1
                if j != 0:
                    searchbox[t][i][j-1] = 1
                
                #east 2 grids = 1
                if j != len(lons):
                    searchbox[t][i][j+1] = 1
                if j < len(lons) - 1:
                    searchbox[t][i][j+2] = 1
'''                
print("all searchboxes created")
#%% Go through report and warn xarray's and see where reports/warnings overlap with searchbox


################################### ALL OF THE BELOW NEEDS TO BE UPDATED

dryline_associated_torwarn_el = np.zeros(dry.shape)
dryline_associated_svrwarn_el = np.zeros(dry.shape)
dryline_associated_tor_el = np.zeros(dry.shape)
dryline_associated_f1tor_el = np.zeros(dry.shape)
dryline_associated_sigtor_el = np.zeros(dry.shape)
dryline_associated_hail_el = np.zeros(dry.shape)
dryline_associated_sighail_el = np.zeros(dry.shape)
dryline_associated_wind_el = np.zeros(dry.shape)
dryline_associated_windm_el = np.zeros(dry.shape)
dryline_associated_sigwind_el = np.zeros(dry.shape)
dryline_associated_allsvr_el = np.zeros(dry.shape)
dryline_associated_allreports_el = np.zeros(dry.shape)
dryline_associated_allwarn_el = np.zeros(dry.shape)
dryline_associated_alltor_el = np.zeros(dry.shape)
dryline_associated_allsig_el = np.zeros(dry.shape)

dryline_associated_torwarn_la = np.zeros(dry.shape)
dryline_associated_svrwarn_la = np.zeros(dry.shape)
dryline_associated_tor_la = np.zeros(dry.shape)
dryline_associated_f1tor_la = np.zeros(dry.shape)
dryline_associated_sigtor_la = np.zeros(dry.shape)
dryline_associated_hail_la = np.zeros(dry.shape)
dryline_associated_sighail_la = np.zeros(dry.shape)
dryline_associated_wind_la = np.zeros(dry.shape)
dryline_associated_windm_la = np.zeros(dry.shape)
dryline_associated_sigwind_la = np.zeros(dry.shape)
dryline_associated_allsvr_la = np.zeros(dry.shape)
dryline_associated_allreports_la = np.zeros(dry.shape)
dryline_associated_allwarn_la = np.zeros(dry.shape)
dryline_associated_alltor_la = np.zeros(dry.shape)
dryline_associated_allsig_la = np.zeros(dry.shape)

dryline_associated_torwarn_n = np.zeros(dry.shape)
dryline_associated_svrwarn_n = np.zeros(dry.shape)
dryline_associated_tor_n = np.zeros(dry.shape)
dryline_associated_f1tor_n = np.zeros(dry.shape)
dryline_associated_sigtor_n = np.zeros(dry.shape)
dryline_associated_hail_n = np.zeros(dry.shape)
dryline_associated_sighail_n = np.zeros(dry.shape)
dryline_associated_wind_n = np.zeros(dry.shape)
dryline_associated_windm_n = np.zeros(dry.shape)
dryline_associated_sigwind_n = np.zeros(dry.shape)
dryline_associated_allsvr_n = np.zeros(dry.shape)
dryline_associated_allreports_n = np.zeros(dry.shape)
dryline_associated_allwarn_n = np.zeros(dry.shape)
dryline_associated_alltor_n = np.zeros(dry.shape)
dryline_associated_allsig_n = np.zeros(dry.shape)

for t in range(len(dryline_times)):
    #define the search times as 3 hours prior to the dryline time (so, for a 00z dryline time, the search time is 21z the "previous day")
    search_time = dryline_times[t] - np.timedelta64(3, 'h') #reports/warn for 21z-03z
    
    if pd.to_datetime(search_time).month != 1: #make sure we don't include any January drylines
    
        #warnings first
        warn_xary = warnings.sel(time=search_time)
        
        #all of these have a nanmax of 1
        tor_warn = warn_xary.tor_warn.values
        svr_warn = warn_xary.svr_warn.values
        all_warn = warn_xary.all_warn.values
        
        #reports
        reports_xary = reports.sel(time=search_time)
        
        #all of these from xary are 1's and 0's
        tor = reports_xary.tornado.values
        f1tor = reports_xary['f1+_tornado'].values
        sigtor = reports_xary.sig_tornado.values
        
        hail = reports_xary.hail.values
        sighail = reports_xary.sig_hail.values
        
        wind = reports_xary.wind.values
        wind_m = reports_xary.wind_measured.values
        sigwind = reports_xary.sig_wind.values
        
        all_reports = reports_xary.all_reports.values
        
       
        #create an "all severe" of warning OR report
        all_severe = all_reports + all_warn
        all_severe = np.where(all_severe > 1, 1, all_severe)
    
        #create an "all tornado" warning OR report
        all_tor = tor_warn + tor
        all_tor = np.where(all_tor > 1, 1, all_tor)
        
        #create an "all sig" report
        all_sig = sigtor + sighail + sigwind
        all_sig = np.where(all_sig > 1, 1, all_sig)

        #determine ENSO phase at this time
        month, year = pd.to_datetime(search_time).month, pd.to_datetime(search_time).year
        indiv_enso = enso[(enso['month']==month) & (enso['year']==year)]
        index = indiv_enso.iloc[0]['index']

        #Associate warnings and reports with drylines, by enso phase
        if index <= -0.5:
            dryline_associated_torwarn_la[t] = np.where(searchbox[t] == 1, tor_warn, dryline_associated_torwarn_la[t])
            dryline_associated_svrwarn_la[t] = np.where(searchbox[t] == 1, svr_warn, dryline_associated_svrwarn_la[t])  
            dryline_associated_allwarn_la[t] = np.where(searchbox[t] == 1, all_warn, dryline_associated_allwarn_la[t]) 
        
            dryline_associated_tor_la[t] = np.where(searchbox[t] == 1, tor, dryline_associated_tor_la[t])
            dryline_associated_f1tor_la[t] = np.where(searchbox[t] == 1, f1tor, dryline_associated_f1tor_la[t])
            dryline_associated_sigtor_la[t] = np.where(searchbox[t] == 1, sigtor, dryline_associated_sigtor_la[t])
            
            dryline_associated_hail_la[t] = np.where(searchbox[t] == 1, hail, dryline_associated_hail_la[t])
            dryline_associated_sighail_la[t] = np.where(searchbox[t] == 1, sighail, dryline_associated_sighail_la[t])
            
            dryline_associated_wind_la[t] = np.where(searchbox[t] == 1, wind, dryline_associated_wind_la[t])
            dryline_associated_windm_la[t] = np.where(searchbox[t] == 1, wind_m, dryline_associated_windm_la[t])
            dryline_associated_sigwind_la[t] = np.where(searchbox[t] == 1, sigwind, dryline_associated_sigwind_la[t])
            
            dryline_associated_allreports_la[t] = np.where(searchbox[t] == 1, all_reports, dryline_associated_allreports_la[t])
            dryline_associated_allsvr_la[t] = np.where(searchbox[t] == 1, all_severe, dryline_associated_allsvr_la[t])
            dryline_associated_alltor_la[t] = np.where(searchbox[t] == 1, all_tor, dryline_associated_alltor_la[t])
            
            dryline_associated_allsig_la[t] = np.where(searchbox[t] == 1, all_sig, dryline_associated_allsig_la[t])
        elif index >= 0.5:
            dryline_associated_torwarn_el[t] = np.where(searchbox[t] == 1, tor_warn, dryline_associated_torwarn_el[t])
            dryline_associated_svrwarn_el[t] = np.where(searchbox[t] == 1, svr_warn, dryline_associated_svrwarn_el[t])  
            dryline_associated_allwarn_el[t] = np.where(searchbox[t] == 1, all_warn, dryline_associated_allwarn_el[t]) 
        
            dryline_associated_tor_el[t] = np.where(searchbox[t] == 1, tor, dryline_associated_tor_el[t])
            dryline_associated_f1tor_el[t] = np.where(searchbox[t] == 1, f1tor, dryline_associated_f1tor_el[t])
            dryline_associated_sigtor_el[t] = np.where(searchbox[t] == 1, sigtor, dryline_associated_sigtor_el[t])
            
            dryline_associated_hail_el[t] = np.where(searchbox[t] == 1, hail, dryline_associated_hail_el[t])
            dryline_associated_sighail_el[t] = np.where(searchbox[t] == 1, sighail, dryline_associated_sighail_el[t])
            
            dryline_associated_wind_el[t] = np.where(searchbox[t] == 1, wind, dryline_associated_wind_el[t])
            dryline_associated_windm_el[t] = np.where(searchbox[t] == 1, wind_m, dryline_associated_windm_el[t])
            dryline_associated_sigwind_el[t] = np.where(searchbox[t] == 1, sigwind, dryline_associated_sigwind_el[t])
            
            dryline_associated_allreports_el[t] = np.where(searchbox[t] == 1, all_reports, dryline_associated_allreports_el[t])
            dryline_associated_allsvr_el[t] = np.where(searchbox[t] == 1, all_severe, dryline_associated_allsvr_el[t])
            dryline_associated_alltor_el[t] = np.where(searchbox[t] == 1, all_tor, dryline_associated_alltor_el[t])
            
            dryline_associated_allsig_el[t] = np.where(searchbox[t] == 1, all_sig, dryline_associated_allsig_el[t])
        else:
            dryline_associated_torwarn_n[t] = np.where(searchbox[t] == 1, tor_warn, dryline_associated_torwarn_n[t])
            dryline_associated_svrwarn_n[t] = np.where(searchbox[t] == 1, svr_warn, dryline_associated_svrwarn_n[t])  
            dryline_associated_allwarn_n[t] = np.where(searchbox[t] == 1, all_warn, dryline_associated_allwarn_n[t]) 
        
            dryline_associated_tor_n[t] = np.where(searchbox[t] == 1, tor, dryline_associated_tor_n[t])
            dryline_associated_f1tor_n[t] = np.where(searchbox[t] == 1, f1tor, dryline_associated_f1tor_n[t])
            dryline_associated_sigtor_n[t] = np.where(searchbox[t] == 1, sigtor, dryline_associated_sigtor_n[t])
            
            dryline_associated_hail_n[t] = np.where(searchbox[t] == 1, hail, dryline_associated_hail_n[t])
            dryline_associated_sighail_n[t] = np.where(searchbox[t] == 1, sighail, dryline_associated_sighail_n[t])
            
            dryline_associated_wind_n[t] = np.where(searchbox[t] == 1, wind, dryline_associated_wind_n[t])
            dryline_associated_windm_n[t] = np.where(searchbox[t] == 1, wind_m, dryline_associated_windm_n[t])
            dryline_associated_sigwind_n[t] = np.where(searchbox[t] == 1, sigwind, dryline_associated_sigwind_n[t])
            
            dryline_associated_allreports_n[t] = np.where(searchbox[t] == 1, all_reports, dryline_associated_allreports_n[t])
            dryline_associated_allsvr_n[t] = np.where(searchbox[t] == 1, all_severe, dryline_associated_allsvr_n[t])
            dryline_associated_alltor_n[t] = np.where(searchbox[t] == 1, all_tor, dryline_associated_alltor_n[t])
            
            dryline_associated_allsig_n[t] = np.where(searchbox[t] == 1, all_sig, dryline_associated_allsig_n[t])
    


# print(np.nanmax(dryline_associated_torwarn))
# print(np.nanmax(dryline_associated_svrwarn))
# print(np.nanmax(dryline_associated_allwarn))

# print(np.nanmax(dryline_associated_tor))
# print(np.nanmax(dryline_associated_f1tor))
# print(np.nanmax(dryline_associated_sigtor))
# print(np.nanmax(dryline_associated_alltor))

# print(np.nanmax(dryline_associated_hail))
# print(np.nanmax(dryline_associated_sighail))

# print(np.nanmax(dryline_associated_wind))
# print(np.nanmax(dryline_associated_windm))

# print(np.nanmax(dryline_associated_allreports))
  
#save the arrays into an output netcdf

xdf = xr.Dataset(data_vars={'tornado_el':(['time','lat','lon'],dryline_associated_tor_el),
                'f1+_tornado_el':(['time','lat','lon'],dryline_associated_f1tor_el),
                'sig_tornado_el':(['time','lat','lon'],dryline_associated_sigtor_el),
                'all_tornado_el':(['time','lat','lon'],dryline_associated_alltor_el),
                'hail_el':(['time','lat','lon'],dryline_associated_hail_el),
                'sig_hail_el':(['time','lat','lon'],dryline_associated_sighail_el),
                'wind_el':(['time','lat','lon'],dryline_associated_wind_el),
                'wind_measured_el':(['time','lat','lon'],dryline_associated_windm_el),
                'sig_wind_el':(['time','lat','lon'],dryline_associated_sigwind_el),
                'all_reports_el':(['time','lat','lon'],dryline_associated_allreports_el),
                'svr_warn_el':(['time','lat','lon'],dryline_associated_svrwarn_el),
                'tor_warn_el':(['time','lat','lon'],dryline_associated_torwarn_el),
                'all_warn_el':(['time','lat','lon'],dryline_associated_allwarn_el),
                'all_severe_el':(['time','lat','lon'],dryline_associated_allsvr_el),
                'all_sig_el':(['time','lat','lon'],dryline_associated_allsig_el),
                'tornado_la':(['time','lat','lon'],dryline_associated_tor_la),
                'f1+_tornado_la':(['time','lat','lon'],dryline_associated_f1tor_la),
                'sig_tornado_la':(['time','lat','lon'],dryline_associated_sigtor_la),
                'all_tornado_la':(['time','lat','lon'],dryline_associated_alltor_la),
                'hail_la':(['time','lat','lon'],dryline_associated_hail_la),
                'sig_hail_la':(['time','lat','lon'],dryline_associated_sighail_la),
                'wind_la':(['time','lat','lon'],dryline_associated_wind_la),
                'wind_measured_la':(['time','lat','lon'],dryline_associated_windm_la),
                'sig_wind_la':(['time','lat','lon'],dryline_associated_sigwind_la),
                'all_reports_la':(['time','lat','lon'],dryline_associated_allreports_la),
                'svr_warn_la':(['time','lat','lon'],dryline_associated_svrwarn_la),
                'tor_warn_la':(['time','lat','lon'],dryline_associated_torwarn_la),
                'all_warn_la':(['time','lat','lon'],dryline_associated_allwarn_la),
                'all_severe_la':(['time','lat','lon'],dryline_associated_allsvr_la),
                'all_sig_la':(['time','lat','lon'],dryline_associated_allsig_la),
                'tornado_n':(['time','lat','lon'],dryline_associated_tor_n),
                'f1+_tornado_n':(['time','lat','lon'],dryline_associated_f1tor_n),
                'sig_tornado_n':(['time','lat','lon'],dryline_associated_sigtor_n),
                'all_tornado_n':(['time','lat','lon'],dryline_associated_alltor_n),
                'hail_n':(['time','lat','lon'],dryline_associated_hail_n),
                'sig_hail_n':(['time','lat','lon'],dryline_associated_sighail_n),
                'wind_n':(['time','lat','lon'],dryline_associated_wind_n),
                'wind_measured_n':(['time','lat','lon'],dryline_associated_windm_n),
                'sig_wind_n':(['time','lat','lon'],dryline_associated_sigwind_n),
                'all_reports_n':(['time','lat','lon'],dryline_associated_allreports_n),
                'svr_warn_n':(['time','lat','lon'],dryline_associated_svrwarn_n),
                'tor_warn_n':(['time','lat','lon'],dryline_associated_torwarn_n),
                'all_warn_n':(['time','lat','lon'],dryline_associated_allwarn_n),
                'all_severe_n':(['time','lat','lon'],dryline_associated_allsvr_n),
                'all_sig_n':(['time','lat','lon'],dryline_associated_allsig_n)},
                coords={'lats':(['lat','lon'],lats),'lons':(['lat','lon'],lons),'time':dryline_times},
                attrs={"El": "El Nino", "N": "Neutral", "La": "La Nina"})

print(xdf)


xdf.to_netcdf('/raid/student/mhosek/era5_drylines/dryline_associated_severe_1995_2020_byenso.nc')


#%%
print('Time to run: ' + str(datetime.utcnow() - start))