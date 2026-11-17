
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

import shapely

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

"""
#read in reports, warnings, and drylines
reports = xr.load_dataset('/home/michael.hosek/Documents/ERA5_drylines/gridded_6hrly_severe_80km_1995_2020.nc') #I think xr.load will be better here than open
# warnings = xr.load_dataset('/home/michael.hosek/Documents/ERA5_drylines/gridded_6hrly_warnings_80km_1995_2020.nc')
drylines = xr.open_dataset('/raid/student/mhosek/era5_drylines/gridded_drylines_80km_1995_2020.nc')

# print(reports.sizes)
# print(reports["lats"])
# print("lat coordinate:", reports.coords.get("lat"))
# print("lon coordinate:", reports.coords.get("lon"))
# print("lats encoding:", reports["lats"].encoding)



# restrict to spring (Feb -- June); restrict warnings/reports to 21z (21z-03z)
reptimes = reports.time.values
keeptimes = []
for i in range(len(reptimes)):
    if pd.to_datetime(reptimes[i]).month >= 2 and pd.to_datetime(reptimes[i]).month <= 6 and pd.to_datetime(reptimes[i]).hour == 21:
        keeptimes.append(reptimes[i])
reports = reports.sel(time=keeptimes)

# warntimes = warnings.time.values
# keeptimes = []
# for i in range(len(warntimes)):
#     if pd.to_datetime(warntimes[i]).month >= 2 and pd.to_datetime(warntimes[i]).month <= 5 and pd.to_datetime(warntimes[i]).hour == 21:
#         keeptimes.append(warntimes[i])
# warnings = warnings.sel(time=keeptimes)

#Note: just like with warnings and reports, will have blindspot for Jan 31/Feb 1 and June 30/July 1
all_drylines_df = pd.read_pickle('/raid/student/mhosek/era5_drylines/pkl/dryline_envi_1940_2020.pkl')
spring_drylines = all_drylines_df[(all_drylines_df['month'] >= 2) & (all_drylines_df['month'] <= 6) & (all_drylines_df['year'] >= 1995)].copy()
#restrict all drylines to just the SGP (use median lat/lon to define whether it's "in" the SGP)
sgp_n = 40
sgp_s = 30
sgp_w = -105
sgp_e = -95
spring_drylines = spring_drylines[(spring_drylines['median_lon'] >= sgp_w) & (spring_drylines['median_lon'] <= sgp_e)]
spring_drylines = spring_drylines[(spring_drylines['median_lat'] >= sgp_s) & (spring_drylines['median_lat'] <= sgp_n)]

spring_dates = pd.to_datetime(spring_drylines['date'])
dryline_dates = pd.DatetimeIndex(drylines.time.values)

#restrict gridded drylines to just spring SGP dryline dates
drylines = drylines.isel(time=dryline_dates.isin(spring_dates))
dryline_times = drylines.time.values

lons, lats = drylines.lons.values, drylines.lats.values



#read in ENSO
enso_xr = xr.open_dataset("/raid/mhosek/era5_drylines/ENSO_1940_2020_ONI.nc")

enso_df = enso_xr[
    ['ENSO_phase', 'oni_detrended', 'SSTa_detrended']
].to_dataframe().reset_index()

def enso_threshold_array(values):
    return np.where(values >= 0.5, 1, np.where(values <= -0.5, -1, 0))

enso_df['year'] = enso_df['time'].dt.year
enso_df['month'] = enso_df['time'].dt.month

enso_df['op_enso'] = enso_df['ENSO_phase']
enso_df['3mo_enso'] = enso_threshold_array(enso_df['oni_detrended'])
enso_df['1mo_enso'] = enso_threshold_array(enso_df['SSTa_detrended'])

enso_df = enso_df[['year', 'month', 'op_enso', '3mo_enso', '1mo_enso']]


print("read in all data")




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



# dryline_associated_torwarn_el = np.zeros(dry.shape)
# dryline_associated_svrwarn_el = np.zeros(dry.shape)
dryline_associated_tor_el = np.zeros(dry.shape)
dryline_associated_f1tor_el = np.zeros(dry.shape)
dryline_associated_sigtor_el = np.zeros(dry.shape)
dryline_associated_hail_el = np.zeros(dry.shape)
dryline_associated_sighail_el = np.zeros(dry.shape)
dryline_associated_wind_el = np.zeros(dry.shape)
dryline_associated_windm_el = np.zeros(dry.shape)
dryline_associated_sigwind_el = np.zeros(dry.shape)
# dryline_associated_allsvr_el = np.zeros(dry.shape)
dryline_associated_allreports_el = np.zeros(dry.shape)
# dryline_associated_allwarn_el = np.zeros(dry.shape)
dryline_associated_alltor_el = np.zeros(dry.shape)
dryline_associated_allsig_el = np.zeros(dry.shape)

# dryline_associated_torwarn_la = np.zeros(dry.shape)
# dryline_associated_svrwarn_la = np.zeros(dry.shape)
dryline_associated_tor_la = np.zeros(dry.shape)
dryline_associated_f1tor_la = np.zeros(dry.shape)
dryline_associated_sigtor_la = np.zeros(dry.shape)
dryline_associated_hail_la = np.zeros(dry.shape)
dryline_associated_sighail_la = np.zeros(dry.shape)
dryline_associated_wind_la = np.zeros(dry.shape)
dryline_associated_windm_la = np.zeros(dry.shape)
dryline_associated_sigwind_la = np.zeros(dry.shape)
# dryline_associated_allsvr_la = np.zeros(dry.shape)
dryline_associated_allreports_la = np.zeros(dry.shape)
# dryline_associated_allwarn_la = np.zeros(dry.shape)
dryline_associated_alltor_la = np.zeros(dry.shape)
dryline_associated_allsig_la = np.zeros(dry.shape)

# dryline_associated_torwarn_n = np.zeros(dry.shape)
# dryline_associated_svrwarn_n = np.zeros(dry.shape)
dryline_associated_tor_n = np.zeros(dry.shape)
dryline_associated_f1tor_n = np.zeros(dry.shape)
dryline_associated_sigtor_n = np.zeros(dry.shape)
dryline_associated_hail_n = np.zeros(dry.shape)
dryline_associated_sighail_n = np.zeros(dry.shape)
dryline_associated_wind_n = np.zeros(dry.shape)
dryline_associated_windm_n = np.zeros(dry.shape)
dryline_associated_sigwind_n = np.zeros(dry.shape)
# dryline_associated_allsvr_n = np.zeros(dry.shape)
dryline_associated_allreports_n = np.zeros(dry.shape)
dryline_associated_allwarn_n = np.zeros(dry.shape)
dryline_associated_alltor_n = np.zeros(dry.shape)
dryline_associated_allsig_n = np.zeros(dry.shape)

for t in range(len(dryline_times)):
    #define the search times as 3 hours prior to the dryline time (so, for a 00z dryline time, the search time is 21z the "previous day")
    search_time = dryline_times[t] - np.timedelta64(3, 'h') #reports/warn for 21z-03z
    
    if pd.to_datetime(search_time).month != 1: #make sure we don't include any January drylines
    
        #warnings first
        # warn_xary = warnings.sel(time=search_time)
        
        #all of these have a nanmax of 1
        # tor_warn = warn_xary.tor_warn.values
        # svr_warn = warn_xary.svr_warn.values
        # all_warn = warn_xary.all_warn.values
        
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
        # all_severe = all_reports + all_warn
        # all_severe = np.where(all_severe > 1, 1, all_severe)
    
        #create an "all tornado" warning OR report
        # all_tor = tor_warn + tor
        # all_tor = np.where(all_tor > 1, 1, all_tor)
        
        #create an "all sig" report
        all_sig = sigtor + sighail + sigwind
        all_sig = np.where(all_sig > 1, 1, all_sig)

        #determine ENSO phase at this time
        month, year = pd.to_datetime(search_time).month, pd.to_datetime(search_time).year
        indiv_enso = enso_df[(enso_df['month']==month) & (enso_df['year']==year)]
        index = indiv_enso.iloc[0]['op_enso']

        #Associate warnings and reports with drylines, by enso phase
        if index <= -0.5:
            # dryline_associated_torwarn_la[t] = np.where(searchbox[t] == 1, tor_warn, dryline_associated_torwarn_la[t])
            # dryline_associated_svrwarn_la[t] = np.where(searchbox[t] == 1, svr_warn, dryline_associated_svrwarn_la[t])  
            # dryline_associated_allwarn_la[t] = np.where(searchbox[t] == 1, all_warn, dryline_associated_allwarn_la[t]) 
        
            dryline_associated_tor_la[t] = np.where(searchbox[t] == 1, tor, dryline_associated_tor_la[t])
            dryline_associated_f1tor_la[t] = np.where(searchbox[t] == 1, f1tor, dryline_associated_f1tor_la[t])
            dryline_associated_sigtor_la[t] = np.where(searchbox[t] == 1, sigtor, dryline_associated_sigtor_la[t])
            
            dryline_associated_hail_la[t] = np.where(searchbox[t] == 1, hail, dryline_associated_hail_la[t])
            dryline_associated_sighail_la[t] = np.where(searchbox[t] == 1, sighail, dryline_associated_sighail_la[t])
            
            dryline_associated_wind_la[t] = np.where(searchbox[t] == 1, wind, dryline_associated_wind_la[t])
            dryline_associated_windm_la[t] = np.where(searchbox[t] == 1, wind_m, dryline_associated_windm_la[t])
            dryline_associated_sigwind_la[t] = np.where(searchbox[t] == 1, sigwind, dryline_associated_sigwind_la[t])
            
            dryline_associated_allreports_la[t] = np.where(searchbox[t] == 1, all_reports, dryline_associated_allreports_la[t])
            # dryline_associated_allsvr_la[t] = np.where(searchbox[t] == 1, all_severe, dryline_associated_allsvr_la[t])
            # dryline_associated_alltor_la[t] = np.where(searchbox[t] == 1, all_tor, dryline_associated_alltor_la[t])
            
            dryline_associated_allsig_la[t] = np.where(searchbox[t] == 1, all_sig, dryline_associated_allsig_la[t])
        elif index >= 0.5:
            # dryline_associated_torwarn_el[t] = np.where(searchbox[t] == 1, tor_warn, dryline_associated_torwarn_el[t])
            # dryline_associated_svrwarn_el[t] = np.where(searchbox[t] == 1, svr_warn, dryline_associated_svrwarn_el[t])  
            # dryline_associated_allwarn_el[t] = np.where(searchbox[t] == 1, all_warn, dryline_associated_allwarn_el[t]) 
        
            dryline_associated_tor_el[t] = np.where(searchbox[t] == 1, tor, dryline_associated_tor_el[t])
            dryline_associated_f1tor_el[t] = np.where(searchbox[t] == 1, f1tor, dryline_associated_f1tor_el[t])
            dryline_associated_sigtor_el[t] = np.where(searchbox[t] == 1, sigtor, dryline_associated_sigtor_el[t])
            
            dryline_associated_hail_el[t] = np.where(searchbox[t] == 1, hail, dryline_associated_hail_el[t])
            dryline_associated_sighail_el[t] = np.where(searchbox[t] == 1, sighail, dryline_associated_sighail_el[t])
            
            dryline_associated_wind_el[t] = np.where(searchbox[t] == 1, wind, dryline_associated_wind_el[t])
            dryline_associated_windm_el[t] = np.where(searchbox[t] == 1, wind_m, dryline_associated_windm_el[t])
            dryline_associated_sigwind_el[t] = np.where(searchbox[t] == 1, sigwind, dryline_associated_sigwind_el[t])
            
            dryline_associated_allreports_el[t] = np.where(searchbox[t] == 1, all_reports, dryline_associated_allreports_el[t])
            # dryline_associated_allsvr_el[t] = np.where(searchbox[t] == 1, all_severe, dryline_associated_allsvr_el[t])
            # dryline_associated_alltor_el[t] = np.where(searchbox[t] == 1, all_tor, dryline_associated_alltor_el[t])
            
            dryline_associated_allsig_el[t] = np.where(searchbox[t] == 1, all_sig, dryline_associated_allsig_el[t])
        else:
            # dryline_associated_torwarn_n[t] = np.where(searchbox[t] == 1, tor_warn, dryline_associated_torwarn_n[t])
            # dryline_associated_svrwarn_n[t] = np.where(searchbox[t] == 1, svr_warn, dryline_associated_svrwarn_n[t])  
            # dryline_associated_allwarn_n[t] = np.where(searchbox[t] == 1, all_warn, dryline_associated_allwarn_n[t]) 
        
            dryline_associated_tor_n[t] = np.where(searchbox[t] == 1, tor, dryline_associated_tor_n[t])
            dryline_associated_f1tor_n[t] = np.where(searchbox[t] == 1, f1tor, dryline_associated_f1tor_n[t])
            dryline_associated_sigtor_n[t] = np.where(searchbox[t] == 1, sigtor, dryline_associated_sigtor_n[t])
            
            dryline_associated_hail_n[t] = np.where(searchbox[t] == 1, hail, dryline_associated_hail_n[t])
            dryline_associated_sighail_n[t] = np.where(searchbox[t] == 1, sighail, dryline_associated_sighail_n[t])
            
            dryline_associated_wind_n[t] = np.where(searchbox[t] == 1, wind, dryline_associated_wind_n[t])
            dryline_associated_windm_n[t] = np.where(searchbox[t] == 1, wind_m, dryline_associated_windm_n[t])
            dryline_associated_sigwind_n[t] = np.where(searchbox[t] == 1, sigwind, dryline_associated_sigwind_n[t])
            
            dryline_associated_allreports_n[t] = np.where(searchbox[t] == 1, all_reports, dryline_associated_allreports_n[t])
            # dryline_associated_allsvr_n[t] = np.where(searchbox[t] == 1, all_severe, dryline_associated_allsvr_n[t])
            # dryline_associated_alltor_n[t] = np.where(searchbox[t] == 1, all_tor, dryline_associated_alltor_n[t])
            
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
                # 'svr_warn_el':(['time','lat','lon'],dryline_associated_svrwarn_el),
                # 'tor_warn_el':(['time','lat','lon'],dryline_associated_torwarn_el),
                # 'all_warn_el':(['time','lat','lon'],dryline_associated_allwarn_el),
                # 'all_severe_el':(['time','lat','lon'],dryline_associated_allsvr_el),
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
                # 'svr_warn_la':(['time','lat','lon'],dryline_associated_svrwarn_la),
                # 'tor_warn_la':(['time','lat','lon'],dryline_associated_torwarn_la),
                # 'all_warn_la':(['time','lat','lon'],dryline_associated_allwarn_la),
                # 'all_severe_la':(['time','lat','lon'],dryline_associated_allsvr_la),
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
                # 'svr_warn_n':(['time','lat','lon'],dryline_associated_svrwarn_n),
                # 'tor_warn_n':(['time','lat','lon'],dryline_associated_torwarn_n),
                # 'all_warn_n':(['time','lat','lon'],dryline_associated_allwarn_n),
                # 'all_severe_n':(['time','lat','lon'],dryline_associated_allsvr_n),
                'all_sig_n':(['time','lat','lon'],dryline_associated_allsig_n)},
                coords={'lats':(['lat','lon'],lats),'lons':(['lat','lon'],lons),'time':dryline_times},
                attrs={"El": "El Nino", "N": "Neutral", "La": "La Nina"})

print(xdf)

#restrict both the reports and dryline-associated severe xarrays to just the SGP
inside_sgp = (
    (xdf["lats"] >= 30) & (xdf["lats"] <= 40)
    & (xdf["lons"] >= -105) & (xdf["lons"] <= -95)
)
xdf = xdf.where(inside_sgp, drop=True)

inside_sgp = (
    (reports["lats"] >= 30) & (reports["lats"] <= 40)
    & (reports["lons"] >= -105) & (reports["lons"] <= -95)
)
reports = reports.where(inside_sgp, drop=True)

print(reports.sizes)
assert reports.sizes["lat"] > 0 and reports.sizes["lon"] > 0


xdf.to_netcdf('/raid/mhosek/era5_drylines/dryline_associated_severe_1995_2020_byenso.nc')
reports.to_netcdf('/raid/mhosek/era5_drylines/SGP_reports_1995_2020.nc')

"""
###################################################
###################################################
################################################### build a pickle
###################################################
###################################################
drylines_svr = xr.open_dataset('/raid/mhosek/era5_drylines/dryline_associated_severe_1995_2020_byenso.nc')
reports = xr.open_dataset('/raid/mhosek/era5_drylines/SGP_reports_1995_2020.nc')

# lon, lat = drylines_svr.lons.values, drylines_svr.lats.values
lon, lat = drylines_svr["lons"], drylines_svr["lats"]


#read in ENSO
enso_xr = xr.open_dataset("/raid/mhosek/era5_drylines/ENSO_1940_2020_ONI.nc")

enso_df = enso_xr[
    ['ENSO_phase', 'oni_detrended', 'SSTa_detrended']
].to_dataframe().reset_index()

def enso_threshold_array(values):
    return np.where(values >= 0.5, 1, np.where(values <= -0.5, -1, 0))

enso_df['year'] = enso_df['time'].dt.year
enso_df['month'] = enso_df['time'].dt.month

enso_df['op_enso'] = enso_df['ENSO_phase']
enso_df['3mo_enso'] = enso_threshold_array(enso_df['oni_detrended'])
enso_df['1mo_enso'] = enso_threshold_array(enso_df['SSTa_detrended'])

enso_df = enso_df[['year', 'month', 'op_enso', '3mo_enso', '1mo_enso']]


#the pickle df will be easy: list each day, then store a 1 or 0 under each category for that day
#categories: tor report, hail report, wind report, any report; for both dryline-associated severe and for entire SGP
#also define 4 quadrants within SGP, and repeat the categories for each quadrant
    #4 quadrants defined by 100W, 35N

#also, save out the area covered by SGP SVR/dryline-associated SVR for each day (# of grid points). For entire SGP and quadrants
#and for ease, add on a column for the enso phase by different definitions (op, 3mo, 1mo)


#define quadrants
q1 = (
    (lon >= -100.0) & (lon <= -95)
    & (lat >= 35) & (lat <= 40)
)

q2 = (
    (lon <= -100.0) & (lon >= -105)
    & (lat >= 35) & (lat <= 40)
)

q3 = (
    (lon <= -100.0) & (lon >= -105)
    & (lat <= 35) & (lat >= 30)
)

q4 = (
    (lon >= -100.0) & (lon <= -95)
    & (lat <= 35) & (lat >= 30)
)

dry_svr_q1 = drylines_svr.where(q1, drop=True)
dry_svr_q2 = drylines_svr.where(q2, drop=True)
dry_svr_q3 = drylines_svr.where(q3, drop=True)
dry_svr_q4 = drylines_svr.where(q4, drop=True)

reports_q1 = reports.where(q1, drop=True)
reports_q2 = reports.where(q2, drop=True)
reports_q3 = reports.where(q3, drop=True)
reports_q4 = reports.where(q4, drop=True)


startdate = datetime(1995, 2, 1)
enddate = datetime(2020, 6, 30)
dates = pd.date_range(start=startdate, end=enddate, freq='D')

#restrict to just feb-June each year
dates = dates[(dates.month >= 2) & (dates.month <= 6)]


def restrict_to_date(xr, date, reports=False):
    #given a single date, restrict the given xarray to just that date
    if reports == False:
        day = pd.Timestamp(date).strftime('%Y-%m-%d %H:%M:%S')
    else:
        date = date - timedelta(days=1) #because it's 21z the "previous day" for a 00z dryline
        day = pd.Timestamp(date).strftime('%Y-%m-%d 21:00:00')
    try:
        return xr.sel(time=np.datetime64(day))
    except KeyError:
        return None


######################### all
all_report_bool = []
all_report_area = []

dry_report_bool = []
dry_report_area = []

dry_report_q1_bool = []
dry_report_q1_area = []
dry_report_q2_bool = []
dry_report_q2_area = []
dry_report_q3_bool = []
dry_report_q3_area = []
dry_report_q4_bool = []
dry_report_q4_area = []

all_report_q1_bool = []
all_report_q1_area = []
all_report_q2_bool = []
all_report_q2_area = []
all_report_q3_bool = []
all_report_q3_area = []
all_report_q4_bool = []
all_report_q4_area = []

######################### tor
tor_report_bool = []
tor_report_area = []

tor_dry_report_bool = []
tor_dry_report_area = []

tor_dry_report_q1_bool = []
tor_dry_report_q1_area = []
tor_dry_report_q2_bool = []
tor_dry_report_q2_area = []
tor_dry_report_q3_bool = []
tor_dry_report_q3_area = []
tor_dry_report_q4_bool = []
tor_dry_report_q4_area = []

tor_all_report_q1_bool = []
tor_all_report_q1_area = []
tor_all_report_q2_bool = []
tor_all_report_q2_area = []
tor_all_report_q3_bool = []
tor_all_report_q3_area = []
tor_all_report_q4_bool = []
tor_all_report_q4_area = []


######################### hail
hail_report_bool = []
hail_report_area = []

hail_dry_report_bool = []
hail_dry_report_area = []

hail_dry_report_q1_bool = []
hail_dry_report_q1_area = []
hail_dry_report_q2_bool = []
hail_dry_report_q2_area = []
hail_dry_report_q3_bool = []
hail_dry_report_q3_area = []
hail_dry_report_q4_bool = []
hail_dry_report_q4_area = []

hail_all_report_q1_bool = []
hail_all_report_q1_area = []
hail_all_report_q2_bool = []
hail_all_report_q2_area = []
hail_all_report_q3_bool = []
hail_all_report_q3_area = []
hail_all_report_q4_bool = []
hail_all_report_q4_area = []

######################### wind
wind_report_bool = []
wind_report_area = []

wind_dry_report_bool = []
wind_dry_report_area = []

wind_dry_report_q1_bool = []
wind_dry_report_q1_area = []
wind_dry_report_q2_bool = []
wind_dry_report_q2_area = []
wind_dry_report_q3_bool = []
wind_dry_report_q3_area = []
wind_dry_report_q4_bool = []
wind_dry_report_q4_area = []

wind_all_report_q1_bool = []
wind_all_report_q1_area = []
wind_all_report_q2_bool = []
wind_all_report_q2_area = []
wind_all_report_q3_bool = []
wind_all_report_q3_area = []
wind_all_report_q4_bool = []
wind_all_report_q4_area = []


#enso
enso_op = []
enso_3mo = []
enso_1mo = []

for d in dates:
    print(d)

    yr, mo = d.year, d.month
    temp_enso = enso_df[(enso_df['year']== yr) & (enso_df['month']== mo)]
    enso_op.append(temp_enso['op_enso'].values[0])
    enso_3mo.append(temp_enso['3mo_enso'].values[0])
    enso_1mo.append(temp_enso['1mo_enso'].values[0])

    
    dry_svr_day = restrict_to_date(drylines_svr, d)
    if dry_svr_day is None:
        dry_report_bool.append(False)
        dry_report_q1_bool.append(False)
        dry_report_q2_bool.append(False)
        dry_report_q3_bool.append(False)
        dry_report_q4_bool.append(False)
        tor_dry_report_bool.append(False)
        tor_dry_report_q1_bool.append(False)
        tor_dry_report_q2_bool.append(False)
        tor_dry_report_q3_bool.append(False)
        tor_dry_report_q4_bool.append(False)
        hail_dry_report_bool.append(False)
        hail_dry_report_q1_bool.append(False)
        hail_dry_report_q2_bool.append(False)
        hail_dry_report_q3_bool.append(False)
        hail_dry_report_q4_bool.append(False)
        wind_dry_report_bool.append(False)
        wind_dry_report_q1_bool.append(False)
        wind_dry_report_q2_bool.append(False)
        wind_dry_report_q3_bool.append(False)
        wind_dry_report_q4_bool.append(False)

        dry_report_area.append(np.nan)
        dry_report_q1_area.append(np.nan)
        dry_report_q2_area.append(np.nan)
        dry_report_q3_area.append(np.nan)
        dry_report_q4_area.append(np.nan)
        tor_dry_report_area.append(np.nan)
        tor_dry_report_q1_area.append(np.nan)
        tor_dry_report_q2_area.append(np.nan)
        tor_dry_report_q3_area.append(np.nan)
        tor_dry_report_q4_area.append(np.nan)
        hail_dry_report_area.append(np.nan)
        hail_dry_report_q1_area.append(np.nan)
        hail_dry_report_q2_area.append(np.nan)
        hail_dry_report_q3_area.append(np.nan)
        hail_dry_report_q4_area.append(np.nan)
        wind_dry_report_area.append(np.nan)
        wind_dry_report_q1_area.append(np.nan)
        wind_dry_report_q2_area.append(np.nan)
        wind_dry_report_q3_area.append(np.nan)
        wind_dry_report_q4_area.append(np.nan)
    else:
        dry_svr_q1_day = restrict_to_date(dry_svr_q1, d)
        dry_svr_q2_day = restrict_to_date(dry_svr_q2, d)
        dry_svr_q3_day = restrict_to_date(dry_svr_q3, d)
        dry_svr_q4_day = restrict_to_date(dry_svr_q4, d)

        dry_report_area.append(np.nansum(dry_svr_day.all_reports_n.values) + np.nansum(dry_svr_day.all_reports_el.values) + np.nansum(dry_svr_day.all_reports_la.values))
        dry_report_q1_area.append(np.nansum(dry_svr_q1_day.all_reports_n.values) + np.nansum(dry_svr_q1_day.all_reports_el.values) + np.nansum(dry_svr_q1_day.all_reports_la.values))
        dry_report_q2_area.append(np.nansum(dry_svr_q2_day.all_reports_n.values) + np.nansum(dry_svr_q2_day.all_reports_el.values) + np.nansum(dry_svr_q2_day.all_reports_la.values))
        dry_report_q3_area.append(np.nansum(dry_svr_q3_day.all_reports_n.values) + np.nansum(dry_svr_q3_day.all_reports_el.values) + np.nansum(dry_svr_q3_day.all_reports_la.values))
        dry_report_q4_area.append(np.nansum(dry_svr_q4_day.all_reports_n.values) + np.nansum(dry_svr_q4_day.all_reports_el.values) + np.nansum(dry_svr_q4_day.all_reports_la.values))
        tor_dry_report_area.append(np.nansum(dry_svr_day.tornado_n.values) + np.nansum(dry_svr_day.tornado_el.values) + np.nansum(dry_svr_day.tornado_la.values))
        tor_dry_report_q1_area.append(np.nansum(dry_svr_q1_day.tornado_n.values) + np.nansum(dry_svr_q1_day.tornado_el.values) + np.nansum(dry_svr_q1_day.tornado_la.values))
        tor_dry_report_q2_area.append(np.nansum(dry_svr_q2_day.tornado_n.values) + np.nansum(dry_svr_q2_day.tornado_el.values) + np.nansum(dry_svr_q2_day.tornado_la.values))
        tor_dry_report_q3_area.append(np.nansum(dry_svr_q3_day.tornado_n.values) + np.nansum(dry_svr_q3_day.tornado_el.values) + np.nansum(dry_svr_q3_day.tornado_la.values))
        tor_dry_report_q4_area.append(np.nansum(dry_svr_q4_day.tornado_n.values) + np.nansum(dry_svr_q4_day.tornado_el.values) + np.nansum(dry_svr_q4_day.tornado_la.values))
        hail_dry_report_area.append(np.nansum(dry_svr_day.hail_n.values) + np.nansum(dry_svr_day.hail_el.values) + np.nansum(dry_svr_day.hail_la.values))
        hail_dry_report_q1_area.append(np.nansum(dry_svr_q1_day.hail_n.values) + np.nansum(dry_svr_q1_day.hail_el.values) + np.nansum(dry_svr_q1_day.hail_la.values))
        hail_dry_report_q2_area.append(np.nansum(dry_svr_q2_day.hail_n.values) + np.nansum(dry_svr_q2_day.hail_el.values) + np.nansum(dry_svr_q2_day.hail_la.values))
        hail_dry_report_q3_area.append(np.nansum(dry_svr_q3_day.hail_n.values) + np.nansum(dry_svr_q3_day.hail_el.values) + np.nansum(dry_svr_q3_day.hail_la.values))
        hail_dry_report_q4_area.append(np.nansum(dry_svr_q4_day.hail_n.values) + np.nansum(dry_svr_q4_day.hail_el.values) + np.nansum(dry_svr_q4_day.hail_la.values))
        wind_dry_report_area.append(np.nansum(dry_svr_day.wind_n.values) + np.nansum(dry_svr_day.wind_el.values) + np.nansum(dry_svr_day.wind_la.values))
        wind_dry_report_q1_area.append(np.nansum(dry_svr_q1_day.wind_n.values) + np.nansum(dry_svr_q1_day.wind_el.values) + np.nansum(dry_svr_q1_day.wind_la.values))
        wind_dry_report_q2_area.append(np.nansum(dry_svr_q2_day.wind_n.values) + np.nansum(dry_svr_q2_day.wind_el.values) + np.nansum(dry_svr_q2_day.wind_la.values))
        wind_dry_report_q3_area.append(np.nansum(dry_svr_q3_day.wind_n.values) + np.nansum(dry_svr_q3_day.wind_el.values) + np.nansum(dry_svr_q3_day.wind_la.values))
        wind_dry_report_q4_area.append(np.nansum(dry_svr_q4_day.wind_n.values) + np.nansum(dry_svr_q4_day.wind_el.values) + np.nansum(dry_svr_q4_day.wind_la.values))

        dry_report_bool.append(dry_report_area[-1] > 0)
        dry_report_q1_bool.append(dry_report_q1_area[-1] > 0)
        dry_report_q2_bool.append(dry_report_q2_area[-1] > 0)
        dry_report_q3_bool.append(dry_report_q3_area[-1] > 0)
        dry_report_q4_bool.append(dry_report_q4_area[-1] > 0)
        tor_dry_report_bool.append(tor_dry_report_area[-1] > 0)
        tor_dry_report_q1_bool.append(tor_dry_report_q1_area[-1] > 0)
        tor_dry_report_q2_bool.append(tor_dry_report_q2_area[-1] > 0)
        tor_dry_report_q3_bool.append(tor_dry_report_q3_area[-1] > 0)
        tor_dry_report_q4_bool.append(tor_dry_report_q4_area[-1] > 0)
        hail_dry_report_bool.append(hail_dry_report_area[-1] > 0)
        hail_dry_report_q1_bool.append(hail_dry_report_q1_area[-1] > 0)
        hail_dry_report_q2_bool.append(hail_dry_report_q2_area[-1] > 0)
        hail_dry_report_q3_bool.append(hail_dry_report_q3_area[-1] > 0)
        hail_dry_report_q4_bool.append(hail_dry_report_q4_area[-1] > 0)
        wind_dry_report_bool.append(wind_dry_report_area[-1] > 0)
        wind_dry_report_q1_bool.append(wind_dry_report_q1_area[-1] > 0)
        wind_dry_report_q2_bool.append(wind_dry_report_q2_area[-1] > 0)
        wind_dry_report_q3_bool.append(wind_dry_report_q3_area[-1] > 0)
        wind_dry_report_q4_bool.append(wind_dry_report_q4_area[-1] > 0)

    #now, do all reports
    reports_day = restrict_to_date(reports, d, reports=True)
    if reports_day is None:
        all_report_bool.append(False)
        all_report_q1_bool.append(False)
        all_report_q2_bool.append(False)
        all_report_q3_bool.append(False)
        all_report_q4_bool.append(False)
        tor_report_bool.append(False)
        tor_all_report_q1_bool.append(False)
        tor_all_report_q2_bool.append(False)
        tor_all_report_q3_bool.append(False)
        tor_all_report_q4_bool.append(False)
        hail_report_bool.append(False)
        hail_all_report_q1_bool.append(False)
        hail_all_report_q2_bool.append(False)
        hail_all_report_q3_bool.append(False)
        hail_all_report_q4_bool.append(False)
        wind_report_bool.append(False)
        wind_all_report_q1_bool.append(False)
        wind_all_report_q2_bool.append(False)
        wind_all_report_q3_bool.append(False)
        wind_all_report_q4_bool.append(False)

        all_report_area.append(np.nan)
        all_report_q1_area.append(np.nan)
        all_report_q2_area.append(np.nan)
        all_report_q3_area.append(np.nan)
        all_report_q4_area.append(np.nan)
        tor_report_area.append(np.nan)
        tor_all_report_q1_area.append(np.nan)
        tor_all_report_q2_area.append(np.nan)
        tor_all_report_q3_area.append(np.nan)
        tor_all_report_q4_area.append(np.nan)
        hail_report_area.append(np.nan)
        hail_all_report_q1_area.append(np.nan)
        hail_all_report_q2_area.append(np.nan)
        hail_all_report_q3_area.append(np.nan)
        hail_all_report_q4_area.append(np.nan)
        wind_report_area.append(np.nan)
        wind_all_report_q1_area.append(np.nan)
        wind_all_report_q2_area.append(np.nan)
        wind_all_report_q3_area.append(np.nan)
        wind_all_report_q4_area.append(np.nan)

    else:
        reports_q1_day = restrict_to_date(reports_q1, d, reports=True)
        reports_q2_day = restrict_to_date(reports_q2, d, reports=True)
        reports_q3_day = restrict_to_date(reports_q3, d, reports=True)
        reports_q4_day = restrict_to_date(reports_q4, d, reports=True)


        all_report_area.append(np.nansum(reports_day.all_reports.values))
        all_report_bool.append(all_report_area[-1] > 0)

        all_report_q1_area.append(np.nansum(reports_q1_day.all_reports.values))
        all_report_q1_bool.append(all_report_q1_area[-1] > 0)

        all_report_q2_area.append(np.nansum(reports_q2_day.all_reports.values))
        all_report_q2_bool.append(all_report_q2_area[-1] > 0)

        all_report_q3_area.append(np.nansum(reports_q3_day.all_reports.values))
        all_report_q3_bool.append(all_report_q3_area[-1] > 0)

        all_report_q4_area.append(np.nansum(reports_q4_day.all_reports.values))
        all_report_q4_bool.append(all_report_q4_area[-1] > 0)

        tor_report_area.append(np.nansum(reports_day.tornado.values))
        tor_all_report_q1_area.append(np.nansum(reports_q1_day.tornado.values))
        tor_all_report_q2_area.append(np.nansum(reports_q2_day.tornado.values))
        tor_all_report_q3_area.append(np.nansum(reports_q3_day.tornado.values))
        tor_all_report_q4_area.append(np.nansum(reports_q4_day.tornado.values))
        tor_report_bool.append(tor_report_area[-1] > 0)
        tor_all_report_q1_bool.append(tor_all_report_q1_area[-1] > 0)
        tor_all_report_q2_bool.append(tor_all_report_q2_area[-1] > 0)
        tor_all_report_q3_bool.append(tor_all_report_q3_area[-1] > 0)
        tor_all_report_q4_bool.append(tor_all_report_q4_area[-1] > 0)

        hail_report_area.append(np.nansum(reports_day.hail.values))
        hail_all_report_q1_area.append(np.nansum(reports_q1_day.hail.values))
        hail_all_report_q2_area.append(np.nansum(reports_q2_day.hail.values))
        hail_all_report_q3_area.append(np.nansum(reports_q3_day.hail.values))
        hail_all_report_q4_area.append(np.nansum(reports_q4_day.hail.values))
        hail_report_bool.append(hail_report_area[-1] > 0)
        hail_all_report_q1_bool.append(hail_all_report_q1_area[-1] > 0)
        hail_all_report_q2_bool.append(hail_all_report_q2_area[-1] > 0)
        hail_all_report_q3_bool.append(hail_all_report_q3_area[-1] > 0)
        hail_all_report_q4_bool.append(hail_all_report_q4_area[-1] > 0)

        wind_report_area.append(np.nansum(reports_day.wind.values))
        wind_all_report_q1_area.append(np.nansum(reports_q1_day.wind.values))
        wind_all_report_q2_area.append(np.nansum(reports_q2_day.wind.values))
        wind_all_report_q3_area.append(np.nansum(reports_q3_day.wind.values))
        wind_all_report_q4_area.append(np.nansum(reports_q4_day.wind.values))
        wind_report_bool.append(wind_report_area[-1] > 0)
        wind_all_report_q1_bool.append(wind_all_report_q1_area[-1] > 0)
        wind_all_report_q2_bool.append(wind_all_report_q2_area[-1] > 0)
        wind_all_report_q3_bool.append(wind_all_report_q3_area[-1] > 0)
        wind_all_report_q4_bool.append(wind_all_report_q4_area[-1] > 0)


svr_summary = pd.DataFrame()
svr_summary['time'] = dates
svr_summary['all_report_area'] = all_report_area
svr_summary['all_report_bool'] = all_report_bool
svr_summary['all_tor_report_area'] = tor_report_area
svr_summary['all_tor_report_bool'] = tor_report_bool
svr_summary['all_hail_report_area'] = hail_report_area
svr_summary['all_hail_report_bool'] = hail_report_bool
svr_summary['all_wind_report_area'] = wind_report_area
svr_summary['all_wind_report_bool'] = wind_report_bool

svr_summary['all_report_q1_bool'] = all_report_q1_bool
svr_summary['all_report_q2_bool'] = all_report_q2_bool
svr_summary['all_report_q3_bool'] = all_report_q3_bool
svr_summary['all_report_q4_bool'] = all_report_q4_bool
svr_summary['all_report_q1_area'] = all_report_q1_area
svr_summary['all_report_q2_area'] = all_report_q2_area
svr_summary['all_report_q3_area'] = all_report_q3_area
svr_summary['all_report_q4_area'] = all_report_q4_area

svr_summary['all_tor_report_q1_area'] = tor_all_report_q1_area
svr_summary['all_tor_report_q2_area'] = tor_all_report_q2_area
svr_summary['all_tor_report_q3_area'] = tor_all_report_q3_area
svr_summary['all_tor_report_q4_area'] = tor_all_report_q4_area
svr_summary['all_tor_report_q1_bool'] = tor_all_report_q1_bool
svr_summary['all_tor_report_q2_bool'] = tor_all_report_q2_bool
svr_summary['all_tor_report_q3_bool'] = tor_all_report_q3_bool
svr_summary['all_tor_report_q4_bool'] = tor_all_report_q4_bool

svr_summary['all_hail_report_q1_area'] = hail_all_report_q1_area
svr_summary['all_hail_report_q2_area'] = hail_all_report_q2_area
svr_summary['all_hail_report_q3_area'] = hail_all_report_q3_area
svr_summary['all_hail_report_q4_area'] = hail_all_report_q4_area
svr_summary['all_hail_report_q1_bool'] = hail_all_report_q1_bool
svr_summary['all_hail_report_q2_bool'] = hail_all_report_q2_bool
svr_summary['all_hail_report_q3_bool'] = hail_all_report_q3_bool
svr_summary['all_hail_report_q4_bool'] = hail_all_report_q4_bool

svr_summary['all_wind_report_q1_area'] = wind_all_report_q1_area
svr_summary['all_wind_report_q2_area'] = wind_all_report_q2_area
svr_summary['all_wind_report_q3_area'] = wind_all_report_q3_area
svr_summary['all_wind_report_q4_area'] = wind_all_report_q4_area
svr_summary['all_wind_report_q1_bool'] = wind_all_report_q1_bool
svr_summary['all_wind_report_q2_bool'] = wind_all_report_q2_bool
svr_summary['all_wind_report_q3_bool'] = wind_all_report_q3_bool
svr_summary['all_wind_report_q4_bool'] = wind_all_report_q4_bool

svr_summary['dry_report_area'] = dry_report_area
svr_summary['dry_report_bool'] = dry_report_bool
svr_summary['dry_tor_report_bool'] = tor_dry_report_bool
svr_summary['dry_tor_report_area'] = tor_dry_report_area
svr_summary['dry_hail_report_bool'] = hail_dry_report_bool
svr_summary['dry_hail_report_area'] = hail_dry_report_area
svr_summary['dry_wind_report_bool'] = wind_dry_report_bool
svr_summary['dry_wind_report_area'] = wind_dry_report_area

svr_summary['dry_report_q1_bool'] = dry_report_q1_bool
svr_summary['dry_report_q2_bool'] = dry_report_q2_bool
svr_summary['dry_report_q3_bool'] = dry_report_q3_bool
svr_summary['dry_report_q4_bool'] = dry_report_q4_bool
svr_summary['dry_report_q1_area'] = dry_report_q1_area
svr_summary['dry_report_q2_area'] = dry_report_q2_area
svr_summary['dry_report_q3_area'] = dry_report_q3_area
svr_summary['dry_report_q4_area'] = dry_report_q4_area

svr_summary['dry_tor_report_q1_bool'] = tor_dry_report_q1_bool
svr_summary['dry_tor_report_q2_bool'] = tor_dry_report_q2_bool
svr_summary['dry_tor_report_q3_bool'] = tor_dry_report_q3_bool
svr_summary['dry_tor_report_q4_bool'] = tor_dry_report_q4_bool
svr_summary['dry_tor_report_q1_area'] = tor_dry_report_q1_area
svr_summary['dry_tor_report_q2_area'] = tor_dry_report_q2_area
svr_summary['dry_tor_report_q3_area'] = tor_dry_report_q3_area
svr_summary['dry_tor_report_q4_area'] = tor_dry_report_q4_area

svr_summary['dry_hail_report_q1_bool'] = hail_dry_report_q1_bool
svr_summary['dry_hail_report_q2_bool'] = hail_dry_report_q2_bool
svr_summary['dry_hail_report_q3_bool'] = hail_dry_report_q3_bool
svr_summary['dry_hail_report_q4_bool'] = hail_dry_report_q4_bool
svr_summary['dry_hail_report_q1_area'] = hail_dry_report_q1_area
svr_summary['dry_hail_report_q2_area'] = hail_dry_report_q2_area
svr_summary['dry_hail_report_q3_area'] = hail_dry_report_q3_area
svr_summary['dry_hail_report_q4_area'] = hail_dry_report_q4_area

svr_summary['dry_wind_report_q1_bool'] = wind_dry_report_q1_bool
svr_summary['dry_wind_report_q2_bool'] = wind_dry_report_q2_bool
svr_summary['dry_wind_report_q3_bool'] = wind_dry_report_q3_bool
svr_summary['dry_wind_report_q4_bool'] = wind_dry_report_q4_bool
svr_summary['dry_wind_report_q1_area'] = wind_dry_report_q1_area
svr_summary['dry_wind_report_q2_area'] = wind_dry_report_q2_area
svr_summary['dry_wind_report_q3_area'] = wind_dry_report_q3_area
svr_summary['dry_wind_report_q4_area'] = wind_dry_report_q4_area

svr_summary['enso_op'] = enso_op
svr_summary['enso_3mo'] = enso_3mo
svr_summary['enso_1mo'] = enso_1mo

# test = svr_summary[(svr_summary['dry_report_bool']==True) & (svr_summary['all_report_bool']==False)]
# print(test)

svr_summary.to_pickle("/home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/dryline_svr_summary.pkl")


#%%
print('Time to run: ' + str(datetime.utcnow() - start))



"""
drylines = xr.open_dataset('/raid/mhosek/era5_drylines/dryline_associated_severe_1995_2020_byenso.nc')
reports = xr.open_dataset('/raid/mhosek/era5_drylines/SGP_reports_1995_2020.nc')

#define SGP region
sgp = shapely.Polygon(((-105,35), (-105, 40), (-100,40), (-95, 40), (-95, 30), (-100, 30), (-105, 30)))
sgpx,sgpy = sgp.exterior.xy
sgpx,sgpy =m(sgpx,sgpy)

lon, lat = drylines.lons.values, drylines.lats.values
x,y = m(lon,lat)

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12, 12),frameon=True)

m.drawcoastlines(linewidth=1,color='k', ax=ax)
m.drawcountries(linewidth=1,color='k', ax=ax)
m.drawstates(linewidth=1, color='k', ax=ax)
m.drawmeridians(range(-180, 180, 5), labels = [0,0,0,1], linewidth=0.25, ax=ax, fontsize=12)
m.drawparallels(range(-10, 90, 5), labels = [1,0,0,0], linewidth=0.25, ax=ax, fontsize=12)

#plot the domains
m.plot(sgpx, sgpy, color='#0072B2', linewidth=2, ax=ax, zorder=13)

inside_sgp = (
    (lon <= -100.0) & (lon >= -105)
    & (lat >= 35) & (lat <= 40)
)

m.pcolormesh(
    x, y,
    inside_sgp.astype(int),
    shading="nearest",
    cmap=colors.ListedColormap(["#eeeeee", "#f4a6a6"]),
    vmin=0,
    vmax=1,
    edgecolors="#777777",
    linewidth=0.35,
    ax=ax,
    zorder=1,
)

# Retain the selected grid centers as a visual reference.
m.scatter(
    x[inside_sgp], y[inside_sgp],
    color="red", s=5, ax=ax, zorder=14,
)

# rep = drylines.all_reports_n.values
# rep = reports.all_reports.values
# m.pcolormesh(
#     x, y,
#     np.sum(rep,axis=0),
#     shading="nearest",
#     cmap=colors.ListedColormap(["#eeeeee", "red"]),
#     vmin=0,
#     vmax=1,
#     edgecolors="#777777",
#     linewidth=0.35,
#     ax=ax,
#     zorder=1,
# )


  
plt.savefig("test.png", format='png', dpi=150, bbox_inches='tight')   
"""