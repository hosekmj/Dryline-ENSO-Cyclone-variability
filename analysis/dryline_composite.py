#create the dryline composite maps, updated for revision

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
from scipy.stats import mannwhitneyu

from scipy import stats
from scipy import signal

from metpy.calc import geopotential_to_height
from metpy.units import units

import glob
import argparse

import paramiko
import os

import palettable
import random

start = datetime.utcnow()

#read in from input
parser = argparse.ArgumentParser()
parser.add_argument('terrain_mask',type=int,help='terrain_mask')
parser.add_argument('enso_type',type=str,help='enso_type')
parser.add_argument('year_cutoff',type=int,help='year_cutoff')
args = parser.parse_args()

terrain_mask = f"t={args.terrain_mask}"
ENSO_TYPE = args.enso_type
YEAR_CUTOFF = args.year_cutoff
FDR = True
print(f"FDR is set to {FDR}")

directory_path = f"{YOUR_FILE_PATH}/{terrain_mask}/{ENSO_TYPE}"
os.makedirs(directory_path, exist_ok=True)

llcrnrlat=28
llcrnrlon=360-107
urcrnrlat=42
urcrnrlon=360-93
lat_0=35
lon_0=360-98
latin1=25
latin2=25
m = Basemap(projection='lcc', llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
            urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,
            lat_1=latin1, lat_2=latin2, lat_0=lat_0, lon_0=lon_0,
            resolution='h', area_thresh=10000, fix_aspect=True)

dryline_xary = xr.open_dataset('/raid/mhosek/era5_drylines/dryline_cat_1940_2025_corrected.nc') #for reading in uncompressed datasets
  
#restrict to spring, 00z, only up to 2020
times = dryline_xary.time.values

keeptimes = []
for i in range(len(times)):
    if pd.to_datetime(times[i]).month >= 2 and pd.to_datetime(times[i]).month <= 6 and pd.to_datetime(times[i]).hour == 0 and pd.to_datetime(times[i]).year <= 2020:
        keeptimes.append(times[i])
dryline_xary = dryline_xary.sel(time=keeptimes)

#restrict to years on or after year cutoff
dryline_xary = dryline_xary.sel(time=dryline_xary['time.year'] >= YEAR_CUTOFF)

num_years = 2020-YEAR_CUTOFF+1 #how many years are in our sample

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

enso_df = enso_df[enso_df['year'] >= YEAR_CUTOFF]


def xary_sel_enso(xary, enso_df):
    #given the xary of drylines, and a dataframe of ENSO index by month, split the given array
    #into 3: one for El Nino, one for neutral, one for La Nina
    
    #these xarrays will already be sorted so it's only for a given month, so you only need to sort by year.
    
    times = xary.time.values
    keeptimes_la = []
    keeptimes_el = []
    keeptimes_nu = []
    for i in range(len(times)):
        year = pd.to_datetime(times[i]).year
        
        df = enso_df[enso_df['year'] == year] #should be a single row
        index = df.iloc[0][ENSO_TYPE]
        
        if index <= -0.5:
            keeptimes_la.append(times[i])
        elif index >= 0.5:
            keeptimes_el.append(times[i])
        else:
            keeptimes_nu.append(times[i])
            
    
    temp_la = xary.sel(time=keeptimes_la)
    temp_el = xary.sel(time=keeptimes_el)
    temp_nu = xary.sel(time=keeptimes_nu)

    return temp_la, temp_nu, temp_el

def short_randomization(obs_anom, all_dryline_xr, climo, enso_df, enso_phase, iterations=10000):
    #attempt to make a quicker/easier randomization test
    #reads in the xarray of all years of [ENSO cateogory], and all years of full sample
    #based on number of years of [ENSO cateogory], it randomly selects that many years from the full sample of years (with replacement)
    #creates a new dryline anomaly field relative to climo, repeat steps 3 and 4 10,000 times
    #see how frequently the randomized anomly is more/less than observed anomaly, determine statistical significance
    
    #inputs:
        #obs_anom: array of observed dryline anomaly for that given ENSO phase (lat x lon)
        #all_dryline_xr: xarray of all dryline months
        #enso_df = dataframe of enso values
        #enso_phase = string for enso phase
        #climo = dryline composite mean climatology (lat x lon)
        #iterations = the number of randomizations (will be 10,000 for final plots (takes ~15 hours to run for all months))
    
    print(f"performing {iterations} iterations")
    years = np.arange(YEAR_CUTOFF, 2021, 1)

    if enso_phase == 'la':
        enso_df = enso_df[enso_df[ENSO_TYPE] <= -0.5]
    elif enso_phase == 'el':
        enso_df = enso_df[enso_df[ENSO_TYPE] >= 0.5]
    elif enso_phase == 'nu':
        enso_df = enso_df[(enso_df[ENSO_TYPE] > -0.5) & (enso_df[ENSO_TYPE] < 0.5)]
    else:
        print("oh no")
        
    num_years = len(enso_df) #the number of years of a particular ENSO phase
    
    anom_dist = np.zeros((iterations, climo.shape[0], climo.shape[1]))*np.nan
    # random_dist = np.zeros((iterations, climo.shape[0], climo.shape[1]))*np.nan
    
    for i in range(iterations):
        
        #select the years for the subsample (with replacement), pluck out those years
        random_years = random.choices(years, k=num_years)
        
        times = all_dryline_xr.time.values
        keeptimes = []
        for t in range(len(times)):
            if pd.to_datetime(times[t]).year in random_years:
                #see how many times that year was randomly selected --> if multiple times, need to add those months to new xarray multiple times
                listcount = random_years.count(pd.to_datetime(times[t]).year)
                for c in range(listcount):
                    keeptimes.append(times[t])
                
        temp_xr = all_dryline_xr.sel(time=keeptimes)

        #take mean across time dimension to find randomized composite; find amonaly compared to climatology
        temp_xr = temp_xr.sum(dim='time')
        random_composite = temp_xr.dryline.values / num_years #put in form of # of days per month
        random_anom = random_composite - climo
        anom_dist[i, :, :] = random_anom 
        # random_dist[i,:,:] = random_composite
        
    #for each grid point, find the number of iterations where randomo climo was above (below) the observed climo
    pval = np.zeros(obs_anom.shape)*np.nan
    for i in range(len(obs_anom)):
        for j in range(len(obs_anom[i])):
            num_greater = 0
            num_less = 0
            for x in range(len(anom_dist)):
                if anom_dist[x,i,j] >= obs_anom[i,j]:
                    num_greater += 1
                if anom_dist[x,i,j] <= obs_anom[i,j]:
                    num_less += 1

            if climo[i,j] != 0: #only look for significance if the climatology is not 0
                temp_pval1 = num_greater / iterations
                temp_pval2 = num_less / iterations
                
                
                #keep the lowest pval
                if temp_pval1 < temp_pval2:
                    pval[i,j] = temp_pval1
                else:
                    pval[i,j] = temp_pval2

    return pval

def fdr_bh(pvalues):
    #use the Benjamini-Hochberg (BH) procedure to control the false discovery rate for the spatial p-values
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(pvalues.shape, np.nan)
    valid = np.isfinite(pvalues)
    p = pvalues[valid]

    if np.any((p < 0) | (p > 1)):
        raise ValueError("P-values must be between 0 and 1.")
    if p.size == 0:
        return adjusted

    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, p.size + 1)

    q_ranked = ranked * p.size / ranks
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)

    q = np.empty_like(q_ranked)
    q[order] = q_ranked
    adjusted[valid] = q
    return adjusted



def permutation_pvalue(annual_fields, labels, phase, iterations=10_000, seed=0):
    use = np.isin(labels, [phase, "nu"])
    fields = annual_fields[use]
    is_phase = labels[use] == phase

    observed = fields[is_phase].mean(axis=0) - fields[~is_phase].mean(axis=0)

    rng = np.random.default_rng(seed)
    exceedances = np.zeros(observed.shape, dtype=int)

    for _ in range(iterations):
        shuffled = rng.permutation(is_phase)
        permuted = fields[shuffled].mean(axis=0) - fields[~shuffled].mean(axis=0)
        exceedances += np.abs(permuted) >= np.abs(observed)

    return (exceedances + 1) / (iterations + 1)

def neutral_composite(m, dryline_xr, enso_df, month):
    #Produce a single-panel plot of the neutral-phase composite dryline frequency for a given month
    
    #inputs:
        #m = basemap projection
        #dryline_xr = xarray of all spring 00z drylines
        #enso_df = df of enso values for that month
        #month = int of what month it is
        #num_years = int, number of years in our sample
    #returns:
        #2-D lat/lon field of the climatology for monthly dryline frequency
    
    #set up longitude for plotting    
    lons = dryline_xr.lon.values
    lats = dryline_xr.lat.values
    x,y = np.meshgrid(lons, lats)
    x,y = m(x,y)
    
    #set up string for month, as well as clevs for colorfill/colorbar
    if month == 2:
        month_str = "February"
        month_title = "feb"
        days = 28
        clevs = np.arange(0, 2.5, 0.25)
        clevs_cbar = np.arange(0, 2.5, 0.5)
    elif month == 3:
        month_str = "March"
        month_title = "march"
        days = 31
        clevs = np.arange(0, 7.5, 0.5)
        clevs_cbar = np.arange(0, 8, 1)
    elif month == 4:
        month_str = "April"
        month_title = 'april'
        days = 30
        clevs = np.arange(0, 16, 1)
        clevs_cbar = np.arange(0, 16, 2)
    elif month == 5:
        month_str = "May"
        month_title = "may"
        days = 31
        clevs = np.arange(0, 24, 2)
        clevs_cbar = np.arange(0, 24, 2)
    elif month == 6:
        month_str = "June"
        month_title = 'june'
        days = 30
        clevs = np.arange(0, 18, 1)
        clevs_cbar = np.arange(0, 18, 2)
    

    #only keep drylines for the specific month
    keeptimes = []
    times = dryline_xr.time.values
    for i in range(len(times)):
        if pd.to_datetime(times[i]).month == month:
            keeptimes.append(times[i])
    dryline_xr = dryline_xr.sel(time=keeptimes)
        
    #restrict to just the central US to make it process faster
    #lat: 45 -- 25
    #lon: -90 -- -110
    dryline_xr = dryline_xr.sel(lat=slice(45,25), lon=slice(250,270))
    
    #split into 3 xarays -- one for each ENSO phase -- will only use neutral here
    dryline_xr_la, dryline_xr_nu, dryline_xr_el = xary_sel_enso(dryline_xr, enso_df)
    
    num_neutral_years = len(enso_df[(enso_df[ENSO_TYPE] > -0.5) & (enso_df[ENSO_TYPE] < 0.5)])
    
    #Take sum across years to make the composite, divide by number of neutral years
    dryline_xr_nu = dryline_xr_nu.sum(dim='time')
    
    field = dryline_xr_nu.dryline.values
    #set 0 to nan so it's white
    field = np.where(field == 0, np.nan, field)
    
    #multiply by number of neutral years in our sample to put in terms of number of drylines in this month per year.
    field = field / num_neutral_years
    

    #reset the dryline climo so where it is np.nan it is now back to 0, so we can do math later
    field = np.where(np.isnan(field), 0, field)
    
    return field 

feb_dryclimo = neutral_composite(m, dryline_xary, enso_df[enso_df['month']==2], 2)
march_dryclimo = neutral_composite(m, dryline_xary, enso_df[enso_df['month']==3], 3)
april_dryclimo = neutral_composite(m, dryline_xary, enso_df[enso_df['month']==4], 4)
may_dryclimo = neutral_composite(m, dryline_xary, enso_df[enso_df['month']==5], 5)
june_dryclimo = neutral_composite(m, dryline_xary, enso_df[enso_df['month']==6], 6)

print("Done making climo")


#%% The dryline composites

def enso_anomaly_composite(m, enso_df, dryline_xr, dryline_climo, month, do_stats=False):
    #Create the El and La anomaly plots and test statistical significance
    
    #inputs:
        #m = basemap projection
        #enso_df: dataframe of  enso index for that particular month
        #dryline_xr = xarray of all drylines
        #dryline_climo = 2-D array of the dryline frequency climatology for that month (avg number of dryline days for neutral phase)
        #month = int, the month we want to plot
        #num_years = int, number of years in our sample
           
    
    #only keep drylines for the specific month
    keeptimes = []
    times = dryline_xr.time.values
    for i in range(len(times)):
        if pd.to_datetime(times[i]).month == month:
            keeptimes.append(times[i])
    dryline_xr = dryline_xr.sel(time=keeptimes)
    
    #restrict to just the central US to make it process faster
    #lat: 45 -- 25
    #lon: -90 -- -110
    dryline_xr = dryline_xr.sel(lat=slice(45,25), lon=slice(250,270))
    
    #split into 3 xarays -- one for each ENSO phase
    dryline_xr_la, dryline_xr_nu, dryline_xr_el = xary_sel_enso(dryline_xr, enso_df)
            
    #Take mean across time to make the composite
    # dryline_xr_el = dryline_xr_el.mean(dim='time')
    # dryline_xr_la = dryline_xr_la.mean(dim='time')
    # dryline_xr_nu = dryline_xr_nu.mean(dim='time')
    
    dryline_xr_el = dryline_xr_el.sum(dim='time')
    dryline_xr_la = dryline_xr_la.sum(dim='time')
    dryline_xr_nu = dryline_xr_nu.sum(dim='time')
    
    #find the number of years with La, El, and neutral
    num_years_el = len(enso_df[enso_df[ENSO_TYPE] >= 0.5])
    num_years_la = len(enso_df[enso_df[ENSO_TYPE] <= -0.5])
    num_years_nu = len(enso_df[(enso_df[ENSO_TYPE] > -0.5) & (enso_df[ENSO_TYPE] < 0.5)])
    
    #calculate field anomlay
    field = dryline_xr_el.dryline.values
    #multiply by number of years in our sample to put in terms of number of drylines in this month per year.
    field = field / num_years_el
    #subtrack the climatology so we have an anomaly
    field_anomaly_el = field - dryline_climo
    
    
    # field = dryline_xr_nu.dryline.values
    # field = field / num_years_nu
    # field_anomaly_nu = field - dryline_climo
    
    field = dryline_xr_la.dryline.values
    field = field / num_years_la
    field_anomaly_la = field - dryline_climo
    
    #determine statistical significance
    if do_stats == True:
        # pvalue_la, pvalue_nu, pvalue_el = monte_carlo_climo(dryline_xr, enso_df, 'index', dryline_climo, field_anomaly_la, field_anomaly_nu, field_anomaly_el, is_enso=True)
        # dryline_xr_la, dryline_xr_nu, dryline_xr_el = xary_sel_enso(dryline_xr, enso_df)
        
        # pvalue_la = short_randomization(field_anomaly_la, dryline_xr, dryline_climo, enso_df, 'la')#, iterations=5) #reduce iterations for testing
        # pvalue_el = short_randomization(field_anomaly_el, dryline_xr, dryline_climo, enso_df, 'el')#, iterations=5)
        # pvalue_nu = short_randomization(field_anomaly_nu, dryline_xr, dryline_climo, enso_df, 'nu')

        # One monthly dryline-count field per year, ordered as (year, lat, lon).
        annual_fields = (
            dryline_xr["dryline"]
            .groupby("time.year")
            .sum(dim="time", skipna=False)
            .transpose("year", "lat", "lon")
        )

        # Match ENSO values to the annual fields by year, not row position.
        enso_by_year = enso_df.set_index("year")[ENSO_TYPE]
        if not enso_by_year.index.is_unique:
            raise ValueError("Expected exactly one ENSO value per year for this month.")

        annual_enso = enso_by_year.reindex(annual_fields["year"].values)
        if annual_enso.isna().any():
            raise ValueError("Missing ENSO values for one or more dryline years.")

        labels = np.where(
            annual_enso.to_numpy() <= -0.5,
            "la",
            np.where(annual_enso.to_numpy() >= 0.5, "el", "nu"),
        )

        pvalue_la = permutation_pvalue(
            annual_fields.values, labels, "la", iterations=10_000
        )
        pvalue_el = permutation_pvalue(
            annual_fields.values, labels, "el", iterations=10_000
        )
        
        # pvalue_la = np.zeros(dryline_climo.shape)*np.nan
        # pvalue_nu = np.zeros(dryline_climo.shape)*np.nan
        # pvalue_el = np.zeros(dryline_climo.shape)*np.nan

        ##########apply the Benjamini-Hochberg procedure to control the false discovery rate

        if FDR == True:
            #build a regional mask for just the SGP region, since that's what we're interested in
            lon_grid, lat_grid = np.meshgrid(
                annual_fields["lon"].values,
                annual_fields["lat"].values,
                )
            regional_mask = (
                (lon_grid >= 255) & (lon_grid <= 265) &
                (lat_grid >= 30) & (lat_grid <= 40)
                )

            pvalue_la = fdr_bh(np.where(regional_mask, pvalue_la, np.nan))
            pvalue_el = fdr_bh(np.where(regional_mask, pvalue_el, np.nan))

            # pvalue_la = fdr_bh(pvalue_la)
            # pvalue_el = fdr_bh(pvalue_el)
        
        print("done with stats")
    else:
        pvalue_la = np.zeros(dryline_climo.shape)*np.nan
        # pvalue_nu = np.zeros(dryline_climo.shape)*np.nan
        pvalue_el = np.zeros(dryline_climo.shape)*np.nan
    
    
    return field_anomaly_la, pvalue_la, field_anomaly_el, pvalue_el

do_stats = True
feb_la, feb_pvalue_la, feb_el, feb_pvalue_el = enso_anomaly_composite(m, enso_df[enso_df['month']==2], dryline_xary, feb_dryclimo, 2, do_stats=do_stats)
march_la, march_pvalue_la, march_el, march_pvalue_el = enso_anomaly_composite(m, enso_df[enso_df['month']==3], dryline_xary, march_dryclimo, 3, do_stats=do_stats)
april_la, april_pvalue_la, april_el, april_pvalue_el = enso_anomaly_composite(m, enso_df[enso_df['month']==4], dryline_xary, april_dryclimo, 4, do_stats=do_stats)
may_la, may_pvalue_la, may_el, may_pvalue_el = enso_anomaly_composite(m, enso_df[enso_df['month']==5], dryline_xary, may_dryclimo, 5, do_stats=do_stats)
june_la, june_pvalue_la, june_el, june_pvalue_el = enso_anomaly_composite(m, enso_df[enso_df['month']==6], dryline_xary, june_dryclimo, 6, do_stats=do_stats)


#%% Create a 12-panel plot
        
#set up longitude for plotting   
#restrict to just the central US to make it process faster
#lat: 45 -- 25
#lon: -90 -- -110
dryline_xary = dryline_xary.sel(lat=slice(45,25), lon=slice(250,270)) 
lons = dryline_xary.lon.values
lats = dryline_xary.lat.values
x,y = np.meshgrid(lons, lats)
x,y = m(x,y)

#location of textbox
textx, texty = m(-75, 22.5)

#define SGP region
sgp = shapely.Polygon(((-105,35), (-105, 40), (-95, 40), (-95, 30), (-105, 30)))
sgpx,sgpy = sgp.exterior.xy
sgpx,sgpy =m(sgpx,sgpy)

#set up anomaly clevs
# clevs = np.arange(-2.125, 2.25, 0.25)
# clevs_cbar = np.arange(-2, 2.5, 0.5)
clevs = np.arange(-2.625, 2.75, 0.25)
clevs_cbar = np.arange(-2.5, 3.0, 0.5)
#set cmap
nlevs = len(clevs) - 1
cmap = plt.get_cmap(name = 'bwr', lut=nlevs)

clevs_cont=[1,3,5]
# clevs_cont=[0.5,1,3,5]

######################################## the plot
fig,((ax1,ax2), (ax3,ax4), (ax5,ax6), (ax7,ax8), (ax9,ax10)) = plt.subplots(nrows=5,ncols=2,figsize=(36, 36),frameon=True)
axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10]
for ax in axes:
    m.drawcoastlines(linewidth=1,color='k', ax=ax)
    m.drawcountries(linewidth=1,color='k', ax=ax)
    m.drawstates(linewidth=1, color='k', ax=ax)
    if ax == ax10 or ax == ax9:
        m.drawmeridians(range(-180, 180, 5), labels = [0,0,0,1], linewidth=0.25, ax=ax, fontsize=18)
    else:
        m.drawmeridians(range(-180, 180, 5), labels = [0,0,0,0], linewidth=0.25, ax=ax, fontsize=12)
    if ax == ax1 or ax == ax3 or ax == ax5 or ax == ax7 or ax == ax9:
        m.drawparallels(range(-10, 90, 5), labels = [1,0,0,0], linewidth=0.25, ax=ax, fontsize=18)
    else:
        m.drawparallels(range(-10, 90, 5), labels = [0,0,0,0], linewidth=0.25, ax=ax, fontsize=12)
    
    m.plot(sgpx, sgpy, color='darkgreen', linewidth=5, ax=ax, zorder=13)
    
############################################################## February
ax1.set_ylabel("February", fontweight='bold', fontsize=36, labelpad=50)
clevs_neutral = np.arange(0, 8.25, 0.25)
clevs_neutral_cbar = np.arange(0, 10, 2)
    
#### El Nino on left    
plot_forcbar = m.contourf(x,y,feb_el.T, clevs, cmap=cmap, extend='both',ax=ax1)
#stipple where significant
stipple_el = np.where(feb_pvalue_el < 0.05, 1, np.nan)
m.scatter(x,y, c=stipple_el, s=10, marker='o', cmap='Wistia', edgecolor='black', linewidth=0.5, zorder=11, ax=ax1)
cont_plot = m.contour(x,y, feb_dryclimo.T, clevs_cont, colors='black', linewidths=3, ax=ax1,zorder=14)
labels = ax1.clabel(cont_plot, fontsize=16, zorder=14)
for label in labels:
    label.set_bbox(dict(facecolor='white', edgecolor='black'))

ax1.set_title("El Ni\u00F1o", fontsize=28)
   
### La Nina in right  
plot = m.contourf(x,y,feb_la.T, clevs, cmap=cmap, extend='both',ax=ax2)
#stipple where significant
stipple_la = np.where(feb_pvalue_la < 0.05, 1, np.nan)
m.scatter(x,y, c=stipple_la, s=10, marker='o', cmap='Wistia', edgecolor='black', linewidth=0.5, zorder=11, ax=ax2)
cont_plot = m.contour(x,y, feb_dryclimo.T, clevs_cont, colors='black', linewidths=3, ax=ax2,zorder=14)
labels = ax2.clabel(cont_plot, fontsize=16, zorder=14)
for label in labels:
    label.set_bbox(dict(facecolor='white', edgecolor='black'))

ax2.set_title("La Ni\u00F1a", fontsize=28)


######################################################## March
ax3.set_ylabel("March", fontweight='bold', fontsize=36, labelpad=50)
    
#### El Nino on top    
plot = m.contourf(x,y,march_el.T, clevs, cmap=cmap, extend='both',ax=ax3)
#stipple where significant
stipple_el = np.where(march_pvalue_el < 0.05, 1, np.nan)
m.scatter(x,y, c=stipple_el, s=10, marker='o', cmap='Wistia', edgecolor='black', linewidth=0.5, zorder=11, ax=ax3)
   
### La Nina in Middle  
plot = m.contourf(x,y,march_la.T, clevs, cmap=cmap, extend='both',ax=ax4)
#stipple where significant
stipple_la = np.where(march_pvalue_la < 0.05, 1, np.nan)
m.scatter(x,y, c=stipple_la, s=10, marker='o', cmap='Wistia', edgecolor='black', linewidth=0.5, zorder=11, ax=ax4)

##############################Neutral Composite
cont_plot = m.contour(x,y, march_dryclimo.T, clevs_cont, colors='black', linewidths=3, ax=ax3,zorder=14)
labels = ax3.clabel(cont_plot, fontsize=16, zorder=14)
for label in labels:
    label.set_bbox(dict(facecolor='white', edgecolor='black'))

cont_plot = m.contour(x,y, march_dryclimo.T, clevs_cont, colors='black', linewidths=3, ax=ax4,zorder=14)
labels = ax4.clabel(cont_plot, fontsize=16, zorder=14)
for label in labels:
    label.set_bbox(dict(facecolor='white', edgecolor='black'))


############################################################### April
ax5.set_ylabel("April", fontweight='bold', fontsize=36, labelpad=50)
    
#### El Nino on top    
plot_forcbar = m.contourf(x,y,april_el.T, clevs, cmap=cmap, extend='both',ax=ax5)
#stipple where significant
stipple_el = np.where(april_pvalue_el < 0.05, 1, np.nan)
m.scatter(x,y, c=stipple_el, s=10, marker='o', cmap='Wistia', edgecolor='black', linewidth=0.5, zorder=11, ax=ax5)

   
### La Nina in Middle  
plot = m.contourf(x,y,april_la.T, clevs, cmap=cmap, extend='both',ax=ax6)
#stipple where significant
stipple_la = np.where(april_pvalue_la < 0.05, 1, np.nan)
m.scatter(x,y, c=stipple_la, s=10, marker='o', cmap='Wistia', edgecolor='black', linewidth=0.5, zorder=11, ax=ax6)



##############################Neutral Composite
#set
cont_plot = m.contour(x,y, april_dryclimo.T, clevs_cont, colors='black', linewidths=3, ax=ax5,zorder=14)
labels = ax5.clabel(cont_plot, fontsize=16, zorder=14)
for label in labels:
    label.set_bbox(dict(facecolor='white', edgecolor='black'))

cont_plot = m.contour(x,y, april_dryclimo.T, clevs_cont, colors='black', linewidths=3, ax=ax6,zorder=14)
labels = ax6.clabel(cont_plot, fontsize=16, zorder=14)
for label in labels:
    label.set_bbox(dict(facecolor='white', edgecolor='black'))


############################################################### May
ax7.set_ylabel("May", fontweight='bold', fontsize=36, labelpad=50)
    
#### El Nino on top    
plot_forcbar = m.contourf(x,y,may_el.T, clevs, cmap=cmap, extend='both',ax=ax7)
#stipple where significant
stipple_el = np.where(may_pvalue_el < 0.05, 1, np.nan)
m.scatter(x,y, c=stipple_el, s=10, marker='o', cmap='Wistia', edgecolor='black', linewidth=0.5, zorder=11, ax=ax7)

   
### La Nina in Middle  
plot = m.contourf(x,y,may_la.T, clevs, cmap=cmap, extend='both',ax=ax8)
#stipple where significant
stipple_la = np.where(may_pvalue_la < 0.05, 1, np.nan)
m.scatter(x,y, c=stipple_la, s=10, marker='o', cmap='Wistia', edgecolor='black', linewidth=0.5, zorder=11, ax=ax8)


##############################Neutral Composite 
for ax in [ax7, ax8]:
    automatic_contours = m.contour(
        x, y, may_dryclimo.T,
        levels=[1, 5],
        colors="black", linewidths=3, ax=ax, zorder=14,
    )
    automatic_labels = ax.clabel(
        automatic_contours, fontsize=16, zorder=14,
    )

    manual_contour = m.contour(
        x, y, may_dryclimo.T,
        levels=[3],
        colors="black", linewidths=3, ax=ax, zorder=14,
    )
    manual_labels = ax.clabel(
        manual_contour,
        manual=[m(-102, 36)],
        fontsize=16, zorder=14,
    )

    for label in [*automatic_labels, *manual_labels]:
        label.set_bbox(dict(facecolor="white", edgecolor="black"))



############################################################### June
ax9.set_ylabel("June", fontweight='bold', fontsize=36, labelpad=50)
    
#### El Nino on top    
plot_forcbar = m.contourf(x,y,june_el.T, clevs, cmap=cmap, extend='both',ax=ax9)
#stipple where significant
stipple_el = np.where(june_pvalue_el < 0.05, 1, np.nan)
m.scatter(x,y, c=stipple_el, s=10, marker='o', cmap='Wistia', edgecolor='black', linewidth=0.5, zorder=11, ax=ax9)

   
### La Nina in Middle  
plot = m.contourf(x,y,june_la.T, clevs, cmap=cmap, extend='both',ax=ax10)
#stipple where significant
stipple_la = np.where(june_pvalue_la < 0.05, 1, np.nan)
m.scatter(x,y, c=stipple_la, s=10, marker='o', cmap='Wistia', edgecolor='black', linewidth=0.5, zorder=11, ax=ax10)


##############################Neutral Composite 
for ax in [ax9, ax10]:
    automatic_contours = m.contour(
        x, y, june_dryclimo.T,
        levels=[5],
        colors="black", linewidths=3, ax=ax, zorder=14,
    )
    automatic_labels = ax.clabel(
        automatic_contours, fontsize=16, zorder=14,
    )

    manual_contour = m.contour(
        x, y, june_dryclimo.T,
        levels=[3],
        colors="black", linewidths=3, ax=ax, zorder=14,
    )
    manual_labels = ax.clabel(
        manual_contour,
        manual=[m(-102, 36)],
        fontsize=16, zorder=14,
    )

    manual_contour = m.contour(
        x, y, june_dryclimo.T,
        levels=[1],
        colors="black", linewidths=3, ax=ax, zorder=14,
    )
    manual_labels += ax.clabel(
        manual_contour,
        manual=[m(-101, 35)],
        fontsize=16, zorder=14,
    )

    for label in [*automatic_labels, *manual_labels]:
        label.set_bbox(dict(facecolor="white", edgecolor="black"))


################## Set up colorbar for anomalies

textxlab = -107.45 #-107.5
textylab = 40.5 #41
textlabels = ['a)', 'b)' , 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 'k)', 'l)']
for ax, label in zip(axes, textlabels):
    ax.text(m(textxlab, textylab)[0], m(textxlab, textylab)[1], label, fontsize=36, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black'), zorder=14)

fig.subplots_adjust(
    left=0.12,
    right=0.98,
    bottom=0.12,
    top=0.88,
    wspace=0.025,
    hspace=0.025,
)

# Match the figure width to the fixed aspect ratio of the maps.
slot = ax1.get_position(original=True)
figure_height = fig.get_figheight()
figure_width = (
    figure_height * slot.height / (slot.width * ax1.get_data_ratio())
)
fig.set_size_inches(figure_width, figure_height)

fig.suptitle("Monthly Dryline Frequency Composites", y=0.91, fontsize=36)

# Resolve the map positions before measuring their edges.
fig.canvas.draw()
left_position = ax9.get_position()
right_position = ax10.get_position()

cbar_ax = fig.add_axes([
    left_position.x0,
    min(left_position.y0, right_position.y0) - 0.025,
    right_position.x1 - left_position.x0,
    0.008,
])

cbar = fig.colorbar(
    plot_forcbar,
    cax=cbar_ax,
    orientation="horizontal",
    extendfrac="auto",
    extendrect=False,
)
cbar.set_label("Anomaly (days)", fontsize=20)
cbar.ax.tick_params(labelsize=18)
cbar.set_ticks(clevs_cbar)

if FDR == True:
    plt.savefig(f"{directory_path}/monthly_dryline_composites_{YEAR_CUTOFF}_FDR.png", format='png', dpi=200, bbox_inches='tight')
else:
    plt.savefig(f"{directory_path}/monthly_dryline_composites_{YEAR_CUTOFF}.png", format='png', dpi=200, bbox_inches='tight')


#%%
dryline_xary.close()

print('Time to run: ' + str(datetime.utcnow() - start))
