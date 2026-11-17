#create the cyclone area composite maps, updated for revision
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

# llcrnrlat=28
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

llcrnrlat=28
llcrnrlon=360-109
urcrnrlat=49
urcrnrlon=360-93
lat_0=35
lon_0=360-98
latin1=25
latin2=25
m = Basemap(projection='lcc', llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
            urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,
            lat_1=latin1, lat_2=latin2, lat_0=lat_0, lon_0=lon_0,
            resolution='h', area_thresh=10000, fix_aspect=True)

#read in from input
parser = argparse.ArgumentParser()
parser.add_argument('terrain_mask',type=int,help='terrain_mask')
parser.add_argument('enso_type',type=str,help='enso_type')
parser.add_argument('year_cutoff',type=int,help='year_cutoff')
parser.add_argument('domain_type',type=str,help='domain_type', nargs='?', default=None)

args = parser.parse_args()

terrain_mask = f"t={args.terrain_mask}"
ENSO_TYPE = args.enso_type
YEAR_CUTOFF = args.year_cutoff
domain_type = args.domain_type
FDR = True
print(f"FDR is set to {FDR}")

if domain_type == "None":
    domain_type = None

if domain_type is not None:
    directory_path = f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/{terrain_mask}/{ENSO_TYPE}/{domain_type}"
else:
    directory_path = f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/{terrain_mask}/{ENSO_TYPE}"
os.makedirs(directory_path, exist_ok=True)

era_path = "/raid/mhosek/cytrack/ERA5_data" #path to ERA5 files

if domain_type is not None:
    cyclones_df = pd.read_pickle(f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/all_cyclones_df_{terrain_mask}_{domain_type}.pkl")
else:
    cyclones_df = pd.read_pickle(f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/all_cyclones_df_{terrain_mask}.pkl")

#keep only 00z times
def keep_00z(row):
    cols = ['time', 'lat', 'lon', 'mslp', 'maxwind', 'outermost_mslp', 'size', 'in_lee_domain', 'outermost_manual_fill',
            'coords', 'approx_area', 'any_within_SGP', 'dryline_within']
    keep_positions = [position for position, time in enumerate(row['time']) if pd.Timestamp(time).hour == 0]

    orig_len = len(row['time'])

    for col in cols:
        values = row[col]
        if len(values) != orig_len:
            raise ValueError(f"Column {col} does not match length of time column")
        row[col] = [values[i] for i in keep_positions]
    return row

cyclones_df = cyclones_df.apply(keep_00z, axis=1)

#only keep times where outermost closed isobar within SGP
def keep_outermost_within_sgp(row):
    cols = ['time', 'lat', 'lon', 'mslp', 'maxwind', 'outermost_mslp', 'size', 'in_lee_domain', 'outermost_manual_fill',
            'coords', 'approx_area', 'any_within_SGP', 'dryline_within']
    keep_positions = [position for position, within_sgp in enumerate(row['any_within_SGP']) if within_sgp]

    orig_len = len(row['any_within_SGP'])

    for col in cols:
        values = row[col]
        if len(values) != orig_len:
            raise ValueError(f"Column {col} does not match length of time column")
        row[col] = [values[i] for i in keep_positions]
    return row

cyclones_df = cyclones_df.apply(keep_outermost_within_sgp, axis=1)

#remove all rows with no timesteps remaining following that
cyclones_df = cyclones_df[cyclones_df['time'].map(len) > 0]


#remove all years prior to the year cutoff
def remove_cutoff_years(row):

    cols = ['time', 'lat', 'lon', 'mslp', 'maxwind', 'outermost_mslp', 'size', 'in_lee_domain', 'outermost_manual_fill',
                'coords', 'approx_area', 'any_within_SGP', 'dryline_within']
    keep_positions = [position for position, time in enumerate(row['time']) if pd.Timestamp(time).year >= YEAR_CUTOFF]

    orig_len = len(row['time'])
    
    for col in cols:
        values = row[col]
        if len(values) != orig_len:
            raise ValueError(f"Column {col} does not match length of time column")
        row[col] = [values[i] for i in keep_positions]
    return row

cyclones_df = cyclones_df.apply(remove_cutoff_years, axis=1)



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

def keep_for_month(row, month):
    #only keep rows for a specified month
    cols = ['time', 'coords']
    keep_positions = [position for position, time in enumerate(row["time"]) if pd.to_datetime(time).month == month]

    for col in cols:
        values = row[col]
        row[col] = [values[position] for position in keep_positions]

    return row

def short_randomization(obs_anom, cydf, climo, enso_df, enso_phase, iterations=10000):
    #attempt to make a quicker/easier randomization test
    #reads in the xarray of all years of [ENSO cateogory], and all years of full sample
    #based on number of years of [ENSO cateogory], it randomly selects that many years from the full sample of years (with replacement)
    #creates a new dryline anomaly field relative to climo, repeat steps 3 and 4 10,000 times
    #see how frequently the randomized anomly is more/less than observed anomaly, determine statistical significance
    
    #inputs:
        #obs_anom: array of observed dryline anomaly for that given ENSO phase (lat x lon)
        #cyclones_df: dataframe of all dryline months
        #enso_df = dataframe of enso values
        #enso_phase = string for enso phase
        #climo = dryline composite mean climatology (lat x lon)
        #iterations = the number of randomizations (will be 10,000 for final plots (takes ~15 hours to run for all months))
    
    print(f"performing {iterations} iterations")
    years = np.arange(YEAR_CUTOFF, 2021, 1)

    if enso_phase == 'la':
        enso_df = enso_df[enso_df[ENSO_TYPE] == -1]
    elif enso_phase == 'el':
        enso_df = enso_df[enso_df[ENSO_TYPE] == 1]
    elif enso_phase == 'nu':
        enso_df = enso_df[enso_df[ENSO_TYPE] == 0]
    else:
        print("oh no")
        
    num_years = len(enso_df) #the number of years of a particular ENSO phase
    
    anom_dist = np.zeros((iterations, climo.shape[0], climo.shape[1]))*np.nan
    # random_dist = np.zeros((iterations, climo.shape[0], climo.shape[1]))*np.nan
    
    for i in range(iterations):
        
        #select the years for the subsample (with replacement), pluck out those years
        random_years = random.choices(years, k=num_years)
    
        counts = np.zeros(cydf['coords'].iloc[0][0][2].shape, dtype=int)
        for times, coords in zip(cydf['time'], cydf['coords']):
            for t, (_,_,mask) in zip(times, coords):
                if pd.to_datetime(t).year in random_years:
                    counts += mask.astype(int)

        #divide by number of years to find the randomized composite
        random_composite = counts / num_years

        #find anomlay
        random_anom = random_composite - climo
        anom_dist[i, :, :] = random_anom 

        
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


def neutral_composite(cyclone_df, enso_df, month):
    
    #inputs:
        #m = basemap projection
        #dryline_df = df of all spring 00z drylines
        #enso_df = df of enso values for that month
        #month = int of what month it is
        #num_years = int, number of years in our sample
    #returns:
        #2-D lat/lon field of the neutral phase climatology for monthly dryline frequency

    #only keep cyclones for a specific month
    cydf = cyclone_df.apply(keep_for_month, month=month, axis=1)
    cydf = cydf[cydf['time'].map(len) > 0]

    #only keep neutral years
    nu_years = set(enso_df.loc[enso_df[ENSO_TYPE] == 0, 'year'])
    num_neutral_years = len(nu_years)

    counts = np.zeros(cydf['coords'].iloc[0][0][2].shape, dtype=int)
    for times, coords in zip(cydf['time'], cydf['coords']):
        for t, (_,_,mask) in zip(times, coords):
            if pd.to_datetime(t).year in nu_years:
                counts += mask.astype(int)

    #divide ny number of neutral years
    counts = counts / num_neutral_years
    field = counts
    return field 

feb_climo = neutral_composite(cyclones_df, enso_df[enso_df['month']==2], 2)
march_climo = neutral_composite(cyclones_df, enso_df[enso_df['month']==3], 3)
april_climo = neutral_composite(cyclones_df, enso_df[enso_df['month']==4], 4)
may_climo = neutral_composite(cyclones_df, enso_df[enso_df['month']==5], 5)
june_climo = neutral_composite(cyclones_df, enso_df[enso_df['month']==6], 6)

print("Done making climo")


#%% The dryline composites

def enso_anomaly_composite(enso_df, cyclone_df, climo, month, do_stats=False):
    #Create the El and La anomaly plots and test statistical significance
    
    #inputs:
        #enso_df: dataframe of  enso index for that particular month
        #enso_type: str, the type of ENSO index to use ('op_enso', '3mo_enso', '1mo_enso')
        #cyclone_df = df of all cyclones for that month
        #climo = 2-D array of the cyclone frequency climatology for that month (avg number of cyclone days for neutral phase)
        #month = int, the month we want to plot
        #num_years = int, number of years in our sample
           
    

    #only keep cyclones for a specific month
    cydf = cyclone_df.apply(keep_for_month, month=month, axis=1)
    cydf = cydf[cydf['time'].map(len) > 0]

    #El Nino years
    el_years = set(enso_df.loc[enso_df[ENSO_TYPE] == 1, 'year'])
    num_el_years = len(el_years)

    counts_el = np.zeros(cydf['coords'].iloc[0][0][2].shape, dtype=int)
    for times, coords in zip(cydf['time'], cydf['coords']):
        for t, (_,_,mask) in zip(times, coords):
            if pd.to_datetime(t).year in el_years:
                counts_el += mask.astype(int)

    field_el = counts_el / num_el_years

    #la Nina years
    la_years = set(enso_df.loc[enso_df[ENSO_TYPE] == -1, 'year'])
    num_la_years = len(la_years)

    counts_la = np.zeros(cydf['coords'].iloc[0][0][2].shape, dtype=int)
    for times, coords in zip(cydf['time'], cydf['coords']):
        for t, (_,_,mask) in zip(times, coords):
            if pd.to_datetime(t).year in la_years:
                counts_la += mask.astype(int)

    field_la = counts_la / num_la_years

    #calculate the anomaly compared to neutral
    field_anomaly_el = field_el - climo
    field_anomaly_la = field_la - climo
    
    
    #determine statistical significance
    if do_stats == True:
        # pvalue_la, pvalue_nu, pvalue_el = monte_carlo_climo(dryline_xr, enso_df, 'index', dryline_climo, field_anomaly_la, field_anomaly_nu, field_anomaly_el, is_enso=True)
        # dryline_xr_la, dryline_xr_nu, dryline_xr_el = xary_sel_enso(dryline_xr, enso_df)
        
        # pvalue_la = short_randomization(field_anomaly_la, cydf, climo, enso_df, 'la')#, iterations=5) #reduce iterations for testing
        # pvalue_el = short_randomization(field_anomaly_el, cydf, climo, enso_df, 'el')#, iterations=5)
        # pvalue_nu = short_randomization(field_anomaly_nu, dryline_xr, dryline_climo, enso_df, 'nu')

        years = np.arange(YEAR_CUTOFF, 2021)
        year_to_index = {year: i for i, year in enumerate(years)}

        enso_by_year = enso_df.set_index("year")[ENSO_TYPE]
        if not enso_by_year.index.is_unique:
            raise ValueError("Expected exactly one ENSO value per year.")

        annual_enso = enso_by_year.reindex(years)
        if annual_enso.isna().any():
            raise ValueError("Missing ENSO values for one or more analysis years.")
        if not annual_enso.isin([-1, 0, 1]).all():
            raise ValueError("Expected ENSO categories of -1, 0, or 1.")

        labels = np.where(
            annual_enso.to_numpy() == -1,
            "la",
            np.where(annual_enso.to_numpy() == 1, "el", "nu"),
        )

        annual_fields = np.zeros(
            (len(years), *climo.shape), dtype=np.int64
        )

        for times, coords in zip(cydf["time"], cydf["coords"]):
            if len(times) != len(coords):
                raise ValueError("Cyclone timestamps and coordinates do not match.")

            for t, (_, _, mask) in zip(times, coords):
                year = pd.Timestamp(t).year
                if year not in year_to_index:
                    continue

                mask = np.asarray(mask)
                if mask.shape != climo.shape or not np.isfinite(mask).all():
                    raise ValueError("Invalid cyclone mask shape or missing values.")

                annual_fields[year_to_index[year]] += mask.astype(np.int64)

        for phase in ("la", "nu", "el"):
            if not np.any(labels == phase):
                raise ValueError(f"No years available for ENSO category {phase!r}.")

        # Check that the tested contrasts match the plotted anomalies.
        neutral_mean = annual_fields[labels == "nu"].mean(axis=0)
        np.testing.assert_allclose(
            annual_fields[labels == "la"].mean(axis=0) - neutral_mean,
            field_anomaly_la,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            annual_fields[labels == "el"].mean(axis=0) - neutral_mean,
            field_anomaly_el,
            atol=1e-12,
        )

        pvalue_la = permutation_pvalue(
            annual_fields, labels, "la", iterations=10_000
        )
        pvalue_el = permutation_pvalue(
            annual_fields, labels, "el", iterations=10_000
        )

        #apply the Benjamini-Hochberg procedure to control the false discovery rate
        if FDR == True:
            #build a regional mask for just the SGP region, since that's what we're interested in
            with xr.open_dataset(era_path + "/mod_uvmslp_ERA5_19790509_00.nc") as ds:
                lon_values = ds["longitude"].values
                lat_values = ds["latitude"].values

            lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
            lon_grid = (lon_grid + 180) % 360 - 180

            regional_mask = (
                (lon_grid >= -105) & (lon_grid <= -95) &
                (lat_grid >= 30) & (lat_grid <= 40)
            )

            pvalue_la = fdr_bh(np.where(regional_mask, pvalue_la, np.nan))
            pvalue_el = fdr_bh(np.where(regional_mask, pvalue_el, np.nan))

            # pvalue_la = fdr_bh(pvalue_la)
            # pvalue_el = fdr_bh(pvalue_el)

        
        print("done with stats")
    else:
        pvalue_la = np.zeros(field_anomaly_la.shape)*np.nan
        # pvalue_nu = np.zeros(dryline_climo.shape)*np.nan
        pvalue_el = np.zeros(field_anomaly_el.shape)*np.nan
    
    
    return field_anomaly_la, pvalue_la, field_anomaly_el, pvalue_el

do_stats = True
feb_la, feb_pvalue_la, feb_el, feb_pvalue_el = enso_anomaly_composite(enso_df[enso_df['month']==2], cyclones_df, feb_climo, 2, do_stats=do_stats)
march_la, march_pvalue_la, march_el, march_pvalue_el = enso_anomaly_composite(enso_df[enso_df['month']==3], cyclones_df, march_climo, 3, do_stats=do_stats)
april_la, april_pvalue_la, april_el, april_pvalue_el = enso_anomaly_composite(enso_df[enso_df['month']==4], cyclones_df, april_climo, 4, do_stats=do_stats)
may_la, may_pvalue_la, may_el, may_pvalue_el = enso_anomaly_composite(enso_df[enso_df['month']==5], cyclones_df, may_climo, 5, do_stats=do_stats)
june_la, june_pvalue_la, june_el, june_pvalue_el = enso_anomaly_composite(enso_df[enso_df['month']==6], cyclones_df, june_climo, 6, do_stats=do_stats)


#%% Create a 10-panel plot
        
#set up longitude for plotting   
for_latlon = xr.open_dataset(era_path + "/mod_uvmslp_ERA5_19790509_00.nc")
lon, lat = for_latlon['longitude'].values, for_latlon['latitude'].values
lon, lat = np.meshgrid(lon, lat)
x, y = m(lon, lat)

#location of textbox
textx, texty = m(-75, 22.5)

#define SGP region
sgp = shapely.Polygon(((-105,35), (-105, 40), (-95, 40), (-95, 30), (-105, 30)))
sgpx,sgpy = sgp.exterior.xy
sgpx,sgpy =m(sgpx,sgpy)

#define lee cyclone region
lc = shapely.Polygon(((-106,32), (-106, 41), (-109, 43), (-109, 47.5), (-106, 47.5), (-100, 47.5), (-100, 47.5), (-98, 41), (-98, 32), (-106,32)))
lcx, lcy = lc.exterior.xy
lcx, lcy = m(lcx, lcy)


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

    m.plot(lcx, lcy, color='chocolate', linewidth=5, ax=ax, zorder=12)
    
############################################################## February
ax1.set_ylabel("February", fontweight='bold', fontsize=36, labelpad=50)
clevs_neutral = np.arange(0, 8.25, 0.25)
clevs_neutral_cbar = np.arange(0, 10, 2)
    
#### El Nino on left    
plot_forcbar = m.contourf(x,y,feb_el, clevs, cmap=cmap, extend='both',ax=ax1)
#stipple where significant
stipple_el = np.where(feb_pvalue_el < 0.05, 1, np.nan)
m.scatter(x[::2, ::2],y[::2, ::2], c=stipple_el[::2, ::2], s=10, marker='o', cmap='Wistia', edgecolor='black', linewidth=0.5, zorder=11, ax=ax1)

ax1.set_title("El Ni\u00F1o", fontsize=28)
   
### La Nina in right  
plot = m.contourf(x,y,feb_la, clevs, cmap=cmap, extend='both',ax=ax2)
#stipple where significant
stipple_la = np.where(feb_pvalue_la < 0.05, 1, np.nan)
m.scatter(x[::2, ::2],y[::2, ::2], c=stipple_la[::2, ::2], s=10, marker='o', cmap='Wistia', edgecolor='black', linewidth=0.5, zorder=11, ax=ax2)

ax2.set_title("La Ni\u00F1a", fontsize=28)

#neutral contour
for ax in [ax1, ax2]:
    manual_contour = m.contour(x,y,
        feb_climo,
        levels=[1],
        colors="black", linewidths=3, ax=ax, zorder=14,
    )
    manual_labels = ax.clabel(
        manual_contour,
        manual=[m(-102, 43)],
        fontsize=16, zorder=14,
    )

    for label in [*manual_labels]:
        label.set_bbox(dict(facecolor="white", edgecolor="black"))


######################################################## March
ax3.set_ylabel("March", fontweight='bold', fontsize=36, labelpad=50)
    
#### El Nino on top    
plot = m.contourf(x,y,march_el, clevs, cmap=cmap, extend='both',ax=ax3)
#stipple where significant
stipple_el = np.where(march_pvalue_el < 0.05, 1, np.nan)
m.scatter(x[::2, ::2],y[::2, ::2], c=stipple_el[::2, ::2], s=10, marker='o', cmap='Wistia', edgecolor='black', linewidth=0.5, zorder=11, ax=ax3)
   
### La Nina in Middle  
plot = m.contourf(x,y,march_la, clevs, cmap=cmap, extend='both',ax=ax4)
#stipple where significant
stipple_la = np.where(march_pvalue_la < 0.05, 1, np.nan)
m.scatter(x[::2, ::2],y[::2, ::2], c=stipple_la[::2, ::2], s=10, marker='o', cmap='Wistia', edgecolor='black', linewidth=0.5, zorder=11, ax=ax4)

##############################Neutral Composite
for ax in [ax3, ax4]:
    manual_contour = m.contour(x,y,
        march_climo,
        levels=[1],
        colors="black", linewidths=3, ax=ax, zorder=14,
    )
    manual_labels = ax.clabel(
        manual_contour,
        manual=[m(-102, 45)],
        fontsize=16, zorder=14,
    )

    manual_contour = m.contour(x,y,
        march_climo,
        levels=[3],
        colors="black", linewidths=3, ax=ax, zorder=14,
    )
    manual_labels += ax.clabel(
        manual_contour,
        manual=[m(-100, 34)],
        fontsize=16, zorder=14,
    )

    for label in [*manual_labels]:
        label.set_bbox(dict(facecolor="white", edgecolor="black"))


############################################################### April
ax5.set_ylabel("April", fontweight='bold', fontsize=36, labelpad=50)
    
#### El Nino on top    
plot_forcbar = m.contourf(x,y,april_el, clevs, cmap=cmap, extend='both',ax=ax5)
#stipple where significant
stipple_el = np.where(april_pvalue_el < 0.05, 1, np.nan)
m.scatter(x[::2, ::2],y[::2, ::2], c=stipple_el[::2, ::2], s=10, marker='o', cmap='Wistia', edgecolor='black', linewidth=0.5, zorder=11, ax=ax5)

   
### La Nina in Middle  
plot = m.contourf(x,y,april_la, clevs, cmap=cmap, extend='both',ax=ax6)
#stipple where significant
stipple_la = np.where(april_pvalue_la < 0.05, 1, np.nan)
m.scatter(x[::2, ::2],y[::2, ::2], c=stipple_la[::2, ::2], s=10, marker='o', cmap='Wistia', edgecolor='black', linewidth=0.5, zorder=11, ax=ax6)



##############################Neutral Composite
for ax in [ax5, ax6]:
    manual_contour = m.contour(x,y,
        april_climo,
        levels=[1],
        colors="black", linewidths=3, ax=ax, zorder=14,
    )
    manual_labels = ax.clabel(
        manual_contour,
        manual=[m(-107, 46)],
        fontsize=16, zorder=14,
    )

    manual_contour = m.contour(x,y,
        april_climo,
        levels=[3],
        colors="black", linewidths=3, ax=ax, zorder=14,
    )
    manual_labels += ax.clabel(
        manual_contour,
        manual=[m(-98, 42)],
        fontsize=16, zorder=14,
    )

    manual_contour = m.contour(x,y,
        april_climo,
        levels=[5],
        colors="black", linewidths=3, ax=ax, zorder=14,
    )
    if domain_type != "only_centroid":
        manual_labels += ax.clabel(
            manual_contour,
            manual=[m(-103, 35)],
            fontsize=16, zorder=14,
        )


    for label in [*manual_labels]:
        label.set_bbox(dict(facecolor="white", edgecolor="black"))

############################################################### May
ax7.set_ylabel("May", fontweight='bold', fontsize=36, labelpad=50)
    
#### El Nino on top    
plot_forcbar = m.contourf(x,y,may_el, clevs, cmap=cmap, extend='both',ax=ax7)
#stipple where significant
stipple_el = np.where(may_pvalue_el < 0.05, 1, np.nan)
m.scatter(x[::2, ::2],y[::2, ::2], c=stipple_el[::2, ::2], s=10, marker='o', cmap='Wistia', edgecolor='black', linewidth=0.5, zorder=11, ax=ax7)

   
### La Nina in Middle  
plot = m.contourf(x,y,may_la, clevs, cmap=cmap, extend='both',ax=ax8)
#stipple where significant
stipple_la = np.where(may_pvalue_la < 0.05, 1, np.nan)
m.scatter(x[::2, ::2],y[::2, ::2], c=stipple_la[::2, ::2], s=10, marker='o', cmap='Wistia', edgecolor='black', linewidth=0.5, zorder=11, ax=ax8)


##############################Neutral Composite
if domain_type != "only_centroid":
    for ax in [ax7, ax8]:
        manual_contour = m.contour(x,y,
            may_climo,
            levels=[1],
            colors="black", linewidths=3, ax=ax, zorder=14,
        )
        manual_labels = ax.clabel(
            manual_contour,
            manual=[m(-99, 48)],
            fontsize=16, zorder=14,
        )

        manual_contour = m.contour(x,y,
            may_climo,
            levels=[3],
            colors="black", linewidths=3, ax=ax, zorder=14,
        )
        manual_labels += ax.clabel(
            manual_contour,
            manual=[m(-104, 42)],
            fontsize=16, zorder=14,
        )

        manual_contour = m.contour(x,y,
            may_climo,
            levels=[5],
            colors="black", linewidths=3, ax=ax, zorder=14,
        )
        manual_labels += ax.clabel(
            manual_contour,
            manual=[m(-104, 37)],
            fontsize=16, zorder=14,
        )


        for label in [*manual_labels]:
            label.set_bbox(dict(facecolor="white", edgecolor="black"))



############################################################### June
ax9.set_ylabel("June", fontweight='bold', fontsize=36, labelpad=50)
    
#### El Nino on top    
plot_forcbar = m.contourf(x,y,june_el, clevs, cmap=cmap, extend='both',ax=ax9)
#stipple where significant
stipple_el = np.where(june_pvalue_el < 0.05, 1, np.nan)
m.scatter(x[::2, ::2],y[::2, ::2], c=stipple_el[::2, ::2], s=10, marker='o', cmap='Wistia', edgecolor='black', linewidth=0.5, zorder=11, ax=ax9)

   
### La Nina in Middle  
plot = m.contourf(x,y,june_la, clevs, cmap=cmap, extend='both',ax=ax10)
#stipple where significant
stipple_la = np.where(june_pvalue_la < 0.05, 1, np.nan)
m.scatter(x[::2, ::2],y[::2, ::2], c=stipple_la[::2, ::2], s=10, marker='o', cmap='Wistia', edgecolor='black', linewidth=0.5, zorder=11, ax=ax10)


##############################Neutral Composite 
if domain_type != "only_centroid":
    for ax in [ax9, ax10]:
        manual_contour = m.contour(x,y,
            june_climo,
            levels=[1],
            colors="black", linewidths=3, ax=ax, zorder=14,
        )
        manual_labels = ax.clabel(
            manual_contour,
            manual=[m(-99, 46)],
            fontsize=16, zorder=14,
        )

        manual_contour = m.contour(x,y,
            june_climo,
            levels=[3],
            colors="black", linewidths=3, ax=ax, zorder=14,
        )
        manual_labels += ax.clabel(
            manual_contour,
            manual=[m(-103, 41)],
            fontsize=16, zorder=14,
        )

        manual_contour = m.contour(x,y,
            june_climo,
            levels=[5],
            colors="black", linewidths=3, ax=ax, zorder=14,
        )
        if YEAR_CUTOFF == 1940:
            manual_labels += ax.clabel(
                manual_contour,
                manual=[m(-105, 37)],
                fontsize=16, zorder=14,
            )

        for label in [*manual_labels]:
            label.set_bbox(dict(facecolor="white", edgecolor="black"))


################## Set up colorbar for anomalies

textxlab = -110.25 #-110.5
textylab = 47.0 #47.25
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

fig.suptitle("Monthly SGP Cyclone\nFrequency Composites", y=0.925, fontsize=36)

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
    plt.savefig(f"{directory_path}/monthly_sgp_cyclone_composites_{terrain_mask}_{ENSO_TYPE}_{YEAR_CUTOFF}_fdrcontrol.png", format='png', dpi=200, bbox_inches='tight')
else:
    plt.savefig(f"{directory_path}/monthly_sgp_cyclone_composites_{terrain_mask}_{ENSO_TYPE}_{YEAR_CUTOFF}.png", format='png', dpi=200, bbox_inches='tight')



print('Time to run: ' + str(datetime.utcnow() - start))
