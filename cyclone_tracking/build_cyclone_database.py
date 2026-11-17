#save out the cytrack information, organized on a per-cyclone basis
#e.g., we can now track individual cyclones

import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
from scipy.signal import detrend

from pyproj import Geod


from matplotlib.path import Path
import contourpy

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

from shapely.geometry import Polygon, MultiPolygon, Point 

WGS84_GEOD = Geod(ellps="WGS84")

#read in from input date
parser = argparse.ArgumentParser()
parser.add_argument('terrain_mask',type=int,help='terrain_mask')
parser.add_argument('domain_type',type=str,help='domain_type')
args = parser.parse_args()

terrain_mask = f"t={args.terrain_mask}"
domain_type = args.domain_type


starttime = datetime.utcnow()

#lee cyclone domain defintion
# west = -105
# north = 45
# south = 32
# east = -98
# lc = shapely.Polygon(((-106,32), (-106, 41), (-109, 43), (-109, 47.5), (-98, 47.5), (-95, 40), (-95, 32), (-106,32)))
lc = shapely.Polygon(((-106,32), (-106, 41), (-109, 43), (-109, 47.5), (-106, 47.5), (-100, 47.5), (-100, 47.5), (-98, 41), (-98, 32), (-106,32)))
    

#SGP domain definition
west = -105
north = 40
south = 30
east = -95

# sgp = shapely.Polygon(((-105,35), (-105, 40), (-100,40), (-95, 40), (-95, 30), (-100, 30), (-105, 30)))
# sgpx,sgpy = sgp.exterior.xy
# sgpx,sgpy =m(sgpx,sgpy)

# #define SGP buffer zone
# sgp_buffer = sgp.buffer(1.0)  # buffer of 1 degree
# sgp_buffer_x, sgp_buffer_y = sgp_buffer.exterior.xy


def parse_cytrack_dat(filepath):
    # Parse the output CyTRACK .dat, and return a dataframe
    #organized such that each row corresponds to a single cyclone
    #return a dataframe

    with open(filepath) as f:
        lines = f.readlines()

    records = [] #list of dictionaries, one per cyclone
    current = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        fields = [f.strip() for f in line.rstrip(',').split(',')]

        #header line: cyclone ID and number of timesteps
        if not fields[0].isdigit() or len(fields[0]) != 8:
            if current is not None:
                records.append(current)
            current = {
                "cyclone_ID": fields[0],
                "time": [],
                "lat": [],
                "lon": [],
                "mslp": [], #hPa
                "maxwind": [], #km/h, max surface surface (10-m?) wind speed
                "outermost_mslp": [], #hPa, pressure of outermost closed isobar
                "size": [] #km, avg radius of outermost closed MSLP isobar
            }
            continue

        date_str, hour_str = fields[0], fields[1]
        dt = datetime.strptime(date_str + hour_str.zfill(2), "%Y%m%d%H")


        current['time'].append(dt)
        current['lat'].append(float(fields[2]))
        current['lon'].append(float(fields[3]))
        current['mslp'].append(float(fields[4]))
        current['maxwind'].append(float(fields[5]))
        current['outermost_mslp'].append(float(fields[7]))
        current['size'].append(float(fields[8]))

    if current is not None:
        records.append(current)

    df = pd.DataFrame(records)
    return df


filepaths = glob.glob(f"{YOUR_FILE_PATH}/CyTRACK_output/{terrain_mask}/CyTRACK_*.dat")
dfs = []
for fp in sorted(filepaths):
    df = parse_cytrack_dat(fp)
    dfs.append(df)
    print(f"done with {fp}")

all_cyclones_df = pd.concat(dfs, ignore_index=True)


#rename cyclone ID to an int
all_cyclones_df['cyclone_ID'] = range(1, len(all_cyclones_df) + 1)


#determine if the cyclone centroid is within the lee cyclone domain
lats_flat = np.concatenate(all_cyclones_df["lat"].to_numpy())
lons_flat = np.concatenate(all_cyclones_df["lon"].to_numpy())

in_domain_flat = shapely.intersects_xy(lc, lons_flat, lats_flat)
# in_domain_flat = ((lats_flat >= south) & (lats_flat <= north) & (lons_flat >= west) & (lons_flat <= east))

lengths = all_cyclones_df['lat'].apply(len).to_numpy()
split_points = np.cumsum(lengths)[:-1]
all_cyclones_df['in_lee_domain'] = [arr.tolist() for arr in np.split(in_domain_flat, split_points)]


###################################################################### FUNCTIONS
def era5_filepath(dt, era_path):
    return os.path.join(era_path, f"mod_uvmslp_ERA5_{dt.strftime('%Y%m%d')}_{dt.strftime('%H')}.nc")

era_path = "{YOUR_FILE_PATH}"

def gridpoints_within_isobar(ds, center_lat, center_lon, outermost_mslp, mslp_var='msl'):
    #return boolean mask of shape MSLP of gridpoints enclosed by outermost closed isobar, for the cyclone
    #with a center of center_lat, center_lon

    lon = ds['longitude'].values
    lat = ds['latitude'].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    mslp = ds[mslp_var].values[0,:,:] / 100 #convert Pa to hPa

    gen = contourpy.contour_generator(x=lon, y=lat, z=mslp)
    lines = gen.lines(outermost_mslp)

    mask = np.zeros(mslp.shape, dtype=bool)
    pts = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    area_km2 = np.nan

    for ring in lines:
        if len(ring) < 4:
            continue

        path = Path(ring)
        if path.contains_point([center_lon, center_lat]):
            inside = path.contains_points(pts)
            mask = inside.reshape(mslp.shape) #was |= - don't think it matters

            if np.allclose(ring[0], ring[-1]): #only calculate area if cyclone entirely in domain
                area_m2, _ = WGS84_GEOD.polygon_area_perimeter(ring[:,0], ring[:,1])
                area_km2 = abs(area_m2) / 1e6

            break #stop once we found the proper loop

    return mask, area_km2

def coords_within_isobar(row, era_path, mslp_var='msl'):
    #return the coordinates that fall within the cyclone's outermost closed isobar
    masks = []
    coords_list = []
    area_list = []
    for t, lat_c, lon_c, outermost_mslp in zip(row['time'], row['lat'], row['lon'], row['outermost_mslp']):
        lon_c = lon_c + 360 if lon_c < 0 else lon_c #convert longitude to 0-360 range
        fp = era5_filepath(t, era_path)
        with xr.open_dataset(fp) as ds:
            mask, physical_area = gridpoints_within_isobar(ds, lat_c, lon_c, outermost_mslp, mslp_var=mslp_var)
            lon2d, lat2d = np.meshgrid(ds['longitude'].values, ds['latitude'].values)
            coords_list.append((lat2d[mask], lon2d[mask], mask))
            area_list.append(physical_area)
        masks.append(mask)

    return coords_list, area_list

#fill in missing value of outermost closed isobar
def fill_missing_outer(row):
    # return [float(np.ceil(m / 4.0) * 4.0) if o == -99999 else o for o, m in zip(row['outermost_mslp'], row['mslp'])] #to nearest plotted isobar
    return [m + 2.0 if o == -99999 and m < 1006 else o for o, m in zip(row['outermost_mslp'], row['mslp'])] #plus 2 mb if min MSLP < 1008

#delete all timesteps where outermost_mslp is still -99999 after the manual fill
def drop_missing_row(row):
    per_step_cols = ['time', 'lat', 'lon', 'mslp', 'maxwind', 'outermost_mslp', 'size', 'in_lee_domain', 'outermost_manual_fill']
    keep = [o != -99999 for o in row['outermost_mslp']]
    for col in per_step_cols:
        row[col] = [v for v, k in zip(row[col], keep) if k]
    return row

#delete all timesteps in January or July (and if any cyclones are entirely in those months, drop those rows)
def drop_january_july(row):
    per_step_cols = ['time', 'lat', 'lon', 'mslp', 'maxwind', 'outermost_mslp', 'size', 'in_lee_domain', 'outermost_manual_fill']
    keep = [t.month not in [1, 7] for t in row['time']]
    for col in per_step_cols:
        row[col] = [v for v, k in zip(row[col], keep) if k]
    return row

#add "any_within_SGP" list of bools if any of the coordinates of outermost closed isobar fall within  SGP domain
def any_within_SGP_domain(coords_list, south, north, west, east):
    west = west + 360 if west < 0 else west
    east = east + 360 if east < 0 else east
    #return list of bools indicating if any of the coordinates are within the SGP domain
    result = []
    for lat_arr, lon_arr, mask in coords_list:
        if lat_arr.size == 0:
            result.append(False)
            continue
        in_lee = (lat_arr >= south) & (lat_arr <= north) & (lon_arr >= west) & (lon_arr <= east) 
        result.append(bool(np.any(in_lee)))
    return result

def centroid_within_SGP_domain(lat, lon, south, north, west, east):
# normalize everything to 0-360 so the longitude comparison is consistent
    west, east = west % 360, east % 360
    result = []
    for lat, lon in zip(lat, lon):
        lon = lon % 360
        result.append(bool(south <= lat <= north and west <= lon <= east))
    return result

###########################################################################################
#First, for cyclones with missing outermost closed isobar, manually create closed cyclone
all_cyclones_df['outermost_mslp'] = all_cyclones_df.apply(fill_missing_outer, axis=1)

#go through each time. If the outermost closed isobar was originally missing, but is now filled, mark that
#time as True.
all_cyclones_df['outermost_manual_fill'] = all_cyclones_df.apply(lambda r: [o == -99999 and a != -99999 for o, a in zip(r['size'], r['outermost_mslp'])], axis=1)

#delete all timesteps where outermost_mslp is still -99999 after the manual fill
#remove any cyclones which have no remaining timesteps
all_cyclones_df = all_cyclones_df.apply(drop_missing_row, axis=1)
all_cyclones_df = all_cyclones_df[all_cyclones_df['outermost_mslp'].str.len() > 0].reset_index(drop=True)

#determine if a cyclone is a lee cyclone (first timestep is in the lee domain)
all_cyclones_df['is_lee_cyclone'] = all_cyclones_df['in_lee_domain'].apply(lambda x: x[0] if x else False)

#delete all timesteps in January or July (and if any cyclones are entirely in those months, drop those rows)
all_cyclones_df = all_cyclones_df.apply(drop_january_july, axis=1)
all_cyclones_df = all_cyclones_df[all_cyclones_df['outermost_mslp'].str.len() > 0].reset_index(drop=True)


#at each time for each cyclone, find the coordinates of the cyclone (each point within the outermost closed isobar)
isobar_results = all_cyclones_df.apply(lambda r: coords_within_isobar(r, era_path), axis=1)
all_cyclones_df['coords'] = isobar_results.apply(lambda x: x[0])
all_cyclones_df['approx_area'] = isobar_results.apply(lambda x: x[1])
#this seems to output the list of coordinates in in form [((lat_array_t0),(lon_array_t0)), ((lat_array_t1),(lon_array_t1)), ...]
#where the coordinates are not indices, but the corresponding lat and lon values



#add "any_within_SGP" list of bools if any of the coordinates of outermost closed isobar fall within SGP domain
#plus a 1 degree buffer around entire SGP domain
if domain_type == "no_buffer" or domain_type == "only_centroid":
    buffer = 0
else:
    buffer = 1

if domain_type == "only_centroid":
    all_cyclones_df['any_within_SGP'] = all_cyclones_df.apply(lambda r: centroid_within_SGP_domain(r['lat'], r['lon'], south=south-buffer, north=north+buffer, west=west-buffer, east=east+buffer),axis=1)
else:
    all_cyclones_df['any_within_SGP'] = all_cyclones_df['coords'].apply(lambda coords: any_within_SGP_domain(coords, south=south-buffer, north=north+buffer, west=west-buffer, east=east+buffer))



#associate cyclones with drylines
    #if any part of the dryline is within the cyclone's outermost closed isobar
def dryline_within_cyclone(cyclone_coords, cyclone_lat_idx, cyclone_lon_idx, time, dryline_overlap):
    _, _, mask = cyclone_coords

    cyclone_overlap = mask[np.ix_(cyclone_lat_idx, cyclone_lon_idx)]

    if time not in dryline_overlap.indexes["time"]:
        print(f"Time {time} not found in dryline_overlap index.")
        return np.nan

    #ensure NaN and 0 become False, 1 becomes ture
    dryline_mask = dryline_overlap.sel(time=time).values > 0

    if cyclone_overlap.shape != dryline_mask.shape:
        raise ValueError(
            f"Shape mismatch: cyclone={cyclone_overlap.shape}, "
            f"dryline={dryline_mask.shape}"
        )

    return bool(np.any(cyclone_overlap & dryline_mask))

def dryline_overlap_for_cyclone(row, dryline_on_cyclone_grid, cyclone_lat_idx, cyclone_lon_idx):
    return [dryline_within_cyclone(coords,cyclone_lat_idx,cyclone_lon_idx,time,dryline_on_cyclone_grid,) for coords, time in zip(row['coords'], row['time'])]


#read in dryline info, slice out overlapping grid 
dryline_xary = xr.open_dataset('{YOUR_FILE_PATH}/dryline_cat_1940_2025_corrected.nc')
# dryline_xary = dryline_xary.sel(time=slice(f"{year}-02-01T00:00:00", f"{year}-06-30T18:00:00"))

with xr.open_dataset(era5_filepath(pd.Timestamp("2000-02-01"), era_path)) as ds:
    cyclone_lat = ds["latitude"].values
    cyclone_lon = ds["longitude"].values

dryline_lat, dryline_lon = dryline_xary["lat"].values, dryline_xary["lon"].values

# Indices of cyclone cells that also exist in the dryline domain.
cyclone_lat_idx = np.flatnonzero(np.isin(cyclone_lat, dryline_lat))
cyclone_lon_idx = np.flatnonzero(np.isin(cyclone_lon, dryline_lon))

overlap_lat = cyclone_lat[cyclone_lat_idx]
overlap_lon = cyclone_lon[cyclone_lon_idx]

dryline_overlap = (
    dryline_xary["dryline"]
    .sel(lat=overlap_lat, lon=overlap_lon)
    .transpose("time", "lat", "lon")
)

#read in the summary spring_drylines pickle -- 0 out dryline_overlap data on any days where spring_drylines does not have an 
#SGP dryline. This works because there are rarely days with multiple drylines, so date masking should absolve any issues with 
#non-SGP drylines entering the calculus
all_drylines = pd.read_pickle('{YOUR_FILE_PATH}/dryline_envi_1940_2020.pkl')
spring_drylines = all_drylines[(all_drylines['month'] >= 2) & (all_drylines['month'] <= 6)].copy()
#restrict all drylines to just the SGP (use median lat/lon to define whether it's "in" the SGP)
sgp_n = 40
sgp_s = 30
sgp_w = -105
sgp_e = -95
spring_drylines = spring_drylines[(spring_drylines['median_lon'] >= -105) & (spring_drylines['median_lon'] <= -95)]
spring_drylines = spring_drylines[(spring_drylines['median_lat'] >= 30) & (spring_drylines['median_lat'] <= 40)]

#zero-out any date in dryline_overlap that does not exist in spring_drylines
dryline_overlap = dryline_overlap.where(dryline_overlap['time'].isin(spring_drylines['date']), 0)

all_cyclones_df['dryline_within'] = all_cyclones_df.apply(dryline_overlap_for_cyclone, axis=1, dryline_on_cyclone_grid=dryline_overlap, cyclone_lat_idx=cyclone_lat_idx, cyclone_lon_idx=cyclone_lon_idx)

# print(all_cyclones_df)
# print(all_cyclones_df.iloc[0])


#save out the dataframe as a pickle
all_cyclones_df.to_pickle(f"{YOUR_FILE_PATH}/all_cyclones_df_{terrain_mask}_{domain_type}.pkl")






print('Time to run: ' + str(datetime.utcnow() - starttime))
