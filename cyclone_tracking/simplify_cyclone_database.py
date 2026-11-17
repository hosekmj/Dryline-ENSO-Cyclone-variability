#Take the output from build_cyclone_database, and create a simplified output for easier statistics
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

if domain_type == "no_buffer":
    domain_type = "no_buffer"
elif domain_type == "only_centroid":
    domain_type = "only_centroid"
else:
    domain_type = None


starttime = datetime.utcnow()


# read in data
if domain_type is not None:
    all_cyclones_df = pd.read_pickle(f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/all_cyclones_df_{terrain_mask}_{domain_type}.pkl")
else:
    all_cyclones_df = pd.read_pickle(f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/all_cyclones_df_{terrain_mask}.pkl")

all_drylines = pd.read_pickle('/raid/student/mhosek/era5_drylines/pkl/dryline_envi_1940_2020.pkl')
spring_drylines = all_drylines[(all_drylines['month'] >= 2) & (all_drylines['month'] <= 6)].copy()
#restrict all drylines to just the SGP (use median lat/lon to define whether it's "in" the SGP)
sgp_n = 40
sgp_s = 30
sgp_w = -105
sgp_e = -95
spring_drylines = spring_drylines[(spring_drylines['median_lon'] >= -105) & (spring_drylines['median_lon'] <= -95)]
spring_drylines = spring_drylines[(spring_drylines['median_lat'] >= 30) & (spring_drylines['median_lat'] <= 40)]


#create a simplified dataframe
    # each row is a single day (with datetime, as well as year+month+hour columns)
    #columns include: 
        # lat (list of floats of centroid lat)
        # lon (list of floats of centroid lon)
        # sgp_within (list of bool true/false if outermost closed isobar is within the SGP domain buffer zone (any_within_SGP))
        # [sgp_mslp] (list of floats, the minimum MSLP of sgp-within cyclone)
        # [sgp_size] (list of floats, the size of sgp-within cyclone)
        # [sgp_maxwind] (list of floats, the maximum wind of sgp-within cyclone)
        # [is_lee_cyclone] (list of bools, if the cyclone is a true lee cyclone (initial timestep within lee/GP domain))
        # dryline (bool true/false if any SGP dryline present)
        # dryline_within_cyclone (bool true/false if any portion of any SGP dryline is within a cyclone's outermost closed isobar)
        # dryline_associated_cyclone (bool true/false if any SGP dryline is present AND a cyclone centroid is within the SGP buffer zone)
        # sgp_cyclone_day: bool, is there at least one cyclone this day with outermost closed isobar within SGP buffer zone?
        # lee_cyclone_day: bool, is there a true lee cyclone within the domain on this day?


#first step: create list of times from Feb-June, 1940-2020
times = pd.date_range(start="1940-02-01", end="2020-06-30", freq="6H")
times = times[(times.month >= 2) & (times.month <= 6)]

#explode the df
per_step_cols = [
    "time", "lat", "lon", "mslp", "maxwind",
    "outermost_mslp", "size", "in_lee_domain",
    "any_within_SGP", "dryline_within", "approx_area",
]

cyclones_by_time = (
    all_cyclones_df[
        ["cyclone_ID", "is_lee_cyclone"] + per_step_cols
    ]
    .explode(per_step_cols, ignore_index=True)
    .assign(time=lambda df: pd.to_datetime(df["time"]))
    .set_index("time")
    .sort_index()
)

#search through each time. Look through cyclone df, pull "in_domain" and "is_any_within" for all cyclones at that time
#if any are true, set the corresponding bools to true
lat = []
lon = []
sgp_within = []
sgp_mslp = [] #stats only for cyclones with centroids within lee domain
sgp_size = []
sgp_maxwind = []

is_lee_cyclone = []


d = []
d_within_c = []
d_associated_c = [] #cyclone outermost closed isobar within SGP buffer zone (doesn't check if d_within_c is true - this is completely independent)


for t in times:
    cyclones_at_t = cyclones_by_time.loc[[t]] if t in cyclones_by_time.index else None
    if cyclones_at_t is not None:

        lat.append(cyclones_at_t["lat"].tolist())
        lon.append(cyclones_at_t["lon"].tolist())
        sgp_within.append(cyclones_at_t["any_within_SGP"].tolist())

        is_lee_cyclone.append(cyclones_at_t["is_lee_cyclone"].tolist())
        d_within_c.append(cyclones_at_t["dryline_within"].tolist())

        sgp_mask = cyclones_at_t["any_within_SGP"].astype(bool)
        sgp_mslp.append(cyclones_at_t.loc[sgp_mask, "mslp"].tolist())
        sgp_size.append(cyclones_at_t.loc[sgp_mask, "approx_area"].tolist())
        sgp_maxwind.append(cyclones_at_t.loc[sgp_mask, "maxwind"].tolist())


        d.append(True if len(spring_drylines[spring_drylines['date'] == t]) > 0 else False)
        dasc = []
        for j in range(len(cyclones_at_t)):
            has_within = bool(cyclones_at_t.iloc[j]["any_within_SGP"])

            if d[-1] == True and has_within:
                dasc.append(True)
            else:
                dasc.append(False)

        d_associated_c.append(dasc)

    
    else:
        lat.append([])
        lon.append([])
        sgp_within.append([])
        sgp_mslp.append([])
        sgp_size.append([])
        sgp_maxwind.append([])
        is_lee_cyclone.append([False])
        d.append(True if len(spring_drylines[spring_drylines['date'] == t]) > 0 else False)
        d_within_c.append([False])
        d_associated_c.append([False])


simple_df = pd.DataFrame()
simple_df['date'] = times
simple_df['year'] = [t.year for t in times]
simple_df['month'] = [t.month for t in times]
simple_df['hour'] = [t.hour for t in times]
simple_df['lat'] = lat
simple_df['lon'] = lon
simple_df['sgp_cyclone'] = sgp_within
simple_df['sgp_mslp'] = sgp_mslp
simple_df['sgp_size'] = sgp_size
simple_df['sgp_maxwind'] = sgp_maxwind
simple_df['is_lee_cyclone'] = is_lee_cyclone
simple_df['dryline'] = d
simple_df['dryline_within_cyclone'] = d_within_c
simple_df['dryline_associated_cyclone'] = d_associated_c


simple_df['sgp_cyclone_day'] = simple_df['sgp_cyclone'].apply(any)
simple_df['lee_cyclone_day'] = simple_df['is_lee_cyclone'].apply(any)

if domain_type is not None:
    simple_df.to_pickle(f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/all_cyclones_simplified_{terrain_mask}_{domain_type}.pkl")
else:
    simple_df.to_pickle(f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/all_cyclones_simplified_{terrain_mask}.pkl")


print('Time to run: ' + str(datetime.utcnow() - starttime))