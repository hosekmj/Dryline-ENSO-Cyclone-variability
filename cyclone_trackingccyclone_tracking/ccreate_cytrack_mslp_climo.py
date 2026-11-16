#Create a 31-day centered 1940-2020 MSLP climatology at each analysis time (00z 06z 12z 18z)
#gpd_plot in bigbang4


import xarray as xr
from multiprocessing import Pool
import os
from netCDF4 import Dataset
from scipy.ndimage import uniform_filter1d
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from scipy.signal import detrend

start = datetime.utcnow()

#calculate the 31-day centered MSLP anomaly at each day/time from the cytrack MSLP files /raid/mhosek/cytrack/ERA5_data
#the climo will be for each hour (e.g., the 31-day climo of 06z MSLP)
#end of June won't be able to go the full forward window, but that should be fine

ROOT = "/raid/mhosek/cytrack/ERA5_data"
VAR = 'msl'
HOURS = [0,6,12,18]
WINDOW = 31 #number of days in centered window
FIRST, LAST = 31, 180 #no leap indices for Feb 1, June 30
HALF = WINDOW // 2

#functions
def path_for(t):
    return f"{ROOT}/mod_uvmslp_ERA5_{t.strftime('%Y%m%d')}_{t.strftime('%H')}.nc"

def noleap_day(t):
    return t.dayofyear - 1 - int(t.is_leap_year and t.month > 2)

def hour_climo(hour):
    times = pd.date_range(start="1940-01-01", end="2020-12-31", freq="D") + pd.Timedelta(hours=hour)
    times = times[~((times.month == 2) & (times.day == 29))]  # drop Feb 29

    d_all = times.dayofyear - 1 - ((times.is_leap_year) & (times.month > 2))
    times = times[(d_all >= FIRST - HALF) & (d_all <= LAST + HALF)]

    sums, counts, missing = None, np.zeros(365), []
    for t in times:
        p = path_for(t)
        if not os.path.exists(p):
            missing.append(p)
            continue
        with Dataset(p) as nc:
            field = np.asarray(nc[VAR][:]).squeeze().astype(np.float64) / 100 #convert Pa --> hPa
        if sums is None:
            sums = np.zeros((365,) + field.shape)
        d = noleap_day(t)
        sums[d] += field
        counts[d] += 1

    win_sum = uniform_filter1d(sums, WINDOW, axis=0, mode="constant")
    win_count = uniform_filter1d(counts, WINDOW, axis=0, mode="constant")
    climo = (win_sum[FIRST:LAST+1] / win_count[FIRST:LAST+1,None,None]).astype(np.float32)

    return hour, climo, missing

if __name__ == "__main__":
    sample = xr.open_dataset(path_for(datetime(1967,5,1,0)))
    lat, lon = sample['latitude'].values, sample['longitude'].values

    print("enter pool")
    with Pool(len(HOURS)) as pool:
        results = pool.map(hour_climo, HOURS)

    print("sort results")
    results.sort(key=lambda r: r[0])
    for h, _, missing in results:
        if missing:
            print(f"{h}: {len(missing)} files") #should be some missing files due to July cutoff

    climo = xr.Dataset(
        data_vars={
            "mslp_climo": (["hour", "time", "latitude", "longitude"], np.stack([r[1] for r in results], axis=0))
        },
        coords={
            "hour": HOURS,
            "time": pd.date_range(start="1945-02-01", end="1945-06-30", freq="D"),
            "latitude": lat,
            "longitude": lon},
        attrs={
            "description": "31-day centered climatology of mean sea level pressure (hPa) for each hour",
            "window": WINDOW
        }
        )

    print(climo)
    climo.to_netcdf("/raid/mhosek/cytrack/mslp_climo.nc")

print('Time to run: ' + str(datetime.utcnow() - start))
