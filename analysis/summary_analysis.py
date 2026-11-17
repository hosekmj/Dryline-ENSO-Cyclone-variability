#create some dryline-ENSO analysis plots and statistics for tables

from numpy.linalg.linalg import norm
import os
import pandas as pd
from datetime import datetime, timedelta
#import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, ScalarFormatter
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from matplotlib.patches import Patch
#import matplotlib.colors as colors
import matplotlib
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.signal import detrend
import matplotlib.patheffects as pe
from shapely.geometry import box, Point
from shapely.ops import transform
from pyproj import Transformer
import matplotlib.colors as colors
from matplotlib.ticker import ScalarFormatter

from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform
from sklearn.model_selection import ParameterSampler
from sklearn.calibration import calibration_curve

from scipy.special import expit, logit
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree

from matplotlib.legend_handler import HandlerTuple

import scikit_posthocs as sp
from statsmodels.stats.multitest import multipletests

#from shapely.geometry import LineString
import shapely

from itertools import combinations
#import random

import xarray as xr

#import multiprocessing
#from itertools import repeat

#import imageio
#import glob

from pygam import LogisticGAM, te, l , f
from pygam.utils import OptimizationError

import statsmodels.formula.api as smf
import statsmodels.api as sm

import argparse


from netCDF4 import Dataset
starttime = datetime.utcnow()

#read in from input
parser = argparse.ArgumentParser()
parser.add_argument('terrain_mask',type=int,help='terrain_mask')
parser.add_argument('enso_type',type=str,help='enso_type')
parser.add_argument('year_cutoff',type=int,help='year_cutoff')
parser.add_argument('domain_type',type=str,nargs='?',default=None,help='domain_type')

args = parser.parse_args()

terrain_mask = f"t={args.terrain_mask}"
ENSO_TYPE = args.enso_type
YEAR_CUTOFF = args.year_cutoff
domain_type = args.domain_type

if domain_type == "None":
    domain_type = None

directory_path = f"{YOUR_FILE_PATH}/{terrain_mask}/{ENSO_TYPE}"
if domain_type is not None:
    directory_path = f"{YOUR_FILE_PATH}/{terrain_mask}/{ENSO_TYPE}/{domain_type}"
os.makedirs(directory_path, exist_ok=True)

llcrnrlat=25 #20
llcrnrlon=360-110 #110
urcrnrlat=52.5 #55
urcrnrlon=360-75
lat_0=35
lon_0=360-98
latin1=25
latin2=25
m = Basemap(projection='lcc', llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
            urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,
            lat_1=latin1, lat_2=latin2, lat_0=lat_0, lon_0=lon_0,
            resolution='h', area_thresh=10000, fix_aspect=True)

#lee cyclone domain defintion
west = -105
north = 45
south = 32
east = -98
to_meters = Transformer.from_crs(
    "EPSG:4326", "EPSG:5070", always_xy=True
).transform

lee_domain = transform(to_meters, box(west, south, east, north))

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


#read in cyclones
if domain_type is not None:
    cyclones_df = pd.read_pickle(f"{YOUR_FILE_PATH}/all_cyclones_simplified_{terrain_mask}_{domain_type}.pkl")
else:
    cyclones_df = pd.read_pickle(f"{YOUR_FILE_PATH}/all_cyclones_simplified_{terrain_mask}.pkl")
#restrict to 00z
cyclones_df = cyclones_df[cyclones_df['hour'] == 0]

#read in drylines
all_drylines = pd.read_pickle('/raid/student/mhosek/era5_drylines/pkl/dryline_envi_1940_2020.pkl')
spring_drylines = all_drylines[(all_drylines['month'] >= 2) & (all_drylines['month'] <= 6)].copy()

#restrict all drylines to just the SGP (use median lat/lon to define whether it's "in" the SGP)
sgp_n = 40
sgp_s = 30
sgp_w = -105
sgp_e = -95
spring_drylines = spring_drylines[(spring_drylines['median_lon'] >= -105) & (spring_drylines['median_lon'] <= -95)]
spring_drylines = spring_drylines[(spring_drylines['median_lat'] >= 30) & (spring_drylines['median_lat'] <= 40)]

#for each dryline, assign it an operational ENSO (ENSO phase), a centered 3-month avg phase (oni_detrended; thresholded),
#and a 1-month phase (SSTa_detrended; thresholded)
spring_drylines = spring_drylines.merge(
    enso_df,
    on=['year', 'month'],
    how='left')

#repeat for cyclones
cyclones_df = cyclones_df.merge(
    enso_df,
    on=['year', 'month'],
    how='left')

#restrict to years beyond the specified cutoff
spring_drylines = spring_drylines[spring_drylines['year'] >= YEAR_CUTOFF]
cyclones_df = cyclones_df[cyclones_df['year'] >= YEAR_CUTOFF]
enso_df = enso_df[enso_df['year'] >= YEAR_CUTOFF]

total_daterange = pd.date_range(start="1945-01-01", end="2020-12-31")
total_daterange = total_daterange[(total_daterange.month >= 2) & (total_daterange.month <= 6)]

print(f"Total number of days: {len(total_daterange)}")

print(f"Total number of dryline days: {len(spring_drylines)}")
print(f"Total number of cyclone days: {len(cyclones_df[cyclones_df['sgp_cyclone_day']==True])}")

sgp_only = cyclones_df[cyclones_df['sgp_cyclone_day']==True]
coincident = spring_drylines['date'].isin(sgp_only['date'])
print(f"Total number of coincident dryline and cyclone days: {coincident.sum()}")


################################################################################################
################################################################################################
################################################################################################
################################################################################################ 9-panel bar plot of ENSO, dryline, cyclone counts by month and phase
################################################################################################ 1940-2020; 1979-2020, 1995-2020
################################################################################################
################################################################################################
################################################################################################

if YEAR_CUTOFF == 1940:

    allspring = []
    feb_dry = []
    march_dry = []
    april_dry = []
    may_dry = []
    june_dry = []

    feb_op_enso = []
    march_op_enso = []
    april_op_enso = []
    may_op_enso = []
    june_op_enso = []

    feb_3mo_enso = []
    march_3mo_enso = []
    april_3mo_enso = []
    may_3mo_enso = []
    june_3mo_enso = []

    feb_1mo_enso = []
    march_1mo_enso = []
    april_1mo_enso = []
    may_1mo_enso = []
    june_1mo_enso = []

    years = np.arange(YEAR_CUTOFF, 2021, 1)

    for yr in years:
        allspring.append(len(spring_drylines[spring_drylines['year'] == yr]))
        feb_dry.append(len(spring_drylines[(spring_drylines['year'] == yr) & (spring_drylines['month'] == 2)]))
        march_dry.append(len(spring_drylines[(spring_drylines['year'] == yr) & (spring_drylines['month'] == 3)]))
        april_dry.append(len(spring_drylines[(spring_drylines['year'] == yr) & (spring_drylines['month'] == 4)]))
        may_dry.append(len(spring_drylines[(spring_drylines['year'] == yr) & (spring_drylines['month'] == 5)]))
        june_dry.append(len(spring_drylines[(spring_drylines['year'] == yr) & (spring_drylines['month'] == 6)]))

        feb_op_enso.append(enso_df[(enso_df['year'] == yr) & (enso_df['month'] == 2)]['op_enso'].values[0])
        march_op_enso.append(enso_df[(enso_df['year'] == yr) & (enso_df['month'] == 3)]['op_enso'].values[0])
        april_op_enso.append(enso_df[(enso_df['year'] == yr) & (enso_df['month'] == 4)]['op_enso'].values[0])
        may_op_enso.append(enso_df[(enso_df['year'] == yr) & (enso_df['month'] == 5)]['op_enso'].values[0])
        june_op_enso.append(enso_df[(enso_df['year'] == yr) & (enso_df['month'] == 6)]['op_enso'].values[0])

        feb_3mo_enso.append(enso_df[(enso_df['year'] == yr) & (enso_df['month'] == 2)]['3mo_enso'].values[0])
        march_3mo_enso.append(enso_df[(enso_df['year'] == yr) & (enso_df['month'] == 3)]['3mo_enso'].values[0])
        april_3mo_enso.append(enso_df[(enso_df['year'] == yr) & (enso_df['month'] == 4)]['3mo_enso'].values[0])
        may_3mo_enso.append(enso_df[(enso_df['year'] == yr) & (enso_df['month'] == 5)]['3mo_enso'].values[0])
        june_3mo_enso.append(enso_df[(enso_df['year'] == yr) & (enso_df['month'] == 6)]['3mo_enso'].values[0])

        feb_1mo_enso.append(enso_df[(enso_df['year'] == yr) & (enso_df['month'] == 2)]['1mo_enso'].values[0])
        march_1mo_enso.append(enso_df[(enso_df['year'] == yr) & (enso_df['month'] == 3)]['1mo_enso'].values[0])
        april_1mo_enso.append(enso_df[(enso_df['year'] == yr) & (enso_df['month'] == 4)]['1mo_enso'].values[0])
        may_1mo_enso.append(enso_df[(enso_df['year'] == yr) & (enso_df['month'] == 5)]['1mo_enso'].values[0])
        june_1mo_enso.append(enso_df[(enso_df['year'] == yr) & (enso_df['month'] == 6)]['1mo_enso'].values[0])

    allspring = np.array(allspring)
    feb_dry = np.array(feb_dry)
    march_dry = np.array(march_dry)
    april_dry = np.array(april_dry)
    may_dry = np.array(may_dry)
    june_dry = np.array(june_dry)

    feb_op_enso = np.array(feb_op_enso)
    march_op_enso = np.array(march_op_enso)
    april_op_enso = np.array(april_op_enso)
    may_op_enso = np.array(may_op_enso)
    june_op_enso = np.array(june_op_enso)

    feb_3mo_enso = np.array(feb_3mo_enso)
    march_3mo_enso = np.array(march_3mo_enso)
    april_3mo_enso = np.array(april_3mo_enso)
    may_3mo_enso = np.array(may_3mo_enso)
    june_3mo_enso = np.array(june_3mo_enso)

    feb_1mo_enso = np.array(feb_1mo_enso)
    march_1mo_enso = np.array(march_1mo_enso)
    april_1mo_enso = np.array(april_1mo_enso)
    may_1mo_enso = np.array(may_1mo_enso)
    june_1mo_enso = np.array(june_1mo_enso)


    def enso_phase_counts(enso_list, yearmask=None):
        #Return count of ENSO phases by month
        la, nu, el = [], [], []
        for enso in enso_list:
            mask = np.ones(len(enso), dtype=bool) if yearmask is None else yearmask
            la.append(np.sum((enso==-1) & mask))
            nu.append(np.sum((enso==0) & mask))
            el.append(np.sum((enso==1) & mask))
        return np.array(la), np.array(nu), np.array(el)

    def counts_by_phase(count_list, enso_list, yearmask=None):
        #return dryline day sum or lee cyclone day sum by month and enso phase
        la, nu, el = [], [], []
        for count, enso in zip(count_list, enso_list):
            mask = np.ones(len(count), dtype=bool) if yearmask is None else yearmask
            la.append(np.sum(count[(enso==-1) & mask]))
            nu.append(np.sum(count[(enso==0) & mask]))
            el.append(np.sum(count[(enso==1) & mask]))
        return np.array(la), np.array(nu), np.array(el)

    #Cyclones
    monthly_lee_cyclone_days = (cyclones_df.groupby(['year', 'month'])['lee_cyclone_day'].sum().unstack('month', fill_value=0).reindex(index=years, columns=range(2,7), fill_value=0))
    monthly_sgp_cyclone_days = (cyclones_df.groupby(['year', 'month'])['sgp_cyclone_day'].sum().unstack('month', fill_value=0).reindex(index=years, columns=range(2,7), fill_value=0))


    feb_lee_cyclones = monthly_lee_cyclone_days[2].to_numpy()
    march_leecyclones = monthly_lee_cyclone_days[3].to_numpy()
    april_lee_cyclones = monthly_lee_cyclone_days[4].to_numpy()
    may_lee_cyclones = monthly_lee_cyclone_days[5].to_numpy()
    june_lee_cyclones = monthly_lee_cyclone_days[6].to_numpy()

    feb_sgp_cyclones = monthly_sgp_cyclone_days[2].to_numpy()
    march_sgp_cyclones = monthly_sgp_cyclone_days[3].to_numpy()
    april_sgp_cyclones = monthly_sgp_cyclone_days[4].to_numpy()
    may_sgp_cyclones = monthly_sgp_cyclone_days[5].to_numpy()
    june_sgp_cyclones = monthly_sgp_cyclone_days[6].to_numpy()


    months = np.arange(2,7,1)
    months_dry = [feb_dry, march_dry, april_dry, may_dry, june_dry]
    months_lee_cyclones = [feb_lee_cyclones, march_leecyclones, april_lee_cyclones, may_lee_cyclones, june_lee_cyclones]
    months_sgp_cyclones = [feb_sgp_cyclones, march_sgp_cyclones, april_sgp_cyclones, may_sgp_cyclones, june_sgp_cyclones]
    months_op_enso = [feb_op_enso, march_op_enso, april_op_enso, may_op_enso, june_op_enso]
    months_3mo_enso = [feb_3mo_enso, march_3mo_enso, april_3mo_enso, may_3mo_enso, june_3mo_enso]
    months_1mo_enso = [feb_1mo_enso, march_1mo_enso, april_1mo_enso, may_1mo_enso, june_1mo_enso]




    years = np.array(years)
    mask_1979 = years >= 1979
    mask_1995 = years >= 1995
    fig, ((ax1,ax2,ax3), (ax4,ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(nrows=3,ncols=3,figsize=(24,24))
    axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]

    la, nu, el = enso_phase_counts(months_op_enso)
    p0 = ax1.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
    p1 = ax1.bar(months, nu, color='gray', align='center', width=0.2)
    p2 = ax1.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)
    ax1.legend((p0[0], p1[0], p2[0]), ('La Nina', 'Neutral', 'El Nino'), loc=2, fontsize=12)

    la, nu, el = enso_phase_counts(months_op_enso, yearmask=mask_1979)
    p0 = ax2.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
    p1 = ax2.bar(months, nu, color='gray', align='center', width=0.2)
    p2 = ax2.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)

    la, nu, el = enso_phase_counts(months_op_enso, yearmask=mask_1995)
    p0 = ax3.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
    p1 = ax3.bar(months, nu, color='gray', align='center', width=0.2)
    p2 = ax3.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)

    ax1.set_title(r"$\bf{1940-2020}$" +  "\nENSO Phase Frequency", fontsize=20)
    ax2.set_title(r"$\bf{1979-2020}$" +  "\nENSO Phase Frequency", fontsize=20)
    ax3.set_title(r"$\bf{1995-2020}$" +  "\nENSO Phase Frequency", fontsize=20)

    la, nu, el = counts_by_phase(months_dry, months_op_enso)
    p0 = ax4.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
    p1 = ax4.bar(months, nu, color='gray', align='center', width=0.2)
    p2 = ax4.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)

    la, nu, el = counts_by_phase(months_dry, months_op_enso, yearmask=mask_1979)
    p0 = ax5.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
    p1 = ax5.bar(months, nu, color='gray', align='center', width=0.2)
    p2 = ax5.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)

    la, nu, el = counts_by_phase(months_dry, months_op_enso, yearmask=mask_1995)
    p0 = ax6.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
    p1 = ax6.bar(months, nu, color='gray', align='center', width=0.2)
    p2 = ax6.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)

    ax4.set_title("Dryline Day Frequency", fontsize=20)
    ax5.set_title("Dryline Day Frequency", fontsize=20)
    ax6.set_title("Dryline Day Frequency", fontsize=20)


    ############## cyclones
    for ax, mask in zip((ax7, ax8, ax9), (None, mask_1979, mask_1995)):
        la, nu, el = counts_by_phase(months_sgp_cyclones, months_op_enso, yearmask=mask)
        ax.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
        ax.bar(months, nu, color='gray', align='center', width=0.2)
        ax.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)

        # la_lee, nu_lee, el_lee = counts_by_phase(months_lee_cyclones, months_op_enso, yearmask=mask)
        # ax.bar(months-0.2, la_lee, color='none', edgecolor='black', hatch='x', align='center', width=0.2)
        # ax.bar(months, nu_lee, color='none', edgecolor='black', hatch='x', align='center', width=0.2)
        # ax.bar(months+0.2, el_lee, color='none', edgecolor='black', hatch='x', align='center', width=0.2)

        ax.set_title("SGP Cyclone Day Frequency", fontsize=20)

    ax1.set_ylabel("Frequency", fontsize=20)
    ax4.set_ylabel("Frequency", fontsize=20)
    ax7.set_ylabel("Frequency", fontsize=20)

    ax7.set_xlabel("Month", fontsize=20)
    ax8.set_xlabel("Month", fontsize=20)
    ax9.set_xlabel("Month", fontsize=20)

    ax1.set_ylim(0, 60)
    ax1.yaxis.set_major_locator(MultipleLocator(10))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.set_ylim(0, 30)
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax3.set_ylim(0, 25)
    ax3.yaxis.set_major_locator(MultipleLocator(5))
    ax3.yaxis.set_minor_locator(MultipleLocator(1))

    ax4.set_ylim(0, 800)
    ax4.yaxis.set_major_locator(MultipleLocator(100))
    ax4.yaxis.set_minor_locator(MultipleLocator(25))
    ax5.set_ylim(0, 400)
    ax5.yaxis.set_major_locator(MultipleLocator(50))
    ax5.yaxis.set_minor_locator(MultipleLocator(10))
    ax6.set_ylim(0, 300)
    ax6.yaxis.set_major_locator(MultipleLocator(50))
    ax6.yaxis.set_minor_locator(MultipleLocator(10))

    ax7.set_ylim(0, 700)
    ax7.yaxis.set_major_locator(MultipleLocator(100))
    ax7.yaxis.set_minor_locator(MultipleLocator(25))
    ax8.set_ylim(0, 350)
    ax8.yaxis.set_major_locator(MultipleLocator(50))
    ax8.yaxis.set_minor_locator(MultipleLocator(10))
    ax9.set_ylim(0, 250)
    ax9.yaxis.set_major_locator(MultipleLocator(50))
    ax9.yaxis.set_minor_locator(MultipleLocator(10))

    for ax in axes:
        ax.set_box_aspect(1)
        ax.set_xticks(months)
        ax.set(xticklabels = ['February', 'March', 'April', 'May', 'June'])
        ax.grid(axis='y', visible=True)

    #add legend for cyclones
    # handles = [Patch(facecolor='none', edgecolor='black', hatch='x')]
    # ax7.legend(handles=handles, labels=["Lee Cyclone Day"], loc=2, fontsize=16, framealpha=1.0)
    # ax8.legend(handles=handles, labels=["Lee Cyclone Day"], loc=2, fontsize=16, framealpha=1.0)
    # ax9.legend(handles=handles, labels=["Lee Cyclone Day"], loc=2, fontsize=16, framealpha=1.0)

    axeslabels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)']
    for ax, label in zip(axes, axeslabels):
        ax.text(0.05, 1.07, label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right', bbox=dict(facecolor='white', edgecolor='black'))

        ax.tick_params(axis='both', which='major', labelsize=16)


    plt.subplots_adjust(wspace=0.1, hspace=0.15)
    print(f"Saved figure to {directory_path}/dryline_enso_cyclone_summary.png")
    plt.savefig(f'{directory_path}/dryline_enso_cyclone_summary.png', format='png', dpi=100, bbox_inches='tight') 


    
    ################################# Repeat for the other definitions of ENSO (3mo and 1mo)

    fig, ((ax1,ax2,ax3), (ax4,ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(nrows=3,ncols=3,figsize=(24,24))
    axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]

    la, nu, el = enso_phase_counts(months_3mo_enso)
    p0 = ax1.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
    p1 = ax1.bar(months, nu, color='gray', align='center', width=0.2)
    p2 = ax1.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)
    ax1.legend((p0[0], p1[0], p2[0]), ('La Nina', 'Neutral', 'El Nino'), loc=2, fontsize=12)

    la, nu, el = enso_phase_counts(months_3mo_enso, yearmask=mask_1979)
    p0 = ax2.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
    p1 = ax2.bar(months, nu, color='gray', align='center', width=0.2)
    p2 = ax2.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)

    la, nu, el = enso_phase_counts(months_3mo_enso, yearmask=mask_1995)
    p0 = ax3.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
    p1 = ax3.bar(months, nu, color='gray', align='center', width=0.2)
    p2 = ax3.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)

    ax1.set_title(r"$\bf{1940-2020}$" +  "\nENSO Phase Frequency", fontsize=20)
    ax2.set_title(r"$\bf{1979-2020}$" +  "\nENSO Phase Frequency", fontsize=20)
    ax3.set_title(r"$\bf{1995-2020}$" +  "\nENSO Phase Frequency", fontsize=20)


    la, nu, el = counts_by_phase(months_dry, months_3mo_enso)
    p0 = ax4.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
    p1 = ax4.bar(months, nu, color='gray', align='center', width=0.2)
    p2 = ax4.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)

    la, nu, el = counts_by_phase(months_dry, months_3mo_enso, yearmask=mask_1979)
    p0 = ax5.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
    p1 = ax5.bar(months, nu, color='gray', align='center', width=0.2)
    p2 = ax5.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)

    la, nu, el = counts_by_phase(months_dry, months_3mo_enso, yearmask=mask_1995)
    p0 = ax6.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
    p1 = ax6.bar(months, nu, color='gray', align='center', width=0.2)
    p2 = ax6.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)

    ax4.set_title("Dryline Day Frequency", fontsize=20)
    ax5.set_title("Dryline Day Frequency", fontsize=20)
    ax6.set_title("Dryline Day Frequency", fontsize=20)


    ############## cyclones
    for ax, mask in zip((ax7, ax8, ax9), (None, mask_1979, mask_1995)):
        la, nu, el = counts_by_phase(months_sgp_cyclones, months_3mo_enso, yearmask=mask)
        ax.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
        ax.bar(months, nu, color='gray', align='center', width=0.2)
        ax.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)

        # la_lee, nu_lee, el_lee = counts_by_phase(months_lee_cyclones, months_3mo_enso, yearmask=mask)
        # ax.bar(months-0.2, la_lee, color='none', edgecolor='black', hatch='x', align='center', width=0.2)
        # ax.bar(months, nu_lee, color='none', edgecolor='black', hatch='x', align='center', width=0.2)
        # ax.bar(months+0.2, el_lee, color='none', edgecolor='black', hatch='x', align='center', width=0.2)

        ax.set_title("SGP Cyclone Day Frequency", fontsize=20)


    ax1.set_ylabel("Frequency", fontsize=16)
    ax4.set_ylabel("Frequency", fontsize=16)
    ax7.set_ylabel("Frequency", fontsize=16)

    ax1.set_ylim(0, 60)
    ax1.yaxis.set_major_locator(MultipleLocator(10))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.set_ylim(0, 30)
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax3.set_ylim(0, 25)
    ax3.yaxis.set_major_locator(MultipleLocator(5))
    ax3.yaxis.set_minor_locator(MultipleLocator(1))

    ax4.set_ylim(0, 800)
    ax4.yaxis.set_major_locator(MultipleLocator(100))
    ax4.yaxis.set_minor_locator(MultipleLocator(25))
    ax5.set_ylim(0, 400)
    ax5.yaxis.set_major_locator(MultipleLocator(50))
    ax5.yaxis.set_minor_locator(MultipleLocator(10))
    ax6.set_ylim(0, 300)
    ax6.yaxis.set_major_locator(MultipleLocator(50))
    ax6.yaxis.set_minor_locator(MultipleLocator(10))

    ax7.set_ylim(0, 700)
    ax7.yaxis.set_major_locator(MultipleLocator(100))
    ax7.yaxis.set_minor_locator(MultipleLocator(25))
    ax8.set_ylim(0, 350)
    ax8.yaxis.set_major_locator(MultipleLocator(50))
    ax8.yaxis.set_minor_locator(MultipleLocator(10))
    ax9.set_ylim(0, 250)
    ax9.yaxis.set_major_locator(MultipleLocator(50))
    ax9.yaxis.set_minor_locator(MultipleLocator(10))

    for ax in axes:
        ax.set_box_aspect(1)
        ax.set_xticks(months)
        ax.set(xticklabels = ['February', 'March', 'April', 'May', 'June'])
        ax.grid(axis='y', visible=True)

    # #add legend for cyclones
    # handles = [Patch(facecolor='none', edgecolor='black', hatch='x')]
    # ax7.legend(handles=handles, labels=["Lee Cyclone Day"], loc=2, fontsize=16, framealpha=1.0)
    # ax8.legend(handles=handles, labels=["Lee Cyclone Day"], loc=2, fontsize=16, framealpha=1.0)
    # ax9.legend(handles=handles, labels=["Lee Cyclone Day"], loc=2, fontsize=16, framealpha=1.0)

    # plt.subplots_adjust(wspace=-0.5, hspace=0.2)
    plt.savefig(f'{directory_path}/dryline_enso_cyclone_summary_3mo.png', format='png', dpi=100, bbox_inches='tight') 

    ######################################################### 1mo
    fig, ((ax1,ax2,ax3), (ax4,ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(nrows=3,ncols=3,figsize=(24,24))
    axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]

    la, nu, el = enso_phase_counts(months_1mo_enso)
    p0 = ax1.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
    p1 = ax1.bar(months, nu, color='gray', align='center', width=0.2)
    p2 = ax1.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)
    ax1.legend((p0[0], p1[0], p2[0]), ('La Nina', 'Neutral', 'El Nino'), loc=2, fontsize=12)

    la, nu, el = enso_phase_counts(months_1mo_enso, yearmask=mask_1979)
    p0 = ax2.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
    p1 = ax2.bar(months, nu, color='gray', align='center', width=0.2)
    p2 = ax2.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)

    la, nu, el = enso_phase_counts(months_1mo_enso, yearmask=mask_1995)
    p0 = ax3.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
    p1 = ax3.bar(months, nu, color='gray', align='center', width=0.2)
    p2 = ax3.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)

    ax1.set_title(r"$\bf{1940-2020}$" +  "\nENSO Phase Frequency", fontsize=20)
    ax2.set_title(r"$\bf{1979-2020}$" +  "\nENSO Phase Frequency", fontsize=20)
    ax3.set_title(r"$\bf{1995-2020}$" +  "\nENSO Phase Frequency", fontsize=20)


    la, nu, el = counts_by_phase(months_dry, months_1mo_enso)
    p0 = ax4.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
    p1 = ax4.bar(months, nu, color='gray', align='center', width=0.2)
    p2 = ax4.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)

    la, nu, el = counts_by_phase(months_dry, months_1mo_enso, yearmask=mask_1979)
    p0 = ax5.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
    p1 = ax5.bar(months, nu, color='gray', align='center', width=0.2)
    p2 = ax5.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)

    la, nu, el = counts_by_phase(months_dry, months_1mo_enso, yearmask=mask_1995)
    p0 = ax6.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
    p1 = ax6.bar(months, nu, color='gray', align='center', width=0.2)
    p2 = ax6.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)

    ax4.set_title("Dryline Day Frequency", fontsize=20)
    ax5.set_title("Dryline Day Frequency", fontsize=20)
    ax6.set_title("Dryline Day Frequency", fontsize=20)


    ############## cyclones
    for ax, mask in zip((ax7, ax8, ax9), (None, mask_1979, mask_1995)):
        la, nu, el = counts_by_phase(months_sgp_cyclones, months_1mo_enso, yearmask=mask)
        ax.bar(months-0.2, la, color='cornflowerblue', align='center', width=0.2)
        ax.bar(months, nu, color='gray', align='center', width=0.2)
        ax.bar(months+0.2, el, color='lightcoral', align='center', width=0.2)

        # la_lee, nu_lee, el_lee = counts_by_phase(months_lee_cyclones, months_1mo_enso, yearmask=mask)
        # ax.bar(months-0.2, la_lee, color='none', edgecolor='black', hatch='x', align='center', width=0.2)
        # ax.bar(months, nu_lee, color='none', edgecolor='black', hatch='x', align='center', width=0.2)
        # ax.bar(months+0.2, el_lee, color='none', edgecolor='black', hatch='x', align='center', width=0.2)

        ax.set_title("SGP Cyclone Day Frequency", fontsize=20)


    ax1.set_ylabel("Frequency", fontsize=16)
    ax4.set_ylabel("Frequency", fontsize=16)
    ax7.set_ylabel("Frequency", fontsize=16)

    ax1.set_ylim(0, 60)
    ax1.yaxis.set_major_locator(MultipleLocator(10))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.set_ylim(0, 30)
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax3.set_ylim(0, 25)
    ax3.yaxis.set_major_locator(MultipleLocator(5))
    ax3.yaxis.set_minor_locator(MultipleLocator(1))

    ax4.set_ylim(0, 800)
    ax4.yaxis.set_major_locator(MultipleLocator(100))
    ax4.yaxis.set_minor_locator(MultipleLocator(25))
    ax5.set_ylim(0, 400)
    ax5.yaxis.set_major_locator(MultipleLocator(50))
    ax5.yaxis.set_minor_locator(MultipleLocator(10))
    ax6.set_ylim(0, 300)
    ax6.yaxis.set_major_locator(MultipleLocator(50))
    ax6.yaxis.set_minor_locator(MultipleLocator(10))

    ax7.set_ylim(0, 700)
    ax7.yaxis.set_major_locator(MultipleLocator(100))
    ax7.yaxis.set_minor_locator(MultipleLocator(25))
    ax8.set_ylim(0, 350)
    ax8.yaxis.set_major_locator(MultipleLocator(50))
    ax8.yaxis.set_minor_locator(MultipleLocator(10))
    ax9.set_ylim(0, 250)
    ax9.yaxis.set_major_locator(MultipleLocator(50))
    ax9.yaxis.set_minor_locator(MultipleLocator(10))

    for ax in axes:
        ax.set_box_aspect(1)
        ax.set_xticks(months)
        ax.set(xticklabels = ['February', 'March', 'April', 'May', 'June'])
        ax.grid(axis='y', visible=True)

    #add legend for cyclones
    # handles = [Patch(facecolor='none', edgecolor='black', hatch='x')]
    # ax7.legend(handles=handles, labels=["Lee Cyclone Day"], loc=2, fontsize=16, framealpha=1.0)
    # ax8.legend(handles=handles, labels=["Lee Cyclone Day"], loc=2, fontsize=16, framealpha=1.0)
    # ax9.legend(handles=handles, labels=["Lee Cyclone Day"], loc=2, fontsize=16, framealpha=1.0)

    # plt.subplots_adjust(wspace=-0.5, hspace=0.2)
    plt.savefig(f'{directory_path}/dryline_enso_cyclone_summary_1mo.png', format='png', dpi=100, bbox_inches='tight') 


#################################################################################################
#################################################################################################
################################################################################################# statistics time
#################################################################################################
#################################################################################################
# Odds ratio
def dryline_cyclone_odds_ratio(cyclones, cytype="sgp_cyclone", enso_type='op_enso', phase=None, toprint=True):
    #given a dataframe of drylines, cyclones, and a phase (optional), calculate the odds ratio between drylines and cyclones
    #phase is -1, 0, or 1 (ints)

    #ONLY VALID CTYPE IS SGP_CYCLONE
    #cytype can be "lee_centroid" or "lee_within"
        #lee_centroid = the centroid is in the lee domain (DEFAULT)
        #lee_within = the outermost closed isobar is within the lee domain

    # if cytype not in ["sgp_cyclone", "sgp_centroid"]:
    #     raise ValueError("cytype must be either 'lee_centroid' or 'sgp_centroid'")


    if phase is not None:
        cyclones = cyclones[cyclones[enso_type] == phase]


    # a: number of sgp domain cyclones with a dryline
        #NOTE: just that a cyclone centroid is within the lee domain; not that its outermost closed isobar overlaps with dryline
    a = len(cyclones[(cyclones['dryline']==True) & (cyclones[cytype].map(any))])

    # b: number of drylines without a sgp domain cyclone
    b = len(cyclones[(cyclones['dryline']==True) & (cyclones[cytype].map(any)==False)])

    # c: number of sgp domain cyclones without a dryline
    c = len(cyclones[(cyclones[cytype].map(any)==True) & (cyclones['dryline']==False)])

    #d : number of days with neither a sgp domain cyclone nor dryline
    d = len(cyclones[(cyclones[cytype].map(any)==False) & (cyclones['dryline']==False)])


    input_ary = np.array([[a, b], [c, d]])
    # result = odds_ratio(input_ary) 

    or_value = (a * d) / (b * c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_lower = np.exp(np.log(or_value) - 1.96 * se)
    ci_upper = np.exp(np.log(or_value) + 1.96 * se)

    #perform chi-square test of independence
    result = chi2_contingency(input_ary)
    chi2_stat, pvalue, dof, expected = result

    if toprint == True:
        #printout results
        print('Odds ratio: ', np.round(or_value,2))
        print('95% CI: ', np.round(ci_lower,2), np.round(ci_upper,2))
        print('p-value: ', np.round(pvalue,4))
        print('Expected frequencies: ', np.round(expected[0,0],2), np.round(expected[0,1],2), np.round(expected[1,0],2), np.round(expected[1,1],2))
        #use expected to determine if chi-square test is appropriate (all values must > 5)
        print('\n')

        return
    else:
        return or_value, ci_lower, ci_upper

###### 1: drylines and sgp cyclones (cyclones with centroid within sgp domain)
toprint = False
if toprint == True:
    print("All drylines/cyclones, centroid within sgp domain")
    dryline_cyclone_odds_ratio(cyclones_df, enso_type=ENSO_TYPE)

    # 2: drylines and lee-associated cyclones (cyclones with outermost closed isobar within lee domain)
    # print("All drylines with cyclones with outermost closed isobar within lee domain")
    # dryline_cyclone_odds_ratio(cyclones_df, cytype="lee_within")

    # 3-8: go through individual months
    for month in range(2, 7, 1):
        print(f"Drylines/cyclones, month {month}")
        dryline_cyclone_odds_ratio(cyclones_df[cyclones_df['month'] == month], enso_type=ENSO_TYPE)

        # print(f"All drylines with cyclones with outermost closed isobar within lee domain for month {month}")
        # dryline_cyclone_odds_ratio(cyclones_df[cyclones_df['month'] == month], cytype="lee_within")

    #next, by enso phase
    for phase in range(-1, 2, 1):
        print(f"all Drylines/cyclones, ENSO phase {phase}")
        dryline_cyclone_odds_ratio(cyclones_df[cyclones_df[ENSO_TYPE] == phase], enso_type=ENSO_TYPE)

        for month in range(2, 7, 1):
            print(f"Drylines/cyclones, ENSO phase {phase}, month {month}")
            dryline_cyclone_odds_ratio(cyclones_df[(cyclones_df[ENSO_TYPE] == phase) & (cyclones_df['month'] == month)], enso_type=ENSO_TYPE)


# records_gp = []
# for phase in range(-1, 2, 1):
#     for month in range(2, 7, 1):
#         or_val, ci_lower, ci_upper = dryline_cyclone_odds_ratio(cyclones_df[(cyclones_df[ENSO_TYPE] == phase) & (cyclones_df['month'] == month)], enso_type=ENSO_TYPE, toprint=False)
#         records_gp.append({'phase': phase, 'month': month, 'or': or_val, 'ci_lower': ci_lower, 'ci_upper': ci_upper})

records_sgp = []
for phase in range(-1, 2, 1):
    for month in range(2, 7, 1):
        or_val, ci_lower, ci_upper = dryline_cyclone_odds_ratio(cyclones_df[(cyclones_df[ENSO_TYPE] == phase) & (cyclones_df['month'] == month)], cytype="sgp_cyclone", enso_type=ENSO_TYPE, toprint=False)
        records_sgp.append({'phase': phase, 'month': month, 'or': or_val, 'ci_lower': ci_lower, 'ci_upper': ci_upper})


# or_df_gp = pd.DataFrame(records_gp)
or_df_sgp = pd.DataFrame(records_sgp)

fig, ax = plt.subplots(figsize=(8,8))
phase_info = [(-1, "La Niña", "royalblue", -0.15, "o"), 
              (0, "Neutral", "gray", 0, "X"), 
              (1, "El Niño", "firebrick", 0.15, "s")]
month_numbers = [2, 3, 4, 5, 6]
month_labels = ["February", "March", "April", "May", "June"]

# for phase, label, color, offset, marker in phase_info:
#     df_p = or_df_gp[or_df_gp['phase'] == phase].sort_values('month')

#     x = df_p['month'] + offset
#     y = df_p['or']
#     yerr = [y - df_p['ci_lower'], df_p['ci_upper'] - y]
#     ax.errorbar(x, y, yerr=yerr, fmt=marker, color=color, markersize=10, markeredgecolor='black', label=label+ ", GP", ecolor=color, elinewidth=1.5, capthick=1.5, capsize=5)

# phase_info_sgp = [(-1, "La Niña", "cornflowerblue", -0.15, "o"), 
#               (0, "Neutral", "silver", 0, "X"), 
#               (1, "El Niño", "indianred", 0.15, "s")]

phase_info_sgp = phase_info
for phase, label, color, offset, marker in phase_info_sgp:
    df_p = or_df_sgp[or_df_sgp['phase'] == phase].sort_values('month')

    x = df_p['month'] + offset - 0.025
    y = df_p['or']
    yerr = [y - df_p['ci_lower'], df_p['ci_upper'] - y]
    ax.errorbar(x, y, yerr=yerr, fmt=marker, color=color, markersize=10, markeredgecolor='black', label=label, ecolor=color, elinewidth=1.5, capthick=1.5, capsize=5)



ax.axhline(1, color='black', linestyle='--', linewidth=1.5)

ax.set_xticks(month_numbers)
ax.set_xticklabels(month_labels, fontsize=12)
ax.set_xlabel("Month", fontsize=18)
ax.set_ylabel("Odds Ratio (log)", fontsize=18)
ax.set_title("Dryline Day Odds Ratio given an SGP Cyclone\nby Month and ENSO phase", fontsize=20)

# ax.set_ylim(0.5, 16)
# ax.yaxis.set_major_locator(MultipleLocator(2))
# ax.yaxis.set_minor_locator(MultipleLocator(1))

ax.set_yscale('log')
ax.set_ylim(0.9, 20)
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.set_yticks([1, 5, 10])
ax.grid(axis='y', which='both')
ax.tick_params(axis='y', which='both', length=6, width=1.5)

ax.legend(title="ENSO Phase", fontsize=16, title_fontsize=16, framealpha=1.0)
plt.tight_layout()
plt.savefig(f'{directory_path}/dryline_cyclone_odds_ratio_{YEAR_CUTOFF}.png', dpi=150)


########################################################## statistics on dryline association with cyclone
#dryline_within_cyclone: dryline is within the outermost closed isobar - highest confidence that dryline is attributable with cyclone
#dryline_associated_cyclone: dryline is present with a cyclone centroid within the lee domain - medium confidence that cyclone influenced dryline characteristics
#dryline_adjacent_cyclone: cyclone outermost closed isobar is within lee domain - it's possible the dryline is influenced by cyclone, but low confidence
#all false: medium confident that dryline is not influenced by a lee cyclone

#go through each month, count up the number of dryline_within_cyclone, dryline_associated_cyclone, dryline_adjacent_cyclone, and quiescent_dryline
#as well as ENSO phase
#plot table with frequencies/ratios by month and ENSO phase

#if it looks like there are sufficient sample size, run the negative binomial regression against **just counts of drylines associated with cyclones**


#build dataframe that has counts (and enso phase) by month
dry_count_df = pd.DataFrame()

years = np.arange(YEAR_CUTOFF, 2021, 1)
months = np.arange(2, 7, 1)

yr = []
mo = []
total_d_count = []
dwc_count = [] #dryline is within cyclone outermost closed isobar
dac_count = [] #a cyclone has its outermost closed isobar within the SGP domain buffer
dajc_count = [] #DEFUNCT
dq_count = [] #dryline is present, but there are NO SGP cyclones
op_enso = []
enso_3mo = []
enso_1mo = []

for y in years:
    for m in months:
        #get the total number of dryline days for this year and month
        total_d_count.append(len(spring_drylines[(spring_drylines['year'] == y) & (spring_drylines['month'] == m)]))

        #dryline_within_cyclone count
        dwc_count.append(len(cyclones_df[(cyclones_df['year'] == y) & (cyclones_df['month'] == m) & (cyclones_df['dryline_within_cyclone'].map(any))]))

        #dryline_associated_cyclone count
        dac_count.append(len(cyclones_df[(cyclones_df['year'] == y) & (cyclones_df['month'] == m) & (cyclones_df['dryline_associated_cyclone'].map(any) & (cyclones_df['dryline_within_cyclone'].map(any)==False))]))

        #dryline_adjacent_cyclone count
        # dajc_count.append(len(cyclones_df[(cyclones_df['year'] == y) & (cyclones_df['month'] == m) & (cyclones_df['dryline_adjacent_cyclone'].map(any) & (cyclones_df['dryline_within_cyclone'].map(any)==False) & (cyclones_df['dryline_associated_cyclone'].map(any)==False))]))
        dajc_count.append(0) #set to 0 since dryline_adjacent_cyclone is not being counted
        
        #find difference to get quiescent cyclone count
        dq_count.append(total_d_count[-1] - dwc_count[-1] - dac_count[-1] - dajc_count[-1])

        #get the ENSO phase for this year and month 
        op_enso_phase = enso_df[(enso_df['year'] == y) & (enso_df['month'] == m)]['op_enso'].values[0]
        enso_phase_3m = enso_df[(enso_df['year'] == y) & (enso_df['month'] == m)]['3mo_enso'].values[0]
        enso_phase_1m = enso_df[(enso_df['year'] == y) & (enso_df['month'] == m)]['1mo_enso'].values[0]

        #append to the model dataframe
        yr.append(y)
        mo.append(m)
        op_enso.append(op_enso_phase)
        enso_3mo.append(enso_phase_3m)
        enso_1mo.append(enso_phase_1m)

dry_count_df['year'] = yr
dry_count_df['month'] = mo
dry_count_df['total_dryline_count'] = total_d_count
dry_count_df['dryline_within_cyclone_count'] = dwc_count
dry_count_df['dryline_associated_cyclone_count'] = dac_count
dry_count_df['dryline_adjacent_cyclone_count'] = dajc_count
dry_count_df['quiescent_dryline_count'] = dq_count
dry_count_df['op_enso'] = op_enso
dry_count_df['3mo_enso'] = enso_3mo
dry_count_df['1mo_enso'] = enso_1mo

#6-panel plot. 5 panels are month, scatterplot of active drylines (dryline_within) vs quiescent drylines. Separate by ENSO phase
#use panel 6 to list significance in linear correlations

#also produce sensitivity test plots: one test where dryline_associated and dryline_adjacent are added to "active",
    #and another where they are added to "quiescent"

def plot_dryline_vs_quiescent(ax, dry_count_df):
    #plot linear correlation by ENSO phase
    #calculate spearman's correlation, return r and p-value (lists in ENSO order)

    phase_colors = {
        -1: "royalblue", #La Nina
        0: "gray", #neutral
        1: "firebrick"} #El Nino

    phase_shapes = {
            -1: "o",
            0: "X",
            1: "s"}

    phase_sizes = {
        -1: 80, #La Nina
        0: 60, #Neutral
        1: 30  #El Nino
    }

    r, p = [], []
    for phase in [-1, 0, 1]:
        phase_df = dry_count_df[dry_count_df[ENSO_TYPE] == phase]
        x = phase_df['dryline_within_cyclone_count']
        y = phase_df['quiescent_dryline_count']
        ax.scatter(x, y, color=phase_colors[phase], marker=phase_shapes[phase], s=phase_sizes[phase], edgecolors='black', linewidths=0.75, label=phase, zorder=5)

        #plot an ordinary least squares regression
        fit_df = pd.DataFrame({"active": x.to_numpy(), "quiescent": y.to_numpy()})
        ols = smf.ols("quiescent ~ active", data=fit_df).fit()
        x_grid = np.linspace(fit_df['active'].min(), fit_df['active'].max(), 100)
        prediction = ols.get_prediction(pd.DataFrame({"active": x_grid})).summary_frame(alpha=0.05)

        ax.plot(x_grid, prediction['mean'], color=phase_colors[phase], linestyle='--', linewidth=3, zorder=6)
        ax.fill_between(x_grid, prediction['mean_ci_lower'], prediction['mean_ci_upper'], color=phase_colors[phase], alpha=0.2, zorder=5)

        #plot the linear correlation
        # slope, intercept = np.polyfit(x, y, 1)
        # ax.plot(np.array(sorted(x)), slope * np.array(sorted(x)) + intercept, color=phase_colors[phase], linestyle='--', linewidth=3, zorder=6)


        #calculate spearman's correlation
        r_phase, p_phase = stats.spearmanr(x, y)
        r.append(r_phase)
        p.append(p_phase)

    return r, p #return's list of r's, p's in form [La Nina, Neutral, El Nino]



fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 16))
ax6.set_axis_off()

feb_dry_count = dry_count_df[dry_count_df['month']==2]
feb_r, feb_p = plot_dryline_vs_quiescent(ax1, feb_dry_count)

march_dry_count = dry_count_df[dry_count_df['month']==3]
march_r, march_p = plot_dryline_vs_quiescent(ax2, march_dry_count)

april_dry_count = dry_count_df[dry_count_df['month']==4]
april_r, april_p = plot_dryline_vs_quiescent(ax3, april_dry_count)

may_dry_count = dry_count_df[dry_count_df['month']==5]
may_r, may_p = plot_dryline_vs_quiescent(ax4, may_dry_count)

june_dry_count = dry_count_df[dry_count_df['month']==6]
june_r, june_p = plot_dryline_vs_quiescent(ax5, june_dry_count)



ax1.set_title("February", fontsize=24)
ax2.set_title("March", fontsize=24)
ax3.set_title("April", fontsize=24)
ax4.set_title("May", fontsize=24)
ax5.set_title("June", fontsize=24)

ax1.set_ylabel("Quiescent Drylines", fontsize=18)
ax3.set_ylabel("Quiescent Drylines", fontsize=18)
ax5.set_ylabel("Quiescent Drylines", fontsize=18)


ax4.set_xlabel("Cyclone-Associated Drylines", fontsize=18)
ax5.set_xlabel("Cyclone-Associated Drylines", fontsize=18)


ax1.set_xlim(-0.5, 5.5)
ax2.set_xlim(-0.5, 11.5)
ax3.set_xlim(-0.5, 11.5)
ax4.set_xlim(-0.5, 17.5)
ax5.set_xlim(-0.5, 17.5)

ax1.set_ylim(-0.5, 5.5)
ax2.set_ylim(-0.5, 11.5)
ax3.set_ylim(-0.5, 11.5)
ax4.set_ylim(-0.5, 17.5)
ax5.set_ylim(-0.5, 17.5)

axes = [ax1,ax2,ax3,ax4,ax5]
panel_labels = ["a)", "b)", "c)", "d)", "e)"]
for i, ax in enumerate(axes):
    ax.text(0.05, 1.1, panel_labels[i], transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right', bbox=dict(facecolor='white', edgecolor='black'))
    ax.grid(visible=True, zorder=2)
    ax.grid(visible=True, zorder=2)
    ax.yaxis.set_major_locator(MultipleLocator((2)))
    ax.xaxis.set_major_locator(MultipleLocator((2)))

    ax.yaxis.set_minor_locator(MultipleLocator((1)))
    ax.xaxis.set_minor_locator(MultipleLocator((1)))

    #plot dashed line representing the 1:1 line
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", color='black', linewidth=1.5, zorder=1)

    ax.tick_params(axis='both', which='major', labelsize=14)


#add legend to the empty ax6
handles = [(Patch(facecolor='firebrick', edgecolor='none', alpha=0.3), Line2D([],[], color='firebrick', linewidth=2, linestyle='--', marker='s', markersize=9, markeredgecolor='black', markeredgewidth=0.75)),
        (Patch(facecolor='gray', edgecolor='none', alpha=0.3), Line2D([],[], color='gray', linewidth=2, linestyle='--', marker='*', markersize=13, markeredgecolor='black', markeredgewidth=0.75)),
        (Patch(facecolor='royalblue', edgecolor='none', alpha=0.3), Line2D([],[], color='royalblue', linewidth=2, linestyle='--', marker='o', markersize=9, markeredgecolor='black', markeredgewidth=0.75))]

ax6.legend(handles=handles, labels=["El Nino", "Neutral", "La Nina"], handler_map={tuple: HandlerTuple(ndivide=1)}, loc='center', fontsize=30, framealpha=1.0)


plt.subplots_adjust(wspace=0.1)
plt.suptitle("Monthly Cyclone-Associated vs Quiescent Dryline Counts", fontsize=30, y=0.95)
plt.savefig(f'{directory_path}/monthly_dryline_active_vs_quiescent_scatter_{YEAR_CUTOFF}.png', format='png', dpi=150, bbox_inches='tight') 
#don't use -- too many years with identical samples stacking

#print out spearman's correlation for table later
toprint_spear = True
if toprint_spear == True:
    print("February r:", feb_r)
    print("February p-value:", feb_p)
    print("March r:", march_r)
    print("March p-value:", march_p)
    print("April r:", april_r)
    print("April p-value:", april_p)
    print("May r:", may_r)
    print("May p-value:", may_p)
    print("June r:", june_r)
    print("June p-value:", june_p)

#Make a single plot. For each month (x-axis) plot 3 boxplots, which are active - quiescent dryline counts, by ENSO phase
def active_minus_quiescent(df, month, phase):
    #given the dataframe of dryline counts
    #return the difference in active - quiescent drylines for that given month and phase
    #phase, month: int
    #returns: list of differences for each year

    temp = df[(df['month']==month) & (df[ENSO_TYPE]==phase)]

    return (temp['dryline_within_cyclone_count'] - temp['quiescent_dryline_count']).tolist()

# feb_el = active_minus_quiescent(dry_count_df, 2, 1) 
# feb_n = active_minus_quiescent(dry_count_df, 2, 0) 
# feb_la = active_minus_quiescent(dry_count_df, 2, -1) 

# march_el = active_minus_quiescent(dry_count_df, 3, 1) 
# march_n = active_minus_quiescent(dry_count_df, 3, 0) 
# march_la = active_minus_quiescent(dry_count_df, 3, -1) 

# april_el = active_minus_quiescent(dry_count_df, 4, 1) 
# april_n = active_minus_quiescent(dry_count_df, 4, 0) 
# april_la = active_minus_quiescent(dry_count_df, 4, -1) 

# may_el = active_minus_quiescent(dry_count_df, 5, 1) 
# may_n = active_minus_quiescent(dry_count_df, 5, 0) 
# may_la = active_minus_quiescent(dry_count_df, 5, -1) 

# june_el = active_minus_quiescent(dry_count_df, 6, 1) 
# june_n = active_minus_quiescent(dry_count_df, 6, 0) 
# june_la = active_minus_quiescent(dry_count_df, 6, -1) 


fig, ax = plt.subplots(1, 1, figsize=(10, 10))

month_numbers = [2,3,4,5,6]
month_labels = ["February", "March", "April", "May", "June"]
month_positions = np.arange(len(month_numbers))

phase_settings = [(-1, "La Nina", "royalblue", -0.25),
                  (0, "Neutral", "gray", 0),
                  (1, "El Nino", "firebrick", 0.25)]

legend_handles = []
for phase, label, color, offset in phase_settings:
    values = [active_minus_quiescent(dry_count_df, month, phase) for month in month_numbers]

    boxes = ax.boxplot(values, positions=month_positions+offset, widths=0.2, patch_artist=True, manage_ticks=False,
                       medianprops={"color": "black", "linewidth": 1.5}, flierprops={"marker":"o", "markerfacecolor": color, "markeredgecolor": "black", "markersize": 8, "alpha": 0.6},
                       showmeans=True, meanprops={"marker": "D", "markeredgecolor": "black", "markerfacecolor": "black", "markersize": 6, "alpha": 0.8})

    for patch in boxes["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    for position, samples in zip(month_positions + offset, values):
        ax.text(
            position,
            0.015,
            f"n={len(samples)}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            color=color,
            fontweight="bold")
    
    legend_handles.append(Patch(facecolor=color, alpha=0.8, label=label))

ax.axhline(0, color="black", linestyle='--', linewidth=1.25)

ax.set_xticks(month_positions)
ax.set_xticklabels(month_labels)
ax.set_xlim(-0.6, len(month_numbers) - 0.4)
ax.set_xlabel("Month", fontsize=16)

ax.set_ylim(-9, 17.5)
ax.yaxis.set_major_locator(MultipleLocator((2)))
ax.yaxis.set_minor_locator(MultipleLocator((1)))

# ax.set_ylabel("Active - Quiescent Dryline counts", fontsize=16)
ymin,ymax = ax.get_ylim()
label_transform = ax.get_yaxis_transform()
ax.text(-0.075, ymax/2, "More Active", transform=label_transform, rotation=90, ha="center", va="center", fontsize=16)
ax.text(-0.075, ymin/2, "More Quiescent", transform=label_transform, rotation=90, ha="center", va="center", fontsize=16)
ax.annotate("",
    xy=(-0.075, 6.0),
    xytext=(-0.075, -1.5),
    xycoords=label_transform,
    textcoords=label_transform,
    arrowprops={
        "arrowstyle": "<->",
        "color": "black",
        "linewidth": 1.5,
        "mutation_scale": 15,
        "shrinkA": 0,
        "shrinkB": 0,
    },
    annotation_clip=False)

ax.set_title("Monthly 'Active' minus 'Quiescent' Dryline Counts", fontsize=20)
leg = ax.legend(handles=legend_handles, title="ENSO Phase", fontsize=18, title_fontsize=18, loc=2, framealpha=1)
leg.set_zorder(100)

# ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.3)

#add KW and Dunn test for each month, and then for each phase across months
phase_names = {-1: "La Niña", 0: "Neutral", 1: "El Niño"}
month_name_map = dict(zip(month_numbers, month_labels))

aq_df = dry_count_df.assign(aq_diff=dry_count_df['dryline_within_cyclone_count'] - dry_count_df['quiescent_dryline_count'],
                            phase_name = dry_count_df[ENSO_TYPE].map(phase_names),
                            month_name = pd.Categorical(dry_count_df['month'].map(month_name_map), categories=month_labels, ordered=True))

#DON'T DO KW TEST WITH BOXPLOTS -- MISLEADING BECAUSE UNDERLYING FREQUENCY OF DRYLINES VARIES BY MONTH (E.G., CAN'T HAVE +2 WHEN ONLY 1 DRYLINE IN FEB)
# def kw_dunn(df, group_col, value_col='aq_diff', p_adjust="holm"):
#     groups = [g[value_col].to_numpy() for _, g in df.groupby(group_col, observed=True)]
#     h_stat, p_kw = stats.kruskal(*groups)
#     #epsilon-square effect size for KW
#     k = df[group_col].nunique()
#     # eps2 = h_stat / ((len(df)**2 - 1) / (len(df) + 1))
#     eta2 = (h_stat - k + 1) / (len(df) - k)
#     dunn = sp.posthoc_dunn(df, val_col=value_col, group_col=group_col, p_adjust=p_adjust)
#     return h_stat, p_kw, eta2, dunn

# kw_records = []
# dunn_results = {}

# #phases compared within month
# for m in month_numbers:
#     sub = aq_df[aq_df['month'] == m]
#     h_stat, p_kw, eta2, dunn = kw_dunn(sub, group_col='phase_name')
#     kw_records.append({"Test": "Phase within month", "subset": month_name_map[m], "H": h_stat, "p": p_kw, "eta2": eta2})
#     dunn_results[("Phase within month", month_name_map[m])] = dunn

# #months compared within each phase
# for phase, pname in phase_names.items():
#     sub = aq_df[aq_df[ENSO_TYPE] == phase]
#     h_stat, p_kw, eta2, dunn = kw_dunn(sub, group_col='month_name')
#     kw_records.append({"Test": "Month within phase", "subset": pname, "H": h_stat, "p": p_kw, "eta2": eta2})
#     dunn_results[("Month within phase", pname)] = dunn

# kw_df = pd.DataFrame(kw_records)

# #apply Holm correction across the KW tests for each family
# kw_df['p_kw_holm'] = kw_df.groupby("Test")['p'].transform(lambda p: multipletests(p, method="holm")[1])
# kw_df['significant'] = kw_df['p_kw_holm'] < 0.05

# print(kw_df.round(4).to_string(index=False))

# # Only report Dunn results where the KW test is significant
# for _, row in kw_df[kw_df["significant"]].iterrows():
#     print(f"\nDunn's test (Holm-adjusted p-values): {row['Test']} - {row['subset']}")
#     print(dunn_results[(row["Test"], row["subset"])].round(4))

# #add significance arrows
# def sig_bracket(ax, phase, start_month, target_months, y_top, y_ends, color, gap=0.3, edge='black', lw=5, edge_lw=8, headsize=18):
#     offset = {p: o for p, _, _, o in phase_settings}[phase]
#     months_all = [start_month] + list(target_months)
#     xs = [month_positions[month_numbers.index(mo)] + offset for mo in months_all]

#     # y_ends = [np.percentile(active_minus_quiescent(dry_count_df, mo, phase), 75) + gap for mo in months_all]

#     segments = [([min(xs), max(xs)], [y_top, y_top])]
#     segments += [([x,x], [y_top, y_end+gap]) for x, y_end in zip(xs, y_ends)]

#     #outline layer under the fill hides seams
#     for width, c, z in [(edge_lw, edge, 50), (lw, color, 52)]:
#         for sx, sy in segments:
#             ax.plot(sx, sy, color=c, linewidth=width, solid_capstyle='projecting', zorder=z)

#     ax.plot(xs, y_ends, linestyle='none', marker='v', markersize=headsize, color=color, markeredgecolor=edge, markeredgewidth=1.5, zorder=51)

# #La Nina: Feb vs April and May
# sig_bracket(ax, phase=-1, start_month=2, target_months=[4, 5], y_top=15, y_ends=[5,9,8], color='royalblue')

# #Neutral: Feb vs April, May, and June
# sig_bracket(ax, phase=0, start_month=2, target_months=[4, 5, 6], y_top=14, y_ends=[5, 11, 13, 9], color='gray')


fig.tight_layout()
plt.savefig(f'{directory_path}/monthly_dryline_active_vs_quiescent_box_{YEAR_CUTOFF}.png', format='png', dpi=150, bbox_inches='tight') 


aq_df['n_aq'] = aq_df['dryline_within_cyclone_count'] + aq_df['quiescent_dryline_count']
# calculate binomial generalized linear model (GLM) for comparing probability of a dryline to be active across phases and months 
def binomial_compare(df, group_col, levels):
    #group_col = column heading for comparison (month or op_enso)
    #levels = group values in the order I want them. The first one is the reference level (Feb for months, neutral (0) for phases)

    # LHS "successes + failures" gives statsmodels a binomial with A+Q trials per row

    #drop year-months with no drylines. This would cause problems
    df = df[df['n_aq'] > 0].copy() 
    df[group_col] = pd.Categorical(df[group_col], categories=levels)

    #fit binomial GLM with logit link.
        # left side: A + Q: with binomial family, a two-column response means (successes, failures). 
        #each row is "A active out of A+Q trials". Months with more drylines get more weight.
        # Right side: one indicator ("dummy") per nono-reference group 
        #The model estimates:
            #       intercept         = logit(p_reference)
            #       C(...)[T.level]   = logit(p_level) - logit(p_reference)   (a log odds ratio)
            #   With only one categorical predictor, the model is saturated, so each group's
            #   fitted p equals its pooled observed proportion: sum(A) / sum(A+Q) over all years.
            # - cov_type='cluster', groups=year: cluster-robust ("sandwich") standard errors.
            #   They allow rows from the same year to be correlated and allow extra-binomial
            #   variance (overdispersion), so p-values and CIs are not overconfident.
            #   The point estimates do not change, only the uncertainty.
    group_term = f"C(Q({group_col!r}))"
    res = smf.glm(f"dryline_within_cyclone_count + quiescent_dryline_count ~ {group_term}",
                  data=df, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': df['year']})

    names = list(res.params.index)
    term = lambda lvl: f"{group_term}[T.{lvl}]"

    # omnibus test: do any groups differ?
    # it is a joint Wald chi-square test that ALL group coefficients are 0 (e.g., each group has same proportion of active)
    # it uses cluster-robust convariance. Wald test returns a table with one row per model term, which we take the p-value from
    omnibus_p = float(res.wald_test_terms(scalar=True).table.loc[group_term, 'pvalue'])


    #loop over every pair of groups, in order given by levels (e.g., 2,3, 2,4). 10 pairs for 5 months, 3 pairs for 3 phases
        # Build the contrast logit(p_b) - logit(p_a):
        #   logit(p_b) - logit(p_a) = [coef_b - coef_ref] - [coef_a - coef_ref] = coef_b - coef_a
        # so the weight is +1 on b's coefficient and -1 on a's.
        # The reference level has no coefficient (it is folded into the intercept),
        # so when a or b is the reference we skip it, which makes its term 0.
        # Example: a = Feb (reference), b = May  ->  c = [0, 0, 0, +1, 0]  ->  coef_May.
        # The intercept weight stays 0 because it cancels in the difference.
    rows = []
    for a, b in combinations(levels, 2):
        c = np.zeros(len(names))
        if a != levels[0]: c[names.index(term(a))] -= 1
        if b != levels[0]: c[names.index(term(b))] += 1

        # Test the linear combination sum(c * params) = 0 using the cluster-robust
        # covariance. Returns the estimate (t.effect), its SE, a z statistic, and a
        # p-value. (For GLMs statsmodels uses the normal distribution by default,
        # so this is effectively a Wald z-test.)
        t = res.t_test(c)

        rows.append({"A": a, "B": b, "odds_ratio_B/A": float(np.exp(t.effect[0])), "p": float(t.pvalue)})
        # - t.effect[0] is the log odds ratio (B vs A); exp() turns it into an odds ratio.
        #   OR > 1 means drylines are more likely to be active in B than in A.
        # - t.pvalue is the unadjusted two-sided p-value.

    # Holm-adjust the p-values 
    pairs = pd.DataFrame(rows)
    pairs['p_holm'] = multipletests(pairs['p'], method='holm')[1]

    # model-based proportion active per group
    pred = res.get_prediction(pd.DataFrame({group_col: pd.Categorical(levels, categories=levels)})).summary_frame()
    pred.index = levels
    return omnibus_p, pairs, pred[['mean', 'mean_ci_lower', 'mean_ci_upper']]
    # Returns:
    # 1. omnibus_p: is there any difference among the groups?
    # 2. pairs:     pairwise odds ratios with raw and Holm-adjusted p-values
    # 3. pred:      proportion active per group with its 95% CI (the values to plot)

# months within each phase
for phase, pname in phase_names.items():
    p, pairs, props = binomial_compare(aq_df[aq_df[ENSO_TYPE] == phase], 'month', month_numbers)
    # print(f"\n{pname}: omnibus p = {p:.4f}\n{props.round(3)}\n{pairs.round(4)}")

# phases within each month
for m in month_numbers:
    p, pairs, props = binomial_compare(aq_df[aq_df['month'] == m], ENSO_TYPE, [0, -1, 1])
    # print(f"\n{month_name_map[m]}: omnibus p = {p:.4f}\n{props.round(3)}\n{pairs.round(4)}")


#fit models: months within each phase, phases within each month
month_results = {phase: binomial_compare(aq_df[aq_df[ENSO_TYPE] == phase], 'month', month_numbers)
                 for phase in phase_names}
phase_results = {m: binomial_compare(aq_df[aq_df['month'] == m], ENSO_TYPE, [0, -1, 1])
                 for m in month_numbers}

#Holm across omnibus tests within each family
month_omnibus = pd.Series({p: r[0] for p, r in month_results.items()})
month_omnibus_holm = pd.Series(multipletests(month_omnibus, method='holm')[1], index=month_omnibus.index)
phase_omnibus = pd.Series({m: r[0] for m, r in phase_results.items()})
phase_omnibus_holm = pd.Series(multipletests(phase_omnibus, method='holm')[1], index=phase_omnibus.index)

# def compact_letters(levels, pairs, alpha=0.05):
#     # letters = maximal sets of levels with no significant pairwise difference
#     sig = {frozenset((a, b)) for a, b, p in pairs[['A', 'B', 'p_holm']].itertuples(index=False) if p < alpha}
#     groups = []
#     for size in range(len(levels), 0, -1):
#         for subset in combinations(levels, size):
#             if any(frozenset(pr) in sig for pr in combinations(subset, 2)):
#                 continue
#             if any(set(subset) <= g for g in groups):
#                 continue
#             groups.append(set(subset))
#     groups.sort(key=lambda g: min(levels.index(l) for l in g))
#     letters = {lvl: "" for lvl in levels}
#     for i, g in enumerate(groups):
#         for lvl in g:
#             letters[lvl] += "abcdefgh"[i]
#     return letters


fig, ax = plt.subplots(figsize=(8, 8))
markers = {-1: "o", 0: "X", 1: "s"}

for phase, label, color, offset in phase_settings:
    omnibus_p, pairs, pred = month_results[phase]
    x = month_positions + offset
    y = pred['mean'].to_numpy()
    lo, hi = pred['mean_ci_lower'].to_numpy(), pred['mean_ci_upper'].to_numpy()

    # ax.plot(x, y, color=color, linewidth=1.5, alpha=0.5, zorder=4)
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt=markers[phase], color=color, markersize=11,
                markeredgecolor='black', ecolor=color, elinewidth=2, capsize=5, capthick=2, label=label, zorder=5)

    # if month_omnibus_holm[phase] < 0.05:
    #     letters = compact_letters(month_numbers, pairs)
    #     for xi, top, mo in zip(x, hi, month_numbers):
    #         ax.text(xi, top + 0.02, letters[mo], ha='center', va='bottom', fontsize=14, fontweight='bold', color=color)

    #number of active+quiescent drylines behind each estimate
    n_label_y = {-1: 0.015, 0: 0.045, 1: 0.015}
    n_cell = aq_df[aq_df[ENSO_TYPE] == phase].groupby('month')['n_aq'].sum().reindex(month_numbers)
    for xi, n in zip(x, n_cell):
        ax.text(xi, n_label_y[phase], f"n={n}", transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                fontsize=10, color=color, fontweight='bold')

#phase differences within a month: mark the month if the omnibus test is significant
# for pos, m in zip(month_positions, month_numbers):
#     if phase_omnibus_holm[m] < 0.05:
#         ax.text(pos, 0.97, "*", transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=24)

ax.axhline(0.5, color='black', linestyle='--', linewidth=1.25)
ax.set_ylim(0.2, 1)
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.set_xticks(month_positions)
ax.set_xticklabels(month_labels)
ax.set_xlim(-0.6, len(month_numbers) - 0.4)
ax.set_xlabel("Month", fontsize=16)
ax.set_ylabel("Proportion", fontsize=16)
ax.set_title("Proportion of All Drylines that are Cyclone-Associated\nby Month and ENSO Phase", fontsize=20)
ax.grid(axis='y', alpha=0.8)
ax.legend(title="ENSO Phase", fontsize=16, title_fontsize=16, loc=1, framealpha=1).set_zorder(100)

fig.tight_layout()
plt.savefig(f'{directory_path}/monthly_dryline_active_proportion_{YEAR_CUTOFF}.png', format='png', dpi=150, bbox_inches='tight')

rows = []
for phase, pname in phase_names.items():
    _, pairs, _ = month_results[phase]
    for _, r in pairs.iterrows():
        rows.append({"Phase": pname, "Comparison": f"{month_name_map[r['B']]} vs {month_name_map[r['A']]}",
                     "Odds ratio": round(r['odds_ratio_B/A'], 2), "p (Holm)": round(r['p_holm'], 4)})
# print(pd.DataFrame(rows).to_string(index=False))


#################################################################################################
################################################################################################# some lee cyclone summary stats
################################################################################################# number of lee cyclones by month and ENSO (violin plot with KW test?)
################################################################################################# proportion of SGP cyclone days from lee cyclones
#################################################################################################

def cliffs_delta(x, y):
    u = stats.mannwhitneyu(
        x, y, alternative="two-sided", method="asymptotic"
    ).statistic
    return 2.0 * u / (len(x) * len(y)) - 1.0

# total number of lee cyclones by month and ENSO phase (by initiated date)
if domain_type is not None:
    complex_cyclone_df = pd.read_pickle(f"{YOUR_FILE_PATH}/all_cyclones_df_{terrain_mask}_{domain_type}.pkl")
else:
    complex_cyclone_df = pd.read_pickle(f"{YOUR_FILE_PATH}/all_cyclones_df_{terrain_mask}.pkl")
lee_cyclones = complex_cyclone_df[complex_cyclone_df['is_lee_cyclone'] == True]

lee_init_time = []
year = []
month = []
for i in range(len(lee_cyclones)):
    if lee_cyclones.iloc[i]['time'][0].year < YEAR_CUTOFF:
        continue
    lee_init_time.append(lee_cyclones.iloc[i]['time'][0])
    year.append(lee_cyclones.iloc[i]['time'][0].year)
    month.append(lee_cyclones.iloc[i]['time'][0].month)

enso_op = []
enso_3mo = []
enso_1mo = []
for i in range(len(lee_init_time)):
    enso_op.append(enso_df[(enso_df['year'] == year[i]) & (enso_df['month'] == month[i])]['op_enso'].values[0])
    enso_3mo.append(enso_df[(enso_df['year'] == year[i]) & (enso_df['month'] == month[i])]['3mo_enso'].values[0])
    enso_1mo.append(enso_df[(enso_df['year'] == year[i]) & (enso_df['month'] == month[i])]['1mo_enso'].values[0])


lee_sorted = pd.DataFrame()
lee_sorted['init_time'] = lee_init_time
lee_sorted['year'] = year
lee_sorted['month'] = month
lee_sorted['op_enso'] = enso_op
lee_sorted['3mo_enso'] = enso_3mo
lee_sorted['1mo_enso'] = enso_1mo

def count_by_month_and_enso(df, month, enso_column, doall=False):
    #df = dataframe of all lee cyclones
    #month = int of the month
    #enso_column = string of the ENSO column to use ('op_enso', '3mo_enso', '1mo_enso')
    #phase = [-1, 0, 1] for ENSO phase
    #doall: whether to do for whole spring or not

    if doall == False:
        la = df[(df['month'] == month) & (df[enso_column] == -1)]
        nu = df[(df['month'] == month) & (df[enso_column] == 0)]
        el = df[(df['month'] == month) & (df[enso_column] == 1)]
        return [len(la), len(nu), len(el)]
    else:
        total_la = df[(df['month'].isin([2,3,4,5,6])) & (df[enso_column] == -1)]
        total_nu = df[(df['month'].isin([2,3,4,5,6])) & (df[enso_column] == 0)]
        total_el = df[(df['month'].isin([2,3,4,5,6])) & (df[enso_column] == 1)]
        return [len(total_la), len(total_nu), len(total_el)]

def count_enso_bymonth_categorize(df, enso_df, month, enso_column=ENSO_TYPE):
    #given a dataframe of drylines, with ENSO index for each dryline, as well as the OG enso dataframe, and a month (int), 
    #return a list of lists of dryline count for each of the three ENSO cateogires
    
    years = np.arange(YEAR_CUTOFF, 2021, 1)

    df = df[df['month'] == month]
    enso_df = enso_df[enso_df['month'] == month]

    counts = []
    for i in range(len(years)):
        temp = df[df['year'] == years[i]]
        counts.append(len(temp))
    
    mean = np.nanmean(counts)
    
    #assign counts to different ENSO categories
    el = []
    la = []
    n = []
    
    for i in range(len(years)):
        temp_enso = enso_df[enso_df['year'] == years[i]]
        
        if temp_enso.iloc[0][enso_column] == -1:
            la.append(counts[i])
        elif temp_enso.iloc[0][enso_column] == 1:
            el.append(counts[i])
        else:
            n.append(counts[i])           
            
    return [la, n, el]



feb_lee_counts = count_enso_bymonth_categorize(lee_sorted, enso_df, 2)
mar_lee_counts = count_enso_bymonth_categorize(lee_sorted, enso_df, 3)
apr_lee_counts = count_enso_bymonth_categorize(lee_sorted, enso_df, 4)
may_lee_counts = count_enso_bymonth_categorize(lee_sorted, enso_df, 5)
jun_lee_counts = count_enso_bymonth_categorize(lee_sorted, enso_df, 6)
# spring_lee_counts = count_by_month_and_enso(lee_sorted, None, enso_column='op_enso', doall=True)

def plot_quartile(ax, var):
    #given an axis and a variable, plot the interquartile range on the violin plots
    
    for i in range(len(var)):
        quartile1, medians, quartile3 = np.percentile(np.array(var[i]), [25,50,75])
        avg = np.nanmean(var[i])
        
        inds = i+1
        ax.scatter(inds, medians, marker='o', color='white', s=150, zorder=3)
        ax.vlines(inds, quartile1, quartile3, color='black', linestyle='-', lw=5)
        ax.scatter(inds, avg, marker='X', color='indianred', s=110, zorder=4)
    
    return

# violin plot of lee cyclone counts by ENSO phase for each month
fig, ((ax1,ax2), (ax3,ax4), (ax5,ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(18, 18))
axes = [ax1, ax2, ax3, ax4, ax5, ax6]

countlist = [feb_lee_counts, mar_lee_counts, apr_lee_counts, may_lee_counts, jun_lee_counts]
titlelist = ["February", "March", "April", "May", "June"]

kw_records = []
for month_name, groups in zip(titlelist, countlist):
    pooled_counts = np.concatenate(groups)

    if np.unique(pooled_counts).size == 1:
        h_stat, p_value = 0.0, 1.0
    else:
        h_stat, p_value = stats.kruskal(*groups)

    kw_records.append({
        "Month": month_name,
        "n_La_Nina": len(groups[0]),
        "n_Neutral": len(groups[1]),
        "n_El_Nino": len(groups[2]),
        "H": h_stat,
        "p_raw": p_value,
    })

kw_results = pd.DataFrame(kw_records)

# kw_results["p_holm"] = multipletests(
#     kw_results["p_raw"], method="holm")[1]
kw_results["significant"] = kw_results["p_raw"] < 0.05

print("\nKruskal-Wallis tests: ENSO phases within each month")
print(kw_results.to_string(index=False, float_format=lambda value: f"{value:.4g}"))

#May is significant, so print out Dunn's test for that month
dunn_results = sp.posthoc_dunn(may_lee_counts, p_adjust='holm')
labels = ["La Niña", "Neutral", "El Niño"]
dunn_results.index = labels
dunn_results.columns = labels

print("\nDunn test: Holm-adjusted p-values")
print(dunn_results.to_string(float_format=lambda p: f"{p:.4g}"))

for a, b in combinations(labels, 2):
    p = dunn_results.loc[a, b]
    significance = "SIGNIFICANT" if p < 0.05 else "not significant"
    print(f"{a} vs {b}: p_adj = {p:.4g} ({significance})")

labs = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
for i,ax in enumerate(axes):
    if ax != ax6:
        parts = ax.violinplot(countlist[i], showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('cornflowerblue')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
            
        plot_quartile(ax, countlist[i])

        # for x, group in enumerate(countlist[i]):
        #     ax.text(x+1, -0.5, f"N={len(group)}", fontsize=16, ha='center', bbox=dict(facecolor='white', edgecolor='black'))

        ax.text(0.7, 9, labs[i], fontsize=18, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black'))

        if ax != ax2 and ax != ax4 and ax != ax6:
            ax.set_ylabel("Count", fontsize=22)
        ax.set_title(f"{titlelist[i]}", fontsize=28)
        ax.set_ylim(-0.99, 10)
        ax.yaxis.set_major_locator(MultipleLocator((2)))
        ax.yaxis.set_minor_locator(MultipleLocator((1)))
        ax.set_xticks([1,2,3], labels=['La Nina', 'Neutral', 'El Nino'])
        ax.grid(visible=True, axis='y')
        ax.tick_params(axis='x', which='major', labelsize=18, length=7, width=3)
        ax.tick_params(axis='y', which='major', labelsize=18, length=7, width=1)
        ax.tick_params(axis='y', which='minor', length=3)

ax6.set_axis_off()
ax6.text(0.04, 0.85, labs[5], transform=ax6.transAxes, ha='left', va='bottom', fontsize=18, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black'))

table_rows = [
    [
        row.Month,
        "<0.001" if row.p_raw < 0.001 else f"{row.p_raw:.3f}",
    ]
    for row in kw_results.itertuples(index=False)]

table = ax6.table(
    cellText=table_rows,
    colLabels=["Month", "p-value"],
    cellLoc="center",
    colLoc="center",
    colWidths=[0.4, 0.6],
    bbox=[0.05, 0.20, 0.90, 0.60],)

table.auto_set_font_size(False)
table.set_fontsize(18)

for (row_index, column_index), cell in table.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(0.75)

    if row_index == 0:
        cell.set_facecolor("lightgray")
        cell.set_text_props(weight="bold")

    #May is significant, so highlight that
    if row_index == 4:
        cell.set_facecolor("lightcoral")
        # cell.set_text_props(style='italic')

#add significance indicators to plot
#for May La vs neutral only
def sig_bracket(ax, start, target, y_top, y_ends, color, gap=0.3, edge='black', lw=5, edge_lw=8, headsize=18, label=None, lee=False):

    xs = [start, target]

    segments = [([min(xs), max(xs)], [y_top, y_top])]
    segments += [([x,x], [y_top, y_end+gap]) for x, y_end in zip(xs, y_ends)]

    #outline layer under the fill hides seams
    for width, c, z in [(edge_lw, edge, 50), (lw, color, 52)]:
        for sx, sy in segments:
            ax.plot(sx, sy, color=c, linewidth=width, solid_capstyle='projecting', zorder=z)

    ax.plot(xs, y_ends, linestyle='none', marker='v', markersize=headsize, color=color, markeredgecolor=edge, markeredgewidth=1.5, zorder=51)

    if lee == True:
        buff = 0.5
    else:
        buff = 0

    if label is not None:
        ax.text(
            (start + target) / 2,
            y_top - buff,
            label,
            ha="center",
            va="center",
            fontsize=16,
            color="black",
            zorder=60,
            bbox=dict(
                boxstyle="square,pad=0.3",
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
            ),
        )


    return

comparisons = [
    ("May", "La Nina - neutral",
     may_lee_counts[0], may_lee_counts[1]),
]

rng = np.random.default_rng(42)
effect_records = []

for month, comparison, x, y in comparisons:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size < 2 or y.size < 2:
        raise ValueError(f"{month}, {comparison}: insufficient observations.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{month}, {comparison}: nonfinite counts.")

    result = stats.bootstrap(
        (x, y),
        statistic=cliffs_delta,
        vectorized=False,
        paired=False,
        confidence_level=0.95,
        n_resamples=10000,
        method="percentile",
        random_state=rng,
    )

    effect_records.append({
        "Month": month,
        "Comparison": comparison,
        "n_other": x.size,
        "n_La_Nina": y.size,
        "Cliffs_delta": cliffs_delta(x, y),
        "CI_lower": result.confidence_interval.low,
        "CI_upper": result.confidence_interval.high,
    })

effect_results = pd.DataFrame(effect_records)

print("\nSelected Lee comparisons: Cliff's delta and bootstrap 95% CIs")
print(effect_results.to_string(
    index=False,
    float_format=lambda value: f"{value:.3f}",
))

effect_labels = {
    (row.Month, row.Comparison): (
        rf"$\delta = {row.Cliffs_delta:.2f}$"
        f"\n({row.CI_lower:.2f}$\u2011${row.CI_upper:.2f})"
    )
    for row in effect_results.itertuples(index=False)
}


sig_bracket(ax4, start=1, target=2, y_top=9.5, y_ends=[7.5,8.5], color='lightcoral', label=effect_labels[("May", "La Nina - neutral")], lee=True)

ax6.set_title("Kruskal-Wallis Tests", fontsize=28, y=0.8, pad=10)

plt.suptitle("Lee Cyclone Frequency", y=0.94, fontsize=36)

plt.subplots_adjust(wspace=0.1)
plt.savefig(f'{directory_path}/enso_vs_lee_frequency_bymonth_category_{YEAR_CUTOFF}.png',dpi=100,bbox_inches='tight')


###################################################### Make violin plots of SGP cyclone frequency and dryline day frequency by month and ENSO
#with KW test and Dunn's test
allsgp_days = cyclones_df[cyclones_df['sgp_cyclone_day'] == True]

feb_sgp_counts = count_enso_bymonth_categorize(allsgp_days, enso_df, 2)
mar_sgp_counts = count_enso_bymonth_categorize(allsgp_days, enso_df, 3)
apr_sgp_counts = count_enso_bymonth_categorize(allsgp_days, enso_df, 4)
may_sgp_counts = count_enso_bymonth_categorize(allsgp_days, enso_df, 5)
jun_sgp_counts = count_enso_bymonth_categorize(allsgp_days, enso_df, 6)

feb_dryline_counts = count_enso_bymonth_categorize(spring_drylines, enso_df, 2)
mar_dryline_counts = count_enso_bymonth_categorize(spring_drylines, enso_df, 3)
apr_dryline_counts = count_enso_bymonth_categorize(spring_drylines, enso_df, 4)
may_dryline_counts = count_enso_bymonth_categorize(spring_drylines, enso_df, 5)
jun_dryline_counts = count_enso_bymonth_categorize(spring_drylines, enso_df, 6)


# violin plot of sgp cyclone counts by ENSO phase for each month
fig, ((ax1,ax2), (ax3,ax4), (ax5,ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(18, 18))
axes = [ax1, ax2, ax3, ax4, ax5, ax6]

countlist = [feb_sgp_counts, mar_sgp_counts, apr_sgp_counts, may_sgp_counts, jun_sgp_counts]
titlelist = ["February", "March", "April", "May", "June"]

kw_records = []
for month_name, groups in zip(titlelist, countlist):
    pooled_counts = np.concatenate(groups)

    if np.unique(pooled_counts).size == 1:
        h_stat, p_value = 0.0, 1.0
    else:
        h_stat, p_value = stats.kruskal(*groups)

    kw_records.append({
        "Month": month_name,
        "n_La_Nina": len(groups[0]),
        "n_Neutral": len(groups[1]),
        "n_El_Nino": len(groups[2]),
        "H": h_stat,
        "p_raw": p_value,
    })

kw_results = pd.DataFrame(kw_records)

# kw_results["p_holm"] = multipletests(
#     kw_results["p_raw"], method="holm")[1]
kw_results["significant"] = kw_results["p_raw"] < 0.05

print("\nKruskal-Wallis tests SGP Cyclone Days: ENSO phases within each month")
print(kw_results.to_string(index=False, float_format=lambda value: f"{value:.4g}"))

# #May is significant, so print out Dunn's test for that month
# dunn_results = sp.posthoc_dunn(may_lee_counts, p_adjust='holm')
# labels = ["La Niña", "Neutral", "El Niño"]
# dunn_results.index = labels
# dunn_results.columns = labels

# print("\nDunn test: Holm-adjusted p-values")
# print(dunn_results.to_string(float_format=lambda p: f"{p:.4g}"))

# for a, b in combinations(labels, 2):
#     p = dunn_results.loc[a, b]
#     significance = "SIGNIFICANT" if p < 0.05 else "not significant"
#     print(f"{a} vs {b}: p_adj = {p:.4g} ({significance})")

labs = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
for i,ax in enumerate(axes):
    if ax != ax6:
        parts = ax.violinplot(countlist[i], showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('cornflowerblue')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
            
        plot_quartile(ax, countlist[i])

        # for x, group in enumerate(countlist[i]):
        #     ax.text(x+1, 1, f"N={len(group)}", fontsize=16, ha='center', bbox=dict(facecolor='white', edgecolor='black'))

        ax.text(0.7, 19, labs[i], fontsize=18, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black'))

        if ax != ax2 and ax != ax4 and ax != ax6:
            ax.set_ylabel("Count", fontsize=22)
        ax.set_title(f"{titlelist[i]}", fontsize=28)
        ax.set_ylim(0, 21)
        ax.yaxis.set_major_locator(MultipleLocator((5)))
        ax.yaxis.set_minor_locator(MultipleLocator((1)))
        ax.set_xticks([1,2,3], labels=['La Nina', 'Neutral', 'El Nino'])
        ax.grid(visible=True, axis='y')
        ax.tick_params(axis='x', which='major', labelsize=18, length=7, width=3)
        ax.tick_params(axis='y', which='major', labelsize=18, length=7, width=1)
        ax.tick_params(axis='y', which='minor', length=3)

ax6.set_axis_off()
ax6.text(0.04, 0.85, labs[5], transform=ax6.transAxes, ha='left', va='bottom', fontsize=18, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black'))

table_rows = [
    [
        row.Month,
        "<0.001" if row.p_raw < 0.001 else f"{row.p_raw:.3f}",
    ]
    for row in kw_results.itertuples(index=False)]

table = ax6.table(
    cellText=table_rows,
    colLabels=["Month", "p-value"],
    cellLoc="center",
    colLoc="center",
    colWidths=[0.4, 0.6],
    bbox=[0.05, 0.20, 0.90, 0.60],)

table.auto_set_font_size(False)
table.set_fontsize(18)

for (row_index, column_index), cell in table.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(0.75)

    if row_index == 0:
        cell.set_facecolor("lightgray")
        cell.set_text_props(weight="bold")

    # #May is significant, so highlight that
    # if row_index == 4:
    #     cell.set_facecolor("lightcoral")
    #     # cell.set_text_props(style='italic')

#add significance indicators to plot
# sig_bracket(ax4, start=1, target=2, y_top=9.5, y_ends=[7.5,8.5], color='lightcoral')

ax6.set_title("Kruskal-Wallis Tests", fontsize=28, y=0.8, pad=10)

plt.suptitle("SGP Cyclone Day Frequency", y=0.94, fontsize=36)

plt.subplots_adjust(wspace=0.1)
plt.savefig(f'{directory_path}/enso_vs_sgp_frequency_bymonth_category_{YEAR_CUTOFF}.png',dpi=100,bbox_inches='tight')


# violin plot of dryline day counts by ENSO phase for each month
fig, ((ax1,ax2), (ax3,ax4), (ax5,ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(18, 18))
axes = [ax1, ax2, ax3, ax4, ax5, ax6]

countlist = [feb_dryline_counts, mar_dryline_counts, apr_dryline_counts, may_dryline_counts, jun_dryline_counts]
titlelist = ["February", "March", "April", "May", "June"]

kw_records = []
for month_name, groups in zip(titlelist, countlist):
    pooled_counts = np.concatenate(groups)

    if np.unique(pooled_counts).size == 1:
        h_stat, p_value = 0.0, 1.0
    else:
        h_stat, p_value = stats.kruskal(*groups)

    kw_records.append({
        "Month": month_name,
        "n_La_Nina": len(groups[0]),
        "n_Neutral": len(groups[1]),
        "n_El_Nino": len(groups[2]),
        "H": h_stat,
        "p_raw": p_value,
    })

kw_results = pd.DataFrame(kw_records)

# kw_results["p_holm"] = multipletests(
#     kw_results["p_raw"], method="holm")[1]
kw_results["significant"] = kw_results["p_raw"] < 0.05

print("\nKruskal-Wallis tests dryline Days: ENSO phases within each month")
print(kw_results.to_string(index=False, float_format=lambda value: f"{value:.4g}"))

#Feb and March is significant, so print out Dunn's test for those months
dunn_results = sp.posthoc_dunn(feb_dryline_counts, p_adjust='holm')
labels = ["La Niña", "Neutral", "El Niño"]
dunn_results.index = labels
dunn_results.columns = labels

print("\nFeb Dryline Dunn test: Holm-adjusted p-values")
print(dunn_results.to_string(float_format=lambda p: f"{p:.4g}"))

dunn_results = sp.posthoc_dunn(mar_dryline_counts, p_adjust='holm')
labels = ["La Niña", "Neutral", "El Niño"]
dunn_results.index = labels
dunn_results.columns = labels

print("\nMarch Dryline Dunn test: Holm-adjusted p-values")
print(dunn_results.to_string(float_format=lambda p: f"{p:.4g}"))

################### calculate and print the effect size
comparisons = [
    ("February", "La Nina - neutral",
     feb_dryline_counts[0], feb_dryline_counts[1]),
    ("February", "La Nina - El Nino",
     feb_dryline_counts[0], feb_dryline_counts[2]),
    ("March", "La Nina - El Nino",
     mar_dryline_counts[0], mar_dryline_counts[2]),
]

rng = np.random.default_rng(42)
effect_records = []

for month, comparison, x, y in comparisons:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size < 2 or y.size < 2:
        raise ValueError(f"{month}, {comparison}: insufficient observations.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{month}, {comparison}: nonfinite counts.")

    result = stats.bootstrap(
        (x, y),
        statistic=cliffs_delta,
        vectorized=False,
        paired=False,
        confidence_level=0.95,
        n_resamples=10000,
        method="percentile",
        random_state=rng,
    )

    effect_records.append({
        "Month": month,
        "Comparison": comparison,
        "n_other": x.size,
        "n_La_Nina": y.size,
        "Cliffs_delta": cliffs_delta(x, y),
        "CI_lower": result.confidence_interval.low,
        "CI_upper": result.confidence_interval.high,
    })

effect_results = pd.DataFrame(effect_records)

print("\nSelected dryline comparisons: Cliff's delta and bootstrap 95% CIs")
print(effect_results.to_string(
    index=False,
    float_format=lambda value: f"{value:.3f}",
))

effect_labels = {
    (row.Month, row.Comparison): (
        rf"$\delta = {row.Cliffs_delta:.2f}$"
        f"\n({row.CI_lower:.2f}$\u2011${row.CI_upper:.2f})"
    )
    for row in effect_results.itertuples(index=False)
}


labs = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
for i,ax in enumerate(axes):
    if ax != ax6:
        parts = ax.violinplot(countlist[i], showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('cornflowerblue')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
            
        plot_quartile(ax, countlist[i])

        # for x, group in enumerate(countlist[i]):
        #     ax.text(x+1, 1, f"N={len(group)}", fontsize=16, ha='center', bbox=dict(facecolor='white', edgecolor='black'))

        ax.text(0.7, 23, labs[i], fontsize=18, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black'))
        

        if ax != ax2 and ax != ax4 and ax != ax6:
            ax.set_ylabel("Count", fontsize=22)
        ax.set_title(f"{titlelist[i]}", fontsize=28)
        ax.set_ylim(-0.5, 25)
        ax.yaxis.set_major_locator(MultipleLocator((5)))
        ax.yaxis.set_minor_locator(MultipleLocator((1)))
        ax.set_xticks([1,2,3], labels=['La Nina', 'Neutral', 'El Nino'])
        ax.grid(visible=True, axis='y')
        ax.tick_params(axis='x', which='major', labelsize=18, length=7, width=3)
        ax.tick_params(axis='y', which='major', labelsize=18, length=7, width=1)
        ax.tick_params(axis='y', which='minor', length=3)


ax6.set_axis_off()
ax6.text(0.04, 0.85, labs[5], transform=ax6.transAxes, ha='left', va='bottom', fontsize=18, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black'))
table_rows = [
    [
        row.Month,
        "<0.001" if row.p_raw < 0.001 else f"{row.p_raw:.3f}",
    ]
    for row in kw_results.itertuples(index=False)]

table = ax6.table(
    cellText=table_rows,
    colLabels=["Month", "p-value"],
    cellLoc="center",
    colLoc="center",
    colWidths=[0.4, 0.6],
    bbox=[0.05, 0.20, 0.90, 0.60],)

table.auto_set_font_size(False)
table.set_fontsize(18)

for (row_index, column_index), cell in table.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(0.75)

    if row_index == 0:
        cell.set_facecolor("lightgray")
        cell.set_text_props(weight="bold")

    # #Feb and March is significant, so highlight that
    if row_index == 1 or row_index == 2:
        cell.set_facecolor("lightcoral")
        # cell.set_text_props(style='italic')

#add significance indicators to plot
#Feb La Nina vs both neutral nad El
sig_bracket(ax1, start=1.1, target=2, y_top=14, y_ends=[8,7.5], color='lightcoral', label=effect_labels[("February", "La Nina - neutral")])
sig_bracket(ax1, start=0.9, target=3, y_top=19, y_ends=[8,6], color='lightcoral', label=effect_labels[("February", "La Nina - El Nino")])

#March La vs El
sig_bracket(ax2, start=1, target=3, y_top=22, y_ends=[16,8], color='lightcoral', label=effect_labels[("March", "La Nina - El Nino")])

ax6.set_title("Kruskal-Wallis Tests", fontsize=28, y=0.8, pad=10)

plt.suptitle("Dryline Day Frequency", y=0.94, fontsize=36)

plt.subplots_adjust(wspace=0.1)
plt.savefig(f'{directory_path}/enso_vs_dryline_frequency_bymonth_category_{YEAR_CUTOFF}.png',dpi=100,bbox_inches='tight')


##################################################################
##################################################################
################################################################## violin of coincident dryline days and SGP cyclone days
##################################################################
##################################################################

def count_enso_bymonth_categorize_both(df_dry, df_sgp, enso_df, month, enso_column=ENSO_TYPE, quiescent=False):
    #given a dataframe of drylines, with ENSO index for each dryline, as well as the OG enso dataframe, and a month (int), 
    #return a list of lists of dryline count for each of the three ENSO cateogires
    
    years = np.arange(YEAR_CUTOFF, 2021, 1)

    df_dry = df_dry[df_dry['month'] == month]
    df_sgp = df_sgp[df_sgp['month'] == month]

    #only keep days in df_dry where there is df_sgp
    # or alternatively, only drylines when there is no SGP cyclone
    if quiescent == False:
        df = df_dry[df_dry['date'].isin(df_sgp['date'])]
    else:
        df = df_dry[~df_dry['date'].isin(df_sgp['date'])]
        
    enso_df = enso_df[enso_df['month'] == month]

    counts = []
    for i in range(len(years)):
        temp = df[df['year'] == years[i]]
        counts.append(len(temp))
    
    mean = np.nanmean(counts)
    
    #assign counts to different ENSO categories
    el = []
    la = []
    n = []
    
    for i in range(len(years)):
        temp_enso = enso_df[enso_df['year'] == years[i]]
        
        if temp_enso.iloc[0][enso_column] == -1:
            la.append(counts[i])
        elif temp_enso.iloc[0][enso_column] == 1:
            el.append(counts[i])
        else:
            n.append(counts[i])           
            
    return [la, n, el]



feb_drycyclone_counts = count_enso_bymonth_categorize_both(spring_drylines, allsgp_days, enso_df, 2)
mar_drycyclone_counts = count_enso_bymonth_categorize_both(spring_drylines, allsgp_days, enso_df, 3)
apr_drycyclone_counts = count_enso_bymonth_categorize_both(spring_drylines, allsgp_days, enso_df, 4)
may_drycyclone_counts = count_enso_bymonth_categorize_both(spring_drylines, allsgp_days, enso_df, 5)
jun_drycyclone_counts = count_enso_bymonth_categorize_both(spring_drylines, allsgp_days, enso_df, 6)

# violin plot of dryline day counts by ENSO phase for each month
fig, ((ax1,ax2), (ax3,ax4), (ax5,ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(18, 18))
axes = [ax1, ax2, ax3, ax4, ax5, ax6]

countlist = [feb_drycyclone_counts, mar_drycyclone_counts, apr_drycyclone_counts, may_drycyclone_counts, jun_drycyclone_counts]
titlelist = ["February", "March", "April", "May", "June"]

kw_records = []
for month_name, groups in zip(titlelist, countlist):
    pooled_counts = np.concatenate(groups)

    if np.unique(pooled_counts).size == 1:
        h_stat, p_value = 0.0, 1.0
    else:
        h_stat, p_value = stats.kruskal(*groups)

    kw_records.append({
        "Month": month_name,
        "n_La_Nina": len(groups[0]),
        "n_Neutral": len(groups[1]),
        "n_El_Nino": len(groups[2]),
        "H": h_stat,
        "p_raw": p_value,
    })

kw_results = pd.DataFrame(kw_records)

# kw_results["p_holm"] = multipletests(
#     kw_results["p_raw"], method="holm")[1]
kw_results["significant"] = kw_results["p_raw"] < 0.05

print("\nKruskal-Wallis tests cyclone + dryline Days: ENSO phases within each month")
print(kw_results.to_string(index=False, float_format=lambda value: f"{value:.4g}"))


#Feb and March is significant, so print out Dunn's test for those months
dunn_results = sp.posthoc_dunn(feb_drycyclone_counts, p_adjust='holm')
labels = ["La Niña", "Neutral", "El Niño"]
dunn_results.index = labels
dunn_results.columns = labels

print("\nFeb Dryline Dunn test: Holm-adjusted p-values")
print(dunn_results.to_string(float_format=lambda p: f"{p:.4g}"))

dunn_results = sp.posthoc_dunn(mar_drycyclone_counts, p_adjust='holm')
labels = ["La Niña", "Neutral", "El Niño"]
dunn_results.index = labels
dunn_results.columns = labels

print("\nMarch Dryline Dunn test: Holm-adjusted p-values")
print(dunn_results.to_string(float_format=lambda p: f"{p:.4g}"))

# for a, b in combinations(labels, 2):
#     p = dunn_results.loc[a, b]
#     significance = "SIGNIFICANT" if p < 0.05 else "not significant"
#     print(f"{a} vs {b}: p_adj = {p:.4g} ({significance})")


labs = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
for i,ax in enumerate(axes):
    if ax != ax6:
        parts = ax.violinplot(countlist[i], showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('cornflowerblue')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
            
        plot_quartile(ax, countlist[i])

        # for x, group in enumerate(countlist[i]):
        #     ax.text(x+1, 1, f"N={len(group)}", fontsize=16, ha='center', bbox=dict(facecolor='white', edgecolor='black'))

        ax.text(0.7, 18, labs[i], fontsize=18, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black'))
        

        if ax != ax2 and ax != ax4 and ax != ax6:
            ax.set_ylabel("Count", fontsize=22)
        ax.set_title(f"{titlelist[i]}", fontsize=28)
        ax.set_ylim(-0.5, 20)
        ax.yaxis.set_major_locator(MultipleLocator((5)))
        ax.yaxis.set_minor_locator(MultipleLocator((1)))
        ax.set_xticks([1,2,3], labels=['La Nina', 'Neutral', 'El Nino'])
        ax.grid(visible=True, axis='y')
        ax.tick_params(axis='x', which='major', labelsize=18, length=7, width=3)
        ax.tick_params(axis='y', which='major', labelsize=18, length=7, width=1)
        ax.tick_params(axis='y', which='minor', length=3)


ax6.set_axis_off()
ax6.text(0.04, 0.85, labs[5], transform=ax6.transAxes, ha='left', va='bottom', fontsize=18, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black'))
table_rows = [
    [
        row.Month,
        "<0.001" if row.p_raw < 0.001 else f"{row.p_raw:.3f}",
    ]
    for row in kw_results.itertuples(index=False)]

table = ax6.table(
    cellText=table_rows,
    colLabels=["Month", "p-value"],
    cellLoc="center",
    colLoc="center",
    colWidths=[0.4, 0.6],
    bbox=[0.05, 0.20, 0.90, 0.60],)

table.auto_set_font_size(False)
table.set_fontsize(18)

for (row_index, column_index), cell in table.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(0.75)

    if row_index == 0:
        cell.set_facecolor("lightgray")
        cell.set_text_props(weight="bold")

    # #Feb and March is significant, so highlight that
    if row_index == 1 or row_index == 2:
        cell.set_facecolor("lightcoral")
        # cell.set_text_props(style='italic')

#add significance indicators to plot
#Feb La Nina vs both neutral nad El
sig_bracket(ax1, start=1.1, target=2, y_top=10, y_ends=[7,6], color='lightcoral')
sig_bracket(ax1, start=0.9, target=3, y_top=13, y_ends=[7,3], color='lightcoral')

#March La vs El
sig_bracket(ax2, start=1, target=3, y_top=12, y_ends=[9,7], color='lightcoral')

ax6.set_title("Kruskal-Wallis Tests", fontsize=28, y=0.8, pad=10)

plt.suptitle("SGP Cyclone + Dryline Day Frequency", y=0.94, fontsize=36)

plt.subplots_adjust(wspace=0.1)
plt.savefig(f'{directory_path}/enso_vs_cyclone+dryline_frequency_bymonth_category_{YEAR_CUTOFF}.png',dpi=100,bbox_inches='tight')


feb_drycyclone_counts = count_enso_bymonth_categorize_both(spring_drylines, allsgp_days, enso_df, 2, quiescent=True)
mar_drycyclone_counts = count_enso_bymonth_categorize_both(spring_drylines, allsgp_days, enso_df, 3, quiescent=True)
apr_drycyclone_counts = count_enso_bymonth_categorize_both(spring_drylines, allsgp_days, enso_df, 4, quiescent=True)
may_drycyclone_counts = count_enso_bymonth_categorize_both(spring_drylines, allsgp_days, enso_df, 5, quiescent=True)
jun_drycyclone_counts = count_enso_bymonth_categorize_both(spring_drylines, allsgp_days, enso_df, 6, quiescent=True)

# violin plot of dryline day counts by ENSO phase for each month
fig, ((ax1,ax2), (ax3,ax4), (ax5,ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(18, 18))
axes = [ax1, ax2, ax3, ax4, ax5, ax6]

countlist = [feb_drycyclone_counts, mar_drycyclone_counts, apr_drycyclone_counts, may_drycyclone_counts, jun_drycyclone_counts]
titlelist = ["February", "March", "April", "May", "June"]

kw_records = []
for month_name, groups in zip(titlelist, countlist):
    pooled_counts = np.concatenate(groups)

    if np.unique(pooled_counts).size == 1:
        h_stat, p_value = 0.0, 1.0
    else:
        h_stat, p_value = stats.kruskal(*groups)

    kw_records.append({
        "Month": month_name,
        "n_La_Nina": len(groups[0]),
        "n_Neutral": len(groups[1]),
        "n_El_Nino": len(groups[2]),
        "H": h_stat,
        "p_raw": p_value,
    })

kw_results = pd.DataFrame(kw_records)

# kw_results["p_holm"] = multipletests(
#     kw_results["p_raw"], method="holm")[1]
kw_results["significant"] = kw_results["p_raw"] < 0.05

print("\nKruskal-Wallis tests dryline day without cyclone: ENSO phases within each month")
print(kw_results.to_string(index=False, float_format=lambda value: f"{value:.4g}"))


#Feb and March is significant, so print out Dunn's test for those months
dunn_results = sp.posthoc_dunn(feb_drycyclone_counts, p_adjust='holm')
labels = ["La Niña", "Neutral", "El Niño"]
dunn_results.index = labels
dunn_results.columns = labels

print("\nFeb Dryline Dunn test: Holm-adjusted p-values")
print(dunn_results.to_string(float_format=lambda p: f"{p:.4g}"))

dunn_results = sp.posthoc_dunn(mar_drycyclone_counts, p_adjust='holm')
labels = ["La Niña", "Neutral", "El Niño"]
dunn_results.index = labels
dunn_results.columns = labels

print("\nMarch Dryline Dunn test: Holm-adjusted p-values")
print(dunn_results.to_string(float_format=lambda p: f"{p:.4g}"))

# for a, b in combinations(labels, 2):
#     p = dunn_results.loc[a, b]
#     significance = "SIGNIFICANT" if p < 0.05 else "not significant"
#     print(f"{a} vs {b}: p_adj = {p:.4g} ({significance})")


labs = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
for i,ax in enumerate(axes):
    if ax != ax6:
        parts = ax.violinplot(countlist[i], showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('cornflowerblue')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
            
        plot_quartile(ax, countlist[i])

        # for x, group in enumerate(countlist[i]):
        #     ax.text(x+1, 1, f"N={len(group)}", fontsize=16, ha='center', bbox=dict(facecolor='white', edgecolor='black'))

        ax.text(0.7, 18, labs[i], fontsize=18, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black'))
        

        if ax != ax2 and ax != ax4 and ax != ax6:
            ax.set_ylabel("Count", fontsize=22)
        ax.set_title(f"{titlelist[i]}", fontsize=28)
        ax.set_ylim(-0.5, 20)
        ax.yaxis.set_major_locator(MultipleLocator((5)))
        ax.yaxis.set_minor_locator(MultipleLocator((1)))
        ax.set_xticks([1,2,3], labels=['La Nina', 'Neutral', 'El Nino'])
        ax.grid(visible=True, axis='y')
        ax.tick_params(axis='x', which='major', labelsize=18, length=7, width=3)
        ax.tick_params(axis='y', which='major', labelsize=18, length=7, width=1)
        ax.tick_params(axis='y', which='minor', length=3)


ax6.set_axis_off()
ax6.text(0.04, 0.85, labs[5], transform=ax6.transAxes, ha='left', va='bottom', fontsize=18, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black'))
table_rows = [
    [
        row.Month,
        "<0.001" if row.p_raw < 0.001 else f"{row.p_raw:.3f}",
    ]
    for row in kw_results.itertuples(index=False)]

table = ax6.table(
    cellText=table_rows,
    colLabels=["Month", "p-value"],
    cellLoc="center",
    colLoc="center",
    colWidths=[0.4, 0.6],
    bbox=[0.05, 0.20, 0.90, 0.60],)

table.auto_set_font_size(False)
table.set_fontsize(18)

for (row_index, column_index), cell in table.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(0.75)

    if row_index == 0:
        cell.set_facecolor("lightgray")
        cell.set_text_props(weight="bold")

    # #Feb and March is significant, so highlight that
    if row_index == 1 or row_index == 2:
        cell.set_facecolor("lightcoral")
        # cell.set_text_props(style='italic')

#add significance indicators to plot
#Feb La Nina vs both neutral nad El
sig_bracket(ax1, start=1.1, target=2, y_top=10, y_ends=[7,6], color='lightcoral')
sig_bracket(ax1, start=0.9, target=3, y_top=13, y_ends=[7,3], color='lightcoral')

#March La vs El
sig_bracket(ax2, start=1, target=3, y_top=12, y_ends=[9,7], color='lightcoral')

ax6.set_title("Kruskal-Wallis Tests", fontsize=28, y=0.8, pad=10)

plt.suptitle("Dryline Day (without SGP cyclone) Frequency", y=0.94, fontsize=36)

plt.subplots_adjust(wspace=0.1)
plt.savefig(f'{directory_path}/enso_vs_dryline_without_cyclone_frequency_bymonth_category_{YEAR_CUTOFF}.png',dpi=100,bbox_inches='tight')




############################################## what proportion of SGP cyclone days come from a lee cyclone? By month and ENSO
years = np.arange(YEAR_CUTOFF, 2021, 1)
months = np.arange(2, 7, 1)

allsgp_days = cyclones_df[cyclones_df['sgp_cyclone_day'] == True]


def binomial_share_compare(sgp_days, group_col, levels):
    # sgp_days must contain only SGP cyclone-day rows.
    df = sgp_days.copy()

    df["lee_sgp_day"] = df.apply(
        lambda row: any(
            sgp and lee
            for sgp, lee in zip(row["sgp_cyclone"], row["is_lee_cyclone"])),axis=1)

    # Aggregate daily successes and trials within each year and group.
    grouped = (
        df.groupby(["year", group_col], observed=True).agg(
            lee_days=("lee_sgp_day", "sum"),
            sgp_days=("lee_sgp_day", "size")).reset_index())
    
    grouped["nonlee_days"] = grouped["sgp_days"] - grouped["lee_days"]

    missing_levels = [level for level in levels if not grouped[group_col].eq(level).any()]
    if missing_levels:
        raise ValueError(
            f"No SGP cyclone days for {group_col} level(s): {missing_levels}")

    grouped[group_col] = pd.Categorical(grouped[group_col], categories=levels)

    group_term = f"C(Q({group_col!r}))"
    result = smf.glm(
        f"lee_days + nonlee_days ~ {group_term}",
        data=grouped,
        family=sm.families.Binomial()).fit(
        cov_type="cluster",
        cov_kwds={"groups": grouped["year"]})

    omnibus_p = float(
        result.wald_test_terms(scalar=True).table.loc[group_term, "pvalue"])

    names = list(result.params.index)

    def term(level):
        return f"{group_term}[T.{level}]"

    pairwise_rows = []
    for a, b in combinations(levels, 2):
        contrast = np.zeros(len(names))
        if a != levels[0]:
            contrast[names.index(term(a))] -= 1
        if b != levels[0]:
            contrast[names.index(term(b))] += 1

        test = result.t_test(contrast)
        pairwise_rows.append({
            "A": a,
            "B": b,
            "odds_ratio_B/A": float(np.exp(test.effect[0])),
            "p": float(test.pvalue),
        })

    pairs = pd.DataFrame(pairwise_rows)
    pairs["p_holm"] = multipletests(pairs["p"], method="holm")[1]

    prediction_data = pd.DataFrame({
        group_col: pd.Categorical(levels, categories=levels)
    })
    proportions = result.get_prediction(prediction_data).summary_frame()
    proportions.index = levels

    return (
        omnibus_p,
        pairs,
        proportions[["mean", "mean_ci_lower", "mean_ci_upper"]])


#compare both within a month across ENSO phases, and across months of the same ENSO phase
phase_results = {
    month: binomial_share_compare(
        allsgp_days[allsgp_days["month"] == month],
        ENSO_TYPE, [0, -1, 1]) for month in months}

month_results = {
    phase: binomial_share_compare(
        allsgp_days[allsgp_days[ENSO_TYPE] == phase],
        "month", months) for phase in [-1, 0, 1]}

###### print out
phase_labels = {-1: "La Niña", 0: "Neutral", 1: "El Niño"}
month_labels = {
    2: "February", 3: "March", 4: "April", 5: "May", 6: "June"
}

share_rows = []
pair_rows = []

# ENSO phases compared within each month
for month, (omnibus_p, pairs, props) in phase_results.items():
    for phase, row in props.iterrows():
        share_rows.append({
            "Comparison": f"ENSO phase, {month_labels[month]}",
            "Group": phase_labels[phase],
            "Estimated share": row["mean"],
            "95% CI": f'{row["mean_ci_lower"]:.3f}–{row["mean_ci_upper"]:.3f}',
            "Omnibus p": omnibus_p,
        })

    for _, row in pairs.iterrows():
        pair_rows.append({
            "Comparison": f"ENSO phase, {month_labels[month]}",
            "A": phase_labels[row["A"]],
            "B": phase_labels[row["B"]],
            "Odds ratio (B/A)": row["odds_ratio_B/A"],
            "Holm-adjusted p": row["p_holm"],
        })

# Months compared within each ENSO phase
for phase, (omnibus_p, pairs, props) in month_results.items():
    for month, row in props.iterrows():
        share_rows.append({
            "Comparison": f"Month, {phase_labels[phase]}",
            "Group": month_labels[month],
            "Estimated share": row["mean"],
            "95% CI": f'{row["mean_ci_lower"]:.3f}–{row["mean_ci_upper"]:.3f}',
            "Omnibus p": omnibus_p,
        })

    for _, row in pairs.iterrows():
        pair_rows.append({
            "Comparison": f"Month, {phase_labels[phase]}",
            "A": month_labels[row["A"]],
            "B": month_labels[row["B"]],
            "Odds ratio (B/A)": row["odds_ratio_B/A"],
            "Holm-adjusted p": row["p_holm"],
        })

share_table = pd.DataFrame(share_rows)
pairwise_table = pd.DataFrame(pair_rows)

######## COMMENTED OUT SO IT DOESN'T CLOG UP TERMINAL
# print("\nEstimated share of SGP cyclone days that were Lee cyclone days")
# print(share_table.to_string(
#     index=False,
#     formatters={
#         "Estimated share": "{:.3f}".format,
#         "Omnibus p": "{:.4g}".format,
#     },
# ))

# print("\nPairwise comparisons")
# print(pairwise_table.to_string(
#     index=False,
#     formatters={
#         "Odds ratio (B/A)": "{:.2f}".format,
#         "Holm-adjusted p": "{:.4g}".format,
#     },
# ))
#############
#make the plot 
month_numbers = [2,3,4,5,6]
month_labels = ["February", "March", "April", "May", "June"]
month_positions = np.arange(len(month_numbers))

phase_settings = [(-1, "La Nina", "royalblue", -0.25),
                  (0, "Neutral", "gray", 0),
                  (1, "El Nino", "firebrick", 0.25)]

fig, ax = plt.subplots(figsize=(8, 8))
markers = {-1: "o", 0: "X", 1: "s"}

for phase, label, color, offset in phase_settings:
    omnibus_p, pairs, pred = month_results[phase]
    x = month_positions + offset
    y = pred['mean'].to_numpy()
    lo, hi = pred['mean_ci_lower'].to_numpy(), pred['mean_ci_upper'].to_numpy()

    # ax.plot(x, y, color=color, linewidth=1.5, alpha=0.5, zorder=4)
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt=markers[phase], color=color, markersize=11,
                markeredgecolor='black', ecolor=color, elinewidth=2, capsize=5, capthick=2, label=label, zorder=5)

    # if month_omnibus_holm[phase] < 0.05:
    #     letters = compact_letters(month_numbers, pairs)
    #     for xi, top, mo in zip(x, hi, month_numbers):
    #         ax.text(xi, top + 0.02, letters[mo], ha='center', va='bottom', fontsize=14, fontweight='bold', color=color)

    #number of SGP cyclone daysbehind each estimate
    n_label_y = {-1: 0.015, 0: 0.045, 1: 0.015}
    n_cell = [len(allsgp_days[(allsgp_days[ENSO_TYPE] == phase) & (allsgp_days['month'] == m)]) for m in month_numbers]
    for xi, n in zip(x, n_cell):
        ax.text(xi, n_label_y[phase], f"n={n}", transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                fontsize=10, color=color, fontweight='bold')

#phase differences within a month: mark the month if the omnibus test is significant
# for pos, m in zip(month_positions, month_numbers):
#     if phase_omnibus_holm[m] < 0.05:
#         ax.text(pos, 0.97, "*", transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=24)


# allsgp_days["lee_sgp_day"] = allsgp_days.apply(
#         lambda row: any(
#             sgp and lee
#             for sgp, lee in zip(row["sgp_cyclone"], row["is_lee_cyclone"])),axis=1)
# #plot the observed proportion of SGP cyclone days from lee cyclones for each phase and month (across all years)
# for phase, label, color, offset in phase_settings:
#     for m, pos in zip(month_numbers, month_positions):
#         obs_prop = len(allsgp_days[(allsgp_days['op_enso'] == phase) & (allsgp_days['month'] == m) & (allsgp_days['lee_sgp_day'])]) / len(allsgp_days[(allsgp_days['op_enso'] == phase) & (allsgp_days['month'] == m)])
#         ax.scatter(pos + offset, obs_prop, color=color, edgecolor='black', s=100, zorder=6)

#add marker for significance based on omnibus p-value
#only May
# ax.scatter(month_positions[3], 0.2, marker="*", color='black', s=500, zorder=500)

# #neutral and El Nino
# ax.scatter(0.4, 0.78, marker="*", color='black', s=400, zorder=500)
# ax.scatter(0.42, 0.81, marker="*", color='black', s=400, zorder=500)


ax.axhline(0.5, color='black', linestyle='--', linewidth=1.25)
ax.set_ylim(0.1, 0.9)
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.set_xticks(month_positions)
ax.set_xticklabels(month_labels)
ax.set_xlim(-0.6, len(month_numbers) - 0.4)
ax.set_xlabel("Month", fontsize=16)
ax.set_ylabel("Proportion", fontsize=16)
ax.set_title("Proportion of SGP Cyclone Days\nfrom Lee Cyclones", fontsize=20)
ax.grid(axis='y', alpha=0.8)
ax.legend(title="ENSO Phase", fontsize=16, title_fontsize=16, loc=2, framealpha=1).set_zorder(100)

fig.tight_layout()
plt.savefig(f'{directory_path}/monthly_sgp_cyclone_proportion_{YEAR_CUTOFF}.png', format='png', dpi=150, bbox_inches='tight')

#################################################################################################
#################################################################################################
################################################################################################# negative binomial regression
################################################################################################# given monthly cyclone frequency, predict monthly dryline counts
#################################################################################################
do_the_model = True

if do_the_model == True:
    #test for overdispersion (is variance greater than mean?)
    # print('Feb overdispersion test: mean =', feb_dry.mean(), 'variance =', feb_dry.var())
    # print('Mar overdispersion test: mean =', march_dry.mean(), 'variance =', march_dry.var())
    # print('Apr overdispersion test: mean =', april_dry.mean(), 'variance =', april_dry.var())
    # print('May overdispersion test: mean =', may_dry.mean(), 'variance =', may_dry.var())
    # print('Jun overdispersion test: mean =', june_dry.mean(), 'variance =', june_dry.var())
    # print('All overdispersion test: mean =', allspring.mean(), 'variance =', allspring.var())

    #predictors: ENSO, SGP cyclone count, "interaction (ENSO x GP cyclone count)"

    # gp = False
    # if gp == True:
    #     sgp = False
    # else:
    #     sgp = True

    #first, create a df that contains yearly/monthly cyclone and dryline counts, as well as ENSO defintiions
    model_df = pd.DataFrame()

    years = np.arange(YEAR_CUTOFF, 2021, 1)
    months = np.arange(2, 7, 1)

    yr = []
    mo = []
    lc = []
    dc = []
    op_enso = []
    enso_3mo = []
    enso_1mo = []



    for y in years:
        for m in months:
            #get the number of S(GP) cyclones for this year and month
            cyclone_count = len(cyclones_df[(cyclones_df['year'] == y) & (cyclones_df['month'] == m) & (cyclones_df['sgp_cyclone'].map(any))])
            # if gp ==  True:
            #     cyclone_count = len(cyclones_df[(cyclones_df['year'] == y) & (cyclones_df['month'] == m) & (cyclones_df['lee_centroid'].map(any))])
            # else:
            #     cyclone_count = len(cyclones_df[(cyclones_df['year'] == y) & (cyclones_df['month'] == m) & (cyclones_df['sgp_centroid'].map(any))])

            #get the number of drylines for this year and month
            # dryline_count = len(spring_drylines[(spring_drylines['year'] == y) & (spring_drylines['month'] == m)])

            #get the total number of dryline days for this year and month
            total_d_count =(len(spring_drylines[(spring_drylines['year'] == y) & (spring_drylines['month'] == m)]))

            #dryline_within_cyclone count
            dwc_count =(len(cyclones_df[(cyclones_df['year'] == y) & (cyclones_df['month'] == m) & (cyclones_df['dryline_within_cyclone'].map(any))]))

            #dryline_associated_cyclone count
            dac_count =(len(cyclones_df[(cyclones_df['year'] == y) & (cyclones_df['month'] == m) & (cyclones_df['dryline_associated_cyclone'].map(any) & (cyclones_df['dryline_within_cyclone'].map(any)==False))]))

            #find difference to get quiescent cyclone count
            dq_count = (total_d_count - dwc_count - dac_count)

            #get the ENSO phase for this year and month 
            op_enso_phase = enso_df[(enso_df['year'] == y) & (enso_df['month'] == m)]['op_enso'].values[0]
            enso_phase_3m = enso_df[(enso_df['year'] == y) & (enso_df['month'] == m)]['3mo_enso'].values[0]
            enso_phase_1m = enso_df[(enso_df['year'] == y) & (enso_df['month'] == m)]['1mo_enso'].values[0]

            #append to the model dataframe
            yr.append(y)
            mo.append(m)
            lc.append(cyclone_count)
            dc.append(total_d_count)
            op_enso.append(op_enso_phase)
            enso_3mo.append(enso_phase_3m)
            enso_1mo.append(enso_phase_1m)

    model_df['year'] = yr
    model_df['month'] = mo
    model_df['cyclone_count'] = lc
    model_df['dryline_count'] = dc
    model_df['op_enso'] = op_enso
    model_df['3mo_enso'] = enso_3mo
    model_df['1mo_enso'] = enso_1mo

    model_df["days"] = pd.to_datetime(model_df[["year", "month"]].assign(day=1)).dt.days_in_month #number of days for the given month
    model_df['cyclone_anomaly'] = model_df['cyclone_count'] - model_df.groupby('month')['cyclone_count'].transform('mean')  #cyclone count minus the historical average cyclone count for that month

    #define "phase" NOTE: if wanting to test other ENSO defintions, will need to change ENSO column here
    model_df['phase'] = pd.Categorical(model_df[ENSO_TYPE].map({0: "Neutral", -1: "La Nina", 1: "El Nino"}), categories=["Neutral", "La Nina", "El Nino"])

    # #entire spring
    # formula = "dryline_count ~ C(month, Treatment(reference=5)) + C(phase, Treatment(reference='Neutral')) * cyclone_anomaly"
    # #the * means it'll do both main effects plus inteactions: C(phase) + cyclone_anomaly + C(phase):cyclone_anomaly
    # #"treatment" explicitly defines the reference month/phase for the category
    #     #define neutral phase and May as references
    # #might want to consider adding a third * with C(month, Treatment(reference=5)) to capture interactions between month and cyclone anomaly

    # result = smf.negativebinomial(formula, data=model_df, offset=np.log(model_df["days"])).fit(disp=False,cov_type='cluster', cov_kwds={'groups': model_df['year']})

    # print(result.summary())
    

    feb_df = model_df[model_df['month'] == 2]
    mar_df = model_df[model_df['month'] == 3]
    apr_df = model_df[model_df['month'] == 4]
    may_df = model_df[model_df['month'] == 5]
    jun_df = model_df[model_df['month'] == 6]

    formula = "dryline_count ~ C(phase, Treatment(reference='Neutral')) * cyclone_anomaly"

    for month_name, monthly_data in [
        ("February", feb_df),
        ("April", apr_df),
        ("March", mar_df),
        ("May", may_df),
        ("June", jun_df)]:

        #print the pearson dispersion for the poisson model
        poisson_result = smf.poisson(
                formula,
                data=monthly_data,
                offset=np.log(monthly_data["days"]),
            ).fit(disp=False, maxiter=200)
        
        expected_count = poisson_result.predict()
        observed_count = poisson_result.model.endog
        pearson_dispersion = np.sum(
            (observed_count - expected_count) ** 2 / expected_count) / poisson_result.df_resid

        print(month_name)
        # print("Converged:", poisson_result.mle_retvals["converged"])
        print("Pearson dispersion:", pearson_dispersion)

        #keep the Poisson model for April, since it's close to 1
        if month_name in ["April"] and YEAR_CUTOFF != 1940:
            result = poisson_result
        elif month_name in ["February", "April", "May"] and YEAR_CUTOFF == 1979:
            result = poisson_result

        #else, use the negative binomial model
        else:
            result = smf.negativebinomial(
                formula,
                data=monthly_data,
                offset=np.log(monthly_data["days"]),
            ).fit(disp=False, maxiter=200)

        #save out the model for each month
        if month_name == "February":
            feb_result = result
        elif month_name == "March":
            mar_result = result
        elif month_name == "April":
            apr_result = result
        elif month_name == "May":
            may_result = result
        elif month_name == "June":
            jun_result = result


    ########### assess the ability of the model to represent observed counts

    #scatter plot showing observed dryline counts vs lee cyclone days, colored by ENSO
    #fit the expected-count curve for each phase, with confidence bands

    def build_prediction_df(df, phase):
        #Given a dataframe, build a prediction dataframe for plotting
        #this prediction dataframe will contain cyclone_anomaly from minimum to maximum (based off of obs) at 0.1 increments
        #and will also provide only for one specific ENSO phase (so need to run separately for El, La, and nuetral)

        df = df[df['phase']==phase]

        cyclone_min = df["cyclone_anomaly"].min()
        cyclone_max = df["cyclone_anomaly"].max()

        cyclone_range = np.arange(cyclone_min, cyclone_max + 0.1, 0.1)
        enso_phase = [phase] * len(cyclone_range)

        predict_df = pd.DataFrame({
            "cyclone_anomaly": cyclone_range,
            "phase": enso_phase
        })

        return predict_df

    def plot_prediction(ax, df, result):
        #given an axis, dataframe, and model, plot the El/Nu/La predictions
        phase_colors = {
        "La Nina": "royalblue",
        "Neutral": "gray",
        "El Nino": "firebrick"}

        phase_shapes = {
            "La Nina": "o",
            "Neutral": "X",
            "El Nino": "s"}

        prediction_days = df['days'].mean()
        for phase, color in phase_colors.items():
            phase_df = df[df['phase'] == phase]
            
            predict_df = build_prediction_df(df, phase)

            prediction = result.get_prediction(exog=predict_df, offset=np.full(len(predict_df), np.log(prediction_days)))
            confidence_interval = prediction.conf_int(alpha=0.05)

            predict_df['predicted_mean'] = prediction.predicted
            predict_df['conf_lower'] = confidence_interval[:, 0]
            predict_df['conf_upper'] = confidence_interval[:, 1]

            ax.plot(predict_df["cyclone_anomaly"], predict_df["predicted_mean"], color=color, linewidth=2, zorder=10)
            ax.fill_between(predict_df["cyclone_anomaly"], predict_df["conf_lower"], predict_df["conf_upper"], color=color, alpha=0.3, zorder=10)

            if phase == "La Nina":
                s=80
            elif phase == "Neutral":
                s=60
            elif phase == "El Nino":
                s=30

            ax.scatter(phase_df["cyclone_anomaly"], phase_df["dryline_count"], color=color, edgecolors='black', marker=phase_shapes[phase], s=s, linewidths=0.75, zorder=5)

        return

    def monthly_irr_table(monthly_results, alpha=0.05):
        # calculate cyclonoe-day incidence rate ratios (with 95% CI) (if cyclone anomaly increased by 1, what would the % change in dryline counts be?)
        # also calculate a joint interaction p-value (tests whether the slopes differ among enso phases)

        phase_term = "C(phase, Treatment(reference='Neutral'))"
        interaction_names = {phase: f"{phase_term}[T.{phase}]:cyclone_anomaly" for phase in ["La Nina", "El Nino"]}

        rows = []
        for month, result in monthly_results.items():
            names = list(result.params.index)
            required = ['cyclone_anomaly', *interaction_names.values()]
            missing = [name for name in required if name not in names]
            if missing:
                raise ValueError(f"Missing required terms in month {month}: {missing}")

            row = {"Month": month, "Model": "NB" if "alpha" in names else "Poisson", "N": int(result.nobs)}
            for phase in ["La Nina", "Neutral", "El Nino"]:
                contrast = np.zeros(len(names))
                contrast[names.index('cyclone_anomaly')] = 1
                if phase != "Neutral":
                    contrast[names.index(interaction_names[phase])] = 1
                
                slope_test = result.t_test(contrast, use_t=False)
                log_irr = np.asarray(slope_test.effect).item()
                lower, upper = slope_test.conf_int(alpha=alpha)[0]

                row[f"{phase} IRR"] = np.exp(log_irr)
                row[f"{phase} CI Lower"] = np.exp(lower)
                row[f"{phase} CI Upper"] = np.exp(upper)

            restrictions = np.zeros((2, len(names)))
            for position, name in enumerate(interaction_names.values()):
                restrictions[position, names.index(name)] = 1

            interaction_test = result.wald_test(restrictions, use_f=False, scalar=True)
            row["Interaction p-value"] = np.asarray(interaction_test.pvalue).item()
            rows.append(row)



        return pd.DataFrame(rows).set_index("Month")


    fig, ((ax1,ax2), (ax3,ax4), (ax5,ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(18, 18))

    plot_prediction(ax1, feb_df, feb_result)
    plot_prediction(ax2, mar_df, mar_result)
    plot_prediction(ax3, apr_df, apr_result)
    plot_prediction(ax4, may_df, may_result)
    plot_prediction(ax5, jun_df, jun_result)



    ax1.set_title("February (NB)", fontsize=28)
    ax2.set_title("March (NB)", fontsize=28)
    ax3.set_title("April (P)", fontsize=28)
    ax4.set_title("May (NB)", fontsize=28)
    ax5.set_title("June (NB)", fontsize=28)


    ax1.set_ylabel("Dryline Counts", fontsize=20)
    ax3.set_ylabel("Dryline Counts", fontsize=20)
    ax5.set_ylabel("Dryline Counts", fontsize=20)

    # if gp == True:
    #     ax4.set_xlabel("GP Cyclone Day Anomaly", fontsize=16)
    #     ax5.set_xlabel("GP Cyclone Day Anomaly", fontsize=16)
    
    ax4.set_xlabel("SGP Cyclone Day Anomaly", fontsize=20)
    ax5.set_xlabel("SGP Cyclone Day Anomaly", fontsize=20)

    ax6.set_axis_off()

    #for GP cyclone
    # if gp == True:
    #     ax1.set_xlim(-5.5, 5.5)
    #     ax2.set_xlim(-7.5, 7.5)
    #     ax3.set_xlim(-7.5, 7.5)
    #     ax4.set_xlim(-6.5, 8.5)
    #     ax5.set_xlim(-9.5, 8.5)

    #     ax1.set_ylim(-0.5, 9)
    #     ax2.set_ylim(-0.5, 16)
    #     ax3.set_ylim(-0.5, 21)
    #     ax4.set_ylim(-0.5, 27)
    #     ax5.set_ylim(-0.5, 24)
    # else: #for SGP cyclone
    ax1.set_xlim(-5.5, 6.5)
    ax2.set_xlim(-5.5, 6.5)
    ax3.set_xlim(-7.5, 7)
    ax4.set_xlim(-6.5, 8.5)
    ax5.set_xlim(-9, 10)

    ax1.set_ylim(-0.5, 9.5)
    ax2.set_ylim(-0.5, 17)
    ax3.set_ylim(-0.5, 19)
    ax4.set_ylim(-0.5, 26)
    ax5.set_ylim(-0.5, 28)


    axes = [ax1,ax2,ax3,ax4,ax5]
    panel_labels = ["a)", "b)", "c)", "d)", "e)"]
    for i, ax in enumerate(axes):
        ax.text(0.05, 1.1, panel_labels[i], transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right', bbox=dict(facecolor='white', edgecolor='black'))
        ax.grid(visible=True, zorder=10000)
        ax.grid(visible=True, zorder=10000)
        if ax in [ax1, ax2, ax3]:
            ax.yaxis.set_major_locator(MultipleLocator((2)))
        else:
            ax.yaxis.set_major_locator(MultipleLocator((4)))
        ax.xaxis.set_major_locator(MultipleLocator((2)))

        ax.yaxis.set_minor_locator(MultipleLocator((1)))
        ax.xaxis.set_minor_locator(MultipleLocator((1)))

        # ax.vlines(0, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], colors='black', linestyles='dashed', linewidth=3, zorder=1)
        ax.axvline(x=0, color='black', linestyle='dashed', linewidth=3, zorder=1, snap=False)

        ax.tick_params(axis='both', which='major', labelsize=16)


    #add legend to the empty ax6
    handles = [(Patch(facecolor='firebrick', edgecolor='none', alpha=0.3), Line2D([],[], color='firebrick', linewidth=2, marker='s', markersize=9, markeredgecolor='black', markeredgewidth=0.75)),
            (Patch(facecolor='gray', edgecolor='none', alpha=0.3), Line2D([],[], color='gray', linewidth=2, marker='*', markersize=13, markeredgecolor='black', markeredgewidth=0.75)),
            (Patch(facecolor='royalblue', edgecolor='none', alpha=0.3), Line2D([],[], color='royalblue', linewidth=2, marker='o', markersize=9, markeredgecolor='black', markeredgewidth=0.75))]

    ax6.legend(handles=handles, labels=["El Nino", "Neutral", "La Nina"], handler_map={tuple: HandlerTuple(ndivide=1)}, loc='center', fontsize=30, framealpha=1.0)

    irr_table = monthly_irr_table({
        "February": feb_result,
        "March": mar_result,
        "April": apr_result,
        "May": may_result,
        "June": jun_result
    })

    print(irr_table)

    #create the table -- keep table separate
    # cell_text = []
    # cell_text.append(["Month", "N_El Nino", "N_Neutral", "N_La Nina", "El Nino IRR", "Neutral IRR", "La Nina IRR", "Interaction p-value"])

    # columns = ["Month", "N_El Nino", "N_Neutral", "N_La Nina", "El Nino IRR", "Neutral IRR", "La Nina IRR", "Interaction p-value"]
    # rows = ['February']#, 'March', 'April', 'May', 'June']

    # table = ax6.table(cellText=cell_text, rowLabels=rows, colLabels=columns, rowLoc='center', cellLoc='center', loc='center')
    # table.set_fontsize(18)
    # # table.scale(0.75, 1.75) #row width, col height
    # ax6.set_title("Incidence Rate Ratios and Interaction P-value", fontsize=18, y=0.75)

    plt.subplots_adjust(wspace=0.1)
    # if gp == True:
    #     plt.suptitle("Monthly GP Cyclone and Dryline Count Regression", fontsize=30, y=0.95)
    #     plt.savefig('monthly_dryline_gp_cyclone_enso_regression.png', format='png', dpi=150, bbox_inches='tight') 
    # elif sgp == True:
    plt.suptitle("Monthly SGP Cyclone and Dryline Count Regression by ENSO", fontsize=30, y=0.95)
    plt.savefig(f'{directory_path}/monthly_dryline_sgp_cyclone_enso_regression_{YEAR_CUTOFF}.png', format='png', dpi=150, bbox_inches='tight') 

    ########### assess the importance of different predictors for the different months

    #Run model with different predictor groups removed, compare BIC (Bayesian Information Criterion) to compare 
    #how model performance compares when different predictors are included/removed

    test_formulas = {
        "full_model": "dryline_count ~ C(phase, Treatment(reference='Neutral')) * cyclone_anomaly",
        "No predictors": "dryline_count ~ 1",
        "No ENSO": "dryline_count ~ cyclone_anomaly",
        "No cyclone": "dryline_count ~ C(phase, Treatment(reference='Neutral'))",
        "No interaction": "dryline_count ~ C(phase, Treatment(reference='Neutral')) + cyclone_anomaly"
    }

    tests = []
    for tf in test_formulas:
        for month_name, monthly_data in [
            ("February", feb_df),
            ("April", apr_df),
            ("March", mar_df),
            ("May", may_df),
            ("June", jun_df)]:

            if month_name == "April": #use poisson model
                result = smf.poisson(
                    test_formulas[tf],
                    data=monthly_data,
                    offset=np.log(monthly_data["days"]),
                ).fit(disp=False, maxiter=200)
            else: #use negative binomial
                result = smf.negativebinomial(
                    test_formulas[tf],
                    data=monthly_data,
                    offset=np.log(monthly_data["days"]),
                ).fit(disp=False, maxiter=200)

            tests.append({
                "month": month_name,
                "test": tf,
                "result": result.bic
            })

    #convert dictionary into a dataframe
    tests_df = pd.DataFrame(tests)
    # print(tests_df)

    #create a simple table of BIC values relative to the full_model for each individual month
    bic_relative_to_full_model = tests_df.pivot(index="month", columns="test", values="result")
    bic_relative_to_full_model = bic_relative_to_full_model.subtract(bic_relative_to_full_model["full_model"], axis=0)
    print(bic_relative_to_full_model) #this will report [test] - [full_model]



# testing
# # feb_result = smf.negativebinomial(formula, data=feb_df, offset=np.log(feb_df["days"])).fit(disp=False,maxiter=200)
# mar_result = smf.negativebinomial(formula, data=mar_df, offset=np.log(mar_df["days"])).fit(disp=False,maxiter=200)
# # apr_result = smf.negativebinomial(formula, data=apr_df, offset=np.log(apr_df["days"])).fit(disp=False,maxiter=200)
# may_result = smf.negativebinomial(formula, data=may_df, offset=np.log(may_df["days"])).fit(disp=False,maxiter=200)
# jun_result = smf.negativebinomial(formula, data=jun_df, offset=np.log(jun_df["days"])).fit(disp=False,maxiter=200)

# # print("February")
# # print(feb_result.summary())
# print("March")
# print(mar_result.summary())
# # print("April")
# # print(apr_result.summary())
# print("May")
# print(may_result.summary())
# print("June")
# print(jun_result.summary())

# for month_name, monthly_data in [
#     ("February", feb_df),
#     ("April", apr_df),
#     ("March", mar_df),
#     ("May", may_df),
#     ("June", jun_df)]:
#     poisson_result = smf.poisson(
#         formula,
#         data=monthly_data,
#         offset=np.log(monthly_data["days"]),
#     ).fit(disp=False, maxiter=200)

#     expected_count = poisson_result.predict()
#     observed_count = poisson_result.model.endog
#     pearson_dispersion = np.sum(
#         (observed_count - expected_count) ** 2 / expected_count
#     ) / poisson_result.df_resid

#     print(month_name)
#     # print("Converged:", poisson_result.mle_retvals["converged"])
#     print("Pearson dispersion:", pearson_dispersion)

#     print(poisson_result.summary())

#################################################################################################
#################################################################################################
################################################################################################# multivariate logistic regression
################################################################################################# given cyclone characteristics, predict dryline probability
#################################################################################################


print('Time to run: ' + str(datetime.utcnow() - starttime))
