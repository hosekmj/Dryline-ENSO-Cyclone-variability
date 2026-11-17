#Timeseries analysis of drylines, ENSO, and cyclone frequency
#gpd_plot

import os
import pandas as pd
from datetime import datetime, timedelta
#import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from matplotlib.patches import Patch
#import matplotlib.colors as colors
import matplotlib
from scipy import stats
from scipy.signal import detrend
import matplotlib.patheffects as pe

#from shapely.geometry import LineString
#import shapely

#import random

import argparse

import xarray as xr

#import multiprocessing
#from itertools import repeat

#import imageio
#import glob


from netCDF4 import Dataset
starttime = datetime.utcnow()

#read in from input
parser = argparse.ArgumentParser()
parser.add_argument('terrain_mask',type=int,help='terrain_mask')
parser.add_argument('enso_type',type=str,help='enso_type')
parser.add_argument('domain_type',type=str,default=None,help='domain_type')
args = parser.parse_args()

terrain_mask = f"t={args.terrain_mask}"
enso_type = args.enso_type
domain_type = args.domain_type

if domain_type != "no_buffer" and domain_type != "only_centroid":
    domain_type = None

if domain_type is not None:
    directory_path = f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/{terrain_mask}/{enso_type}/{domain_type}"
else:
    directory_path = f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/{terrain_mask}/{enso_type}"
os.makedirs(directory_path, exist_ok=True)


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
    cyclones_df = pd.read_pickle(f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/all_cyclones_simplified_{terrain_mask}_{domain_type}.pkl")
else:
    cyclones_df = pd.read_pickle(f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/all_cyclones_simplified_{terrain_mask}.pkl")
#restrict to 00z
cyclones_df = cyclones_df[cyclones_df['hour'] == 0]

#read in drylines
all_drylines = pd.read_pickle('/raid/student/mhosek/era5_drylines/pkl/dryline_envi_1940_2020.pkl') #only 00z
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

################################################################ helpful functions
def add_trend(ax, x, y, color, label, confidence=0.95):
    #given an axis, x, and y, value, add the linear trend and return the r and p values
    #also plot the 95% confidence interval for the fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    y_fit = intercept + slope * x
    residuals = y - y_fit
    degrees_of_freedom = len(x) - 2 #standard dof for testing of a trend line
    residual_std = np.sqrt(np.sum(residuals**2) / degrees_of_freedom)

    x_mean = np.mean(x)
    sum_sqaured_deviation = np.sum((x-x_mean)**2)

    fit_std_err = residual_std * np.sqrt( 1/len(x) + (x-x_mean)**2 / sum_sqaured_deviation)

    alpha = 1 - confidence
    t_critical = stats.t.ppf(1-alpha/2, df=degrees_of_freedom) #student's t test
    lowerband = y_fit - t_critical * fit_std_err
    upperband = y_fit + t_critical * fit_std_err

    ax.plot(x, y_fit, color=color, label=label, linestyle='--', linewidth=2)
    ax.fill_between(x, lowerband, upperband, color=color, alpha=0.2)

    return r_value, p_value

def format_pvalue(p):
    if p < 0.001:
        return "<0.001"
    else:
        return f"{p:.3f}"

def add_text_box(ax, fullr, fullp, midr, midp, shortr, shortp, full=False, cyclone=False, alpha=0.05):
    #add text box of correlation and p-value; bold by significance

    if full == False:
        box_y = 1940.5
        box_x = 25
    else:
        if cyclone == False:
            box_y = 1940.5
            box_x = 73
        else:
            box_y = 1940.5
            box_x = 78

    stats_lines = [("1940-2020", fullr, fullp), ("1979-2020", midr, midp), ("1995-2020", shortr, shortp)]
    line_boxes = [TextArea(f"{label}: r={r:.2f}, p={format_pvalue(p)}", textprops={"fontsize": 22, "fontweight": "bold" if p < alpha else "normal"}) for label, r, p in stats_lines]
    stats_box = VPacker(children=line_boxes, align="left", pad=0, sep=5)
    text_box = AnchoredOffsetbox(loc='upper left', child=stats_box, bbox_to_anchor=(box_y, box_x), bbox_transform=ax.transData, frameon=True, borderpad=0.5)

    text_box.patch.set_facecolor('white')
    text_box.patch.set_edgecolor('black')
    text_box.patch.set_linewidth(1)
    text_box.patch.set_boxstyle("round,pad=0.5")
    text_box.patch.set_alpha(0.7)

    ax.add_artist(text_box)
    return

#################################################################
#################################################################
################################################################# Plot 1: 6-panel (3x2) timeseries of Feb--June, plus all, dryline days
#################################################################
#################################################################
#################################################################

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

years = np.arange(1940, 2021, 1)

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

fig, ((ax1,ax2), (ax3,ax4), (ax5,ax6)) = plt.subplots(nrows=3,ncols=2,figsize=(24,24))
axes = [ax1,ax2,ax3,ax4,ax5,ax6]

ax1.scatter(years, feb_dry, s=50, c="dimgray")
fullr, fullp = add_trend(ax1, years, feb_dry, color='black', label="1940-2020")
midr, midp = add_trend(ax1, years[years >= 1979], feb_dry[years >= 1979], color='royalblue', label="1979-2020")
shortr, shortp = add_trend(ax1, years[years >= 1995], feb_dry[years >= 1995], color='firebrick', label="1995-2020")
# ax1.text(1940.5, 20.5, "1940-2020: r={:.2f}, p={:.2g}\n1979-2020: r={:.2f}, p={:.2g}\n1995-2020: r={:.2f}, p={:.2g}".format(fullr, fullp, midr, midp, shortr, shortp), fontsize=22, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black'))
add_text_box(ax1, fullr, fullp, midr, midp, shortr, shortp)


ax2.scatter(years, march_dry, s=50, c="dimgray")
fullr, fullp = add_trend(ax2, years, march_dry, color='black', label="1940-2020")
midr, midp = add_trend(ax2, years[years >= 1979], march_dry[years >= 1979], color='royalblue', label="1979-2020")
shortr, shortp = add_trend(ax2, years[years >= 1995], march_dry[years >= 1995], color='firebrick', label="1995-2020")
add_text_box(ax2, fullr, fullp, midr, midp, shortr, shortp)


ax3.scatter(years, april_dry, s=50, c="dimgray")
fullr, fullp = add_trend(ax3, years, april_dry, color='black', label="1940-2020")
midr, midp = add_trend(ax3, years[years >= 1979], april_dry[years >= 1979], color='royalblue', label="1979-2020")
shortr, shortp = add_trend(ax3, years[years >= 1995], april_dry[years >= 1995], color='firebrick', label="1995-2020")
add_text_box(ax3, fullr, fullp, midr, midp, shortr, shortp)


ax4.scatter(years, may_dry, s=50, c="dimgray")
fullr, fullp = add_trend(ax4, years, may_dry, color='black', label="1940-2020")
midr, midp = add_trend(ax4, years[years >= 1979], may_dry[years >= 1979], color='royalblue', label="1979-2020")
shortr, shortp = add_trend(ax4, years[years >= 1995], may_dry[years >= 1995], color='firebrick', label="1995-2020")
add_text_box(ax4, fullr, fullp, midr, midp, shortr, shortp)


ax5.scatter(years, june_dry, s=50, c="dimgray")
fullr, fullp = add_trend(ax5, years, june_dry, color='black', label="1940-2020")
midr, midp = add_trend(ax5, years[years >= 1979], june_dry[years >= 1979], color='royalblue', label="1979-2020")
shortr, shortp = add_trend(ax5, years[years >= 1995], june_dry[years >= 1995], color='firebrick', label="1995-2020")
add_text_box(ax5, fullr, fullp, midr, midp, shortr, shortp)


ax6.scatter(years, allspring, s=50, c="dimgray")
fullr, fullp = add_trend(ax6, years, allspring, color='black', label="1940-2020")
midr, midp = add_trend(ax6, years[years >= 1979], allspring[years >= 1979], color='royalblue', label="1979-2020")
shortr, shortp = add_trend(ax6, years[years >= 1995], allspring[years >= 1995], color='firebrick', label="1995-2020")
add_text_box(ax6, fullr, fullp, midr, midp, shortr, shortp, full=True)



######################### plot parameters
panellabel = ["a)", "b)", "c)", "d)", "e)", "f)"]
for i, ax in enumerate(axes):
    ax.set_xlim(1939.5, 2020.5)
    ax.xaxis.set_major_locator(MultipleLocator((10)))
    ax.xaxis.set_minor_locator(MultipleLocator((1)))

    ax.tick_params(axis='both', which='major', labelsize=16)

    ax.grid(visible=True, axis='y')

    if ax != ax6:
        ax.set_ylim(-0.5, 26)
        ax.yaxis.set_major_locator(MultipleLocator((5)))
        ax.yaxis.set_minor_locator(MultipleLocator((1)))

        ax.text(1940.25, 27, panellabel[i], fontsize=22, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black'))
    else:
        ax.set_ylim(15, 75)
        ax.yaxis.set_major_locator(MultipleLocator((10)))
        ax.yaxis.set_minor_locator(MultipleLocator((5)))
        ax.text(1940.25, 77, panellabel[i], fontsize=22, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black'))

    ax.vlines(1979, *ax.get_ylim(), colors='royalblue', linestyles='dotted', linewidth=2)
    ax.vlines(1995, *ax.get_ylim(), colors='firebrick', linestyles='dotted', linewidth=2)


ax1.set_title("February", fontsize=32)
ax1.set_ylabel("Dryline Days", fontsize=24)
ax1.legend(fontsize=20, edgecolor='black')

ax2.set_title("March", fontsize=32)


ax3.set_title("April", fontsize=32)
ax3.set_ylabel("Dryline Days", fontsize=24)


ax4.set_title("May", fontsize=32)


ax5.set_title("June", fontsize=32)
ax5.set_xlabel("Year", fontsize=24)
ax5.set_ylabel("Dryline Days", fontsize=24)


ax6.set_title("Feb-June", fontsize=32)
ax6.set_xlabel("Year", fontsize=24)


plt.subplots_adjust(wspace=0.065, hspace=0.15)
plt.suptitle("Dryline Day Timeseries", y=0.93, fontsize=48)

if domain_type is not None:
    plt.savefig(f'/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/{terrain_mask}/{enso_type}/{domain_type}/dryline_timeseries.png',dpi=100,bbox_inches='tight')
else:
    plt.savefig(f'/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/{terrain_mask}/{enso_type}/dryline_timeseries.png',dpi=100,bbox_inches='tight')


#################################################################
#################################################################
################################################################# Plot 2: 6-panel (3x2) timeseries of Feb--June, plus all, cyclone days
#################################################################
#################################################################
#################################################################


monthly_lee_cyclone_days = (cyclones_df.groupby(['year', 'month'])['sgp_cyclone_day'].sum().unstack('month', fill_value=0).reindex(index=years, columns=range(2,7), fill_value=0))
figname = "sgp"


feb_cyclones = monthly_lee_cyclone_days[2].to_numpy()
march_cyclones = monthly_lee_cyclone_days[3].to_numpy()
april_cyclones = monthly_lee_cyclone_days[4].to_numpy()
may_cyclones = monthly_lee_cyclone_days[5].to_numpy()
june_cyclones = monthly_lee_cyclone_days[6].to_numpy()

allspring_cyclones = monthly_lee_cyclone_days.sum(axis=1).to_numpy()


fig, ((ax1,ax2), (ax3,ax4), (ax5,ax6)) = plt.subplots(nrows=3,ncols=2,figsize=(24,24))
axes = [ax1,ax2,ax3,ax4,ax5,ax6]

ax1.scatter(years, feb_cyclones, s=50, c="dimgray")
fullr, fullp = add_trend(ax1, years, feb_cyclones, color='black', label="1940-2020")
midr, midp = add_trend(ax1, years[years >= 1979], feb_cyclones[years >= 1979], color='royalblue', label="1979-2020")
shortr, shortp = add_trend(ax1, years[years >= 1995], feb_cyclones[years >= 1995], color='firebrick', label="1995-2020")
# ax1.text(1940.5, 20.5, "1940-2020: r={:.2f}, p={:.2g}\n1979-2020: r={:.2f}, p={:.2g}\n1995-2020: r={:.2f}, p={:.2g}".format(fullr, fullp, midr, midp, shortr, shortp), fontsize=22, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black'))
add_text_box(ax1, fullr, fullp, midr, midp, shortr, shortp)


ax2.scatter(years, march_cyclones, s=50, c="dimgray")
fullr, fullp = add_trend(ax2, years, march_cyclones, color='black', label="1940-2020")
midr, midp = add_trend(ax2, years[years >= 1979], march_cyclones[years >= 1979], color='royalblue', label="1979-2020")
shortr, shortp = add_trend(ax2, years[years >= 1995], march_cyclones[years >= 1995], color='firebrick', label="1995-2020")
add_text_box(ax2, fullr, fullp, midr, midp, shortr, shortp)


ax3.scatter(years, april_cyclones, s=50, c="dimgray")
fullr, fullp = add_trend(ax3, years, april_cyclones, color='black', label="1940-2020")
midr, midp = add_trend(ax3, years[years >= 1979], april_cyclones[years >= 1979], color='royalblue', label="1979-2020")
shortr, shortp = add_trend(ax3, years[years >= 1995], april_cyclones[years >= 1995], color='firebrick', label="1995-2020")
add_text_box(ax3, fullr, fullp, midr, midp, shortr, shortp)


ax4.scatter(years, may_cyclones, s=50, c="dimgray")
fullr, fullp = add_trend(ax4, years, may_cyclones, color='black', label="1940-2020")
midr, midp = add_trend(ax4, years[years >= 1979], may_cyclones[years >= 1979], color='royalblue', label="1979-2020")
shortr, shortp = add_trend(ax4, years[years >= 1995], may_cyclones[years >= 1995], color='firebrick', label="1995-2020")
add_text_box(ax4, fullr, fullp, midr, midp, shortr, shortp)


ax5.scatter(years, june_cyclones, s=50, c="dimgray")
fullr, fullp = add_trend(ax5, years, june_cyclones, color='black', label="1940-2020")
midr, midp = add_trend(ax5, years[years >= 1979], june_cyclones[years >= 1979], color='royalblue', label="1979-2020")
shortr, shortp = add_trend(ax5, years[years >= 1995], june_cyclones[years >= 1995], color='firebrick', label="1995-2020")
add_text_box(ax5, fullr, fullp, midr, midp, shortr, shortp)


ax6.scatter(years, allspring_cyclones, s=50, c="dimgray")
fullr, fullp = add_trend(ax6, years, allspring_cyclones, color='black', label="1940-2020")
midr, midp = add_trend(ax6, years[years >= 1979], allspring_cyclones[years >= 1979], color='royalblue', label="1979-2020")
shortr, shortp = add_trend(ax6, years[years >= 1995], allspring_cyclones[years >= 1995], color='firebrick', label="1995-2020")
add_text_box(ax6, fullr, fullp, midr, midp, shortr, shortp, full=True, cyclone=True)


######################### plot parameters
panellabel = ["a)", "b)", "c)", "d)", "e)", "f)"]
for i, ax in enumerate(axes):
    ax.set_xlim(1939.5, 2020.5)
    ax.xaxis.set_major_locator(MultipleLocator((10)))
    ax.xaxis.set_minor_locator(MultipleLocator((1)))

    ax.tick_params(axis='both', which='major', labelsize=16)

    ax.grid(visible=True, axis='y')

    if ax != ax6:
        ax.set_ylim(-0.5, 26)
        ax.yaxis.set_major_locator(MultipleLocator((5)))
        ax.yaxis.set_minor_locator(MultipleLocator((1)))

        ax.text(1940.25, 27, panellabel[i], fontsize=22, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black'))
    else:
        ax.set_ylim(25, 80)
        ax.yaxis.set_major_locator(MultipleLocator((10)))
        ax.yaxis.set_minor_locator(MultipleLocator((5)))
        ax.text(1940.25, 82, panellabel[i], fontsize=22, fontweight='bold', bbox=dict(facecolor='white', edgecolor='black'))

    ax.vlines(1979, *ax.get_ylim(), colors='royalblue', linestyles='dotted', linewidth=2)
    ax.vlines(1995, *ax.get_ylim(), colors='firebrick', linestyles='dotted', linewidth=2)


ax1.set_title("February", fontsize=32)
ax1.set_ylabel("Cyclone Days", fontsize=24)
ax1.legend(fontsize=20, edgecolor='black')

ax2.set_title("March", fontsize=32)


ax3.set_title("April", fontsize=32)
ax3.set_ylabel("Cyclone Days", fontsize=24)


ax4.set_title("May", fontsize=32)


ax5.set_title("June", fontsize=32)
ax5.set_xlabel("Year", fontsize=24)
ax5.set_ylabel("Cyclone Days", fontsize=24)


ax6.set_title("Feb-June", fontsize=32)
ax6.set_xlabel("Year", fontsize=24)


plt.subplots_adjust(wspace=0.065, hspace=0.15)
plt.suptitle("Southern Great Plains Cyclone Day Timeseries", y=0.93, fontsize=48)

if domain_type is not None:
    plt.savefig(f'/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/{terrain_mask}/{enso_type}/{domain_type}/sgp_cyclone_timeseries_{figname}_{domain_type}.png',dpi=100,bbox_inches='tight')
else:
    plt.savefig(f'/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/{terrain_mask}/{enso_type}/sgp_cyclone_timeseries_{figname}.png',dpi=100,bbox_inches='tight')


#################################################################
#################################################################
################################################################# Plot 3: 15-panel: timeseries of ENSO phase
#################################################################
#################################################################
#################################################################

fig, ((ax1,ax2,ax3), (ax4,ax5,ax6), (ax7,ax8,ax9), (ax10,ax11,ax12), (ax13,ax14,ax15)) = plt.subplots(nrows=5,ncols=3,figsize=(24,24))
axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15]

forlooping = [feb_op_enso, feb_3mo_enso, feb_1mo_enso, 
              march_op_enso, march_3mo_enso, march_1mo_enso,
              april_op_enso, april_3mo_enso, april_1mo_enso, 
              may_op_enso, may_3mo_enso, may_1mo_enso, 
              june_op_enso, june_3mo_enso, june_1mo_enso]

for i,ax in enumerate(axes):
    ax.set_xlim(1939.5, 2020.5)
    ax.xaxis.set_major_locator(MultipleLocator((10)))
    ax.xaxis.set_minor_locator(MultipleLocator((1)))

    ax.set_ylim(-0.5, 0.5)
    ymin, ymax = ax.get_ylim()

    enso = np.array(forlooping[i])

    nino_years = years[enso == 1]
    nina_years = years[enso == -1]

    ax.bar(nino_years, ymax - ymin, bottom=ymin, width=1, align='center', facecolor='red', alpha=0.3)
    ax.bar(nina_years, ymax - ymin, bottom=ymin, width=1, align='center', facecolor='blue', alpha=0.3)

    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.tick_params(axis='y', which='major', labelsize=0)


ax1.set_title("Operational\nFebruary", fontsize=24)
ax2.set_title("3-month ONI\nFebruary", fontsize=24)
ax3.set_title("1-month ONI\nFebruary", fontsize=24)

ax4.set_title("March", fontsize=24)
ax5.set_title("March", fontsize=24)
ax6.set_title("March", fontsize=24)

ax7.set_title("April", fontsize=24)
ax8.set_title("April", fontsize=24)
ax9.set_title("April", fontsize=24)

ax10.set_title("May", fontsize=24)
ax11.set_title("May", fontsize=24)
ax12.set_title("May", fontsize=24)

ax13.set_title("June", fontsize=24)
ax14.set_title("June", fontsize=24)
ax15.set_title("June", fontsize=24)

#add legend
handles = [Patch(facecolor='red', alpha=0.3, label='El Nino'),
           Patch(facecolor='blue', alpha=0.3, label='La Nina')]
ax1.legend(handles=handles, loc='upper right', fontsize=20)

plt.subplots_adjust(wspace=0.065, hspace=0.2)
plt.suptitle("ENSO Definition Timeseries", y=0.94, fontsize=48)

plt.savefig(f'/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/{terrain_mask}/{enso_type}/enso_timeseries.png',dpi=100,bbox_inches='tight')


print('Time to run: ' + str(datetime.utcnow() - starttime))