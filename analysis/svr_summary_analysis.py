#analysis as it related to SGP severe
#gpd_plot

#plots/tables to make (by month and phase unless otherwise stated):
    # % of dryline days that are associated with at least 1 SGP SVR (total, just cyclone-associated, just quiescent?)
    # % of SGP cyclone days that are associated with at least 1 SGP SVR
    # % of all SVR days that have at least 1 dryline-associated SVR
    # % of total SGP SVR area which is dryline-associated
        #divide SGP into 4 quadrants
        #total dryline-associated area, % from cyclone-associated, % from cyclone-adjacent, % from quiescent
    # model that predicts number of dryline-associated SVR days (binomial?)
        # predictors: ENSO phase, SGP cyclone count, interaction
    #GAM to predict if at least 1 SVR report in SGP [MAYBE DON'T NEED THIS?]
        # predictors: ENSO phase, cyclone location, cyclone MSLP, cyclone area, cyclone wind, is lee cyclone, is dryline present


from cmath import phase

from cmath import phase

import os
from numpy.linalg.linalg import norm
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

from statsmodels.stats.proportion import proportion_confint

from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerPatch, HandlerErrorbar
from matplotlib.container import ErrorbarContainer

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

#read in from input
parser = argparse.ArgumentParser()
parser.add_argument('terrain_mask',type=int,help='terrain_mask')
parser.add_argument('enso_type',type=str,help='enso_type')
parser.add_argument('domain_type',type=str,help='domain_type', nargs='?', default=None)
args = parser.parse_args()

terrain_mask = f"t={args.terrain_mask}"
ENSO_TYPE = args.enso_type
domain_type = args.domain_type

if domain_type == "None":
    domain_type = None


directory_path = f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/{terrain_mask}/{ENSO_TYPE}"
if domain_type is not None:
    directory_path = f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/{terrain_mask}/{ENSO_TYPE}/{domain_type}"
os.makedirs(directory_path, exist_ok=True)

from netCDF4 import Dataset
starttime = datetime.utcnow()


#read in SVR pickle with all the info we need
svr_summary = pd.read_pickle("/home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/dryline_svr_summary.pkl")

#read in cyclones
if domain_type is not None:
    cyclones_df = pd.read_pickle(f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/all_cyclones_simplified_{terrain_mask}_{domain_type}.pkl")
else:
    cyclones_df = pd.read_pickle(f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/all_cyclones_simplified_{terrain_mask}.pkl")
#restrict to 00z, 1995+
cyclones_df = cyclones_df[cyclones_df['hour'] == 0]
cyclones_df = cyclones_df[cyclones_df['year'] >= 1995]

#attach cyclone information onto the svr summary dataframe
year = cyclones_df['year'].tolist()
month = cyclones_df['month'].tolist()
sgp_cyclone_day = cyclones_df['sgp_cyclone_day'].tolist()

svr_summary['year'] = year
svr_summary['month'] = month
svr_summary['sgp_cyclone_day'] = sgp_cyclone_day

#bools for single day, if a dryline is one of three types
cyclone_associated_dryline = []
cyclone_adjacent_dryline = []
quiescent_dryline = []

for i in range(len(cyclones_df)):
    row = cyclones_df.iloc[i]
    cyclone_associated_dryline.append(any(row['dryline_within_cyclone']))
    cyclone_adjacent_dryline.append(any(row['dryline_associated_cyclone']) and not any(row['dryline_within_cyclone']))
    quiescent_dryline.append(row['dryline'] and not any(row['dryline_within_cyclone']) and not any(row['dryline_associated_cyclone']))
    # print(cyclone_associated_dryline[-1], cyclone_adjacent_dryline[-1], quiescent_dryline[-1])

svr_summary['cyclone_associated_dryline'] = cyclone_associated_dryline
svr_summary['cyclone_adjacent_dryline'] = cyclone_adjacent_dryline
svr_summary['quiescent_dryline'] = quiescent_dryline
svr_summary['any_dryline'] = svr_summary['cyclone_associated_dryline'] | svr_summary['cyclone_adjacent_dryline'] | svr_summary['quiescent_dryline']

total_years_of_sample = 2020 - 1995 + 1

def odds_ratio(df, condition_a, condition_b, phase=None, toprint=True):
    #given a dataframe of drylines, cyclones, and a phase (optional), calculate the odds ratio between any 2 conditions
    #phase is -1, 0, or 1 (ints)

    if phase is not None:
        df = df[df[ENSO_TYPE] == phase]

    # a: 
    a = len(df[(df[condition_a]==True) & (df[condition_b]==True)])

    # b:
    b = len(df[(df[condition_a]==True) & (df[condition_b]==False)])

    # c:
    c = len(df[(df[condition_b]==True) & (df[condition_a]==False)])

    #d : 
    d = len(df[(df[condition_b]==False) & (df[condition_a]==False)])


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
        print('TEST: ', condition_a, 'vs ', condition_b, " during phase ", phase)
        print('Odds ratio: ', np.round(or_value,2))
        print('95% CI: ', np.round(ci_lower,2), np.round(ci_upper,2))
        print('p-value: ', np.round(pvalue,4))
        print('Expected frequencies: ', np.round(expected[0,0],2), np.round(expected[0,1],2), np.round(expected[1,0],2), np.round(expected[1,1],2))
        #use expected to determine if chi-square test is appropriate (all values must > 5)
        print('\n')

        return
    else:
        return or_value, ci_lower, ci_upper

def proportion(df, condition, event, phase=None, toprint=False):
    #given a dataframe of drylines, cyclones, and a phase (optional), calculate the proportion of days ("condition") which have the specified "event"
    #phase is -1, 0, or 1 (ints)

    if phase is not None:
        df = df[df[ENSO_TYPE] == phase]

    df = df[df[condition]==True]

    success = df[df[event]==True]

    n_success = len(success)
    n = len(df)
    proportion = n_success / n if n > 0 else 0

    # WALD VERSION OF CALCULATION
    # se = np.sqrt(proportion * (1 - proportion) / n) if n > 0 else 0
    # me = se * 1.96 #for 95% CI
    # ci_lower = proportion - me if n > 0 else 0
    # ci_upper = proportion + me if n > 0 else 0

    if n > 0:
        ci_lower, ci_upper = proportion_confint(n_success, n, alpha=0.05, method='wilson')
        # guard against floating-point error leaving the bounds on the wrong side of the estimate
        ci_lower = min(ci_lower, proportion)
        ci_upper = max(ci_upper, proportion)
    else:
        ci_lower = ci_upper = 0

    if toprint:
        print('Proportion of days with', event, 'given', condition, 'during phase', phase, ':', proportion, '95% CI:', ci_lower, "-", ci_upper)
        return
    else:
        return proportion, ci_lower, ci_upper, n_success, n



############################################################
############################################################
############################################################ Table: avg days with SVR, avg days with dryline, proportion of dryline days with any SGP severe 
############################################################ #split by month/ENSO. Do for all-reports, and for each hazard individually

#Dryline vs SGP severe day
phases = [0, -1, 1]
for ph in phases:

    # spring_prop, spring_ci_lower, spring_ci_upper, _, _ = proportion(svr_summary, 'any_dryline', 'all_report_bool', phase=ph)
    # phase_df = svr_summary[svr_summary['enso_op'] == ph] #THIS DOESN'T MAKE SENSE BECAUSE I CAN'T DEFINE SPRING BY ENSO PHASE
    # spring_years_of_sample = len(phase_df['year'].unique())
    # spring_dryday = len(phase_df[phase_df['any_dryline']==True]) / spring_years_of_sample
    # spring_svrday = len(phase_df[phase_df['all_report_bool']==True]) / spring_years_of_sample

    feb_prop, feb_ci_lower, feb_ci_upper, _, _ = proportion(svr_summary[svr_summary['month'] == 2], 'any_dryline', 'all_report_bool', phase=ph)
    phase_df = svr_summary[(svr_summary[ENSO_TYPE] == ph) & (svr_summary['month'] == 2)]
    feb_years_of_sample = len(phase_df['year'].unique())
    feb_dryday = len(phase_df[(phase_df['any_dryline']==True)]) / feb_years_of_sample
    feb_svrday = len(phase_df[(phase_df['all_report_bool']==True)]) / feb_years_of_sample

    march_prop, march_ci_lower, march_ci_upper, _, _ = proportion(svr_summary[svr_summary['month'] == 3], 'any_dryline', 'all_report_bool', phase=ph)
    phase_df = svr_summary[(svr_summary[ENSO_TYPE] == ph) & (svr_summary['month'] == 3)]
    march_years_of_sample = len(phase_df['year'].unique())
    march_dryday = len(phase_df[(phase_df['any_dryline']==True)]) / march_years_of_sample
    march_svrday = len(phase_df[(phase_df['all_report_bool']==True)]) / march_years_of_sample

    april_prop, april_ci_lower, april_ci_upper, _, _ = proportion(svr_summary[svr_summary['month'] == 4], 'any_dryline', 'all_report_bool', phase=ph)
    phase_df = svr_summary[(svr_summary[ENSO_TYPE] == ph) & (svr_summary['month'] == 4)]
    april_years_of_sample = len(phase_df['year'].unique())
    april_dryday = len(phase_df[(phase_df['any_dryline']==True)]) / april_years_of_sample
    april_svrday = len(phase_df[(phase_df['all_report_bool']==True)]) / april_years_of_sample

    may_prop, may_ci_lower, may_ci_upper, _, _ = proportion(svr_summary[svr_summary['month'] == 5], 'any_dryline', 'all_report_bool', phase=ph)
    phase_df = svr_summary[(svr_summary[ENSO_TYPE] == ph) & (svr_summary['month'] == 5)]
    may_years_of_sample = len(phase_df['year'].unique())
    may_dryday = len(phase_df[(phase_df['any_dryline']==True)]) / may_years_of_sample
    may_svrday = len(phase_df[(phase_df['all_report_bool']==True)]) / may_years_of_sample

    june_prop, june_ci_lower, june_ci_upper, _, _ = proportion(svr_summary[svr_summary['month'] == 6], 'any_dryline', 'all_report_bool', phase=ph)
    phase_df = svr_summary[(svr_summary[ENSO_TYPE] == ph) & (svr_summary['month'] == 6)]
    june_years_of_sample = len(phase_df['year'].unique())
    june_dryday = len(phase_df[(phase_df['any_dryline']==True)]) / june_years_of_sample
    june_svrday = len(phase_df[(phase_df['all_report_bool']==True)]) / june_years_of_sample

    rows = [
            # ("Feb-June", spring_years_of_sample,spring_dryday, spring_svrday,
            # spring_prop, spring_ci_lower, spring_ci_upper),
            ("February", feb_years_of_sample, feb_dryday, feb_svrday,
            feb_prop, feb_ci_lower, feb_ci_upper),
            ("March", march_years_of_sample, march_dryday, march_svrday,
            march_prop, march_ci_lower, march_ci_upper),
            ("April", april_years_of_sample, april_dryday, april_svrday,
            april_prop, april_ci_lower, april_ci_upper),
            ("May", may_years_of_sample, may_dryday, may_svrday,
            may_prop, may_ci_lower, may_ci_upper),
            ("June", june_years_of_sample, june_dryday, june_svrday,
            june_prop, june_ci_lower, june_ci_upper),
        ]

    table = pd.DataFrame(
        [
            {
                "Month/group": label,
                "Years": years,
                "Avg dryline days": dry,
                "Avg SGP severe days": severe,
                "SGP severe | dryline": f"{prop:.1%}",
                "95% CI": f"({lower:.1%}, {upper:.1%})",
            }
            for label, years, dry, severe, prop, lower, upper in rows
        ]
    )

    print(f"\nENSO phase: {ph} All SVR")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# by individual hazard
for ph in phases:
    feb_prop, feb_ci_lower, feb_ci_upper, _, _ = proportion(svr_summary[svr_summary['month'] == 2], 'any_dryline', 'all_tor_report_bool', phase=ph)
    phase_df = svr_summary[(svr_summary[ENSO_TYPE] == ph) & (svr_summary['month'] == 2)]
    feb_years_of_sample = len(phase_df['year'].unique())
    feb_dryday = len(phase_df[(phase_df['any_dryline']==True)]) / feb_years_of_sample
    feb_svrday = len(phase_df[(phase_df['all_tor_report_bool']==True)]) / feb_years_of_sample

    march_prop, march_ci_lower, march_ci_upper, _, _ = proportion(svr_summary[svr_summary['month'] == 3], 'any_dryline', 'all_tor_report_bool', phase=ph)
    phase_df = svr_summary[(svr_summary[ENSO_TYPE] == ph) & (svr_summary['month'] == 3)]
    march_years_of_sample = len(phase_df['year'].unique())
    march_dryday = len(phase_df[(phase_df['any_dryline']==True)]) / march_years_of_sample
    march_svrday = len(phase_df[(phase_df['all_tor_report_bool']==True)]) / march_years_of_sample

    april_prop, april_ci_lower, april_ci_upper, _, _ = proportion(svr_summary[svr_summary['month'] == 4], 'any_dryline', 'all_tor_report_bool', phase=ph)
    phase_df = svr_summary[(svr_summary[ENSO_TYPE] == ph) & (svr_summary['month'] == 4)]
    april_years_of_sample = len(phase_df['year'].unique())
    april_dryday = len(phase_df[(phase_df['any_dryline']==True)]) / april_years_of_sample
    april_svrday = len(phase_df[(phase_df['all_tor_report_bool']==True)]) / april_years_of_sample

    may_prop, may_ci_lower, may_ci_upper, _, _ = proportion(svr_summary[svr_summary['month'] == 5], 'any_dryline', 'all_tor_report_bool', phase=ph)
    phase_df = svr_summary[(svr_summary[ENSO_TYPE] == ph) & (svr_summary['month'] == 5)]
    may_years_of_sample = len(phase_df['year'].unique())
    may_dryday = len(phase_df[(phase_df['any_dryline']==True)]) / may_years_of_sample
    may_svrday = len(phase_df[(phase_df['all_tor_report_bool']==True)]) / may_years_of_sample

    june_prop, june_ci_lower, june_ci_upper, _, _ = proportion(svr_summary[svr_summary['month'] == 6], 'any_dryline', 'all_tor_report_bool', phase=ph)
    phase_df = svr_summary[(svr_summary[ENSO_TYPE] == ph) & (svr_summary['month'] == 6)]
    june_years_of_sample = len(phase_df['year'].unique())
    june_dryday = len(phase_df[(phase_df['any_dryline']==True)]) / june_years_of_sample
    june_svrday = len(phase_df[(phase_df['all_tor_report_bool']==True)]) / june_years_of_sample

    rows = [
            # ("Feb-June", spring_years_of_sample,spring_dryday, spring_svrday,
            # spring_prop, spring_ci_lower, spring_ci_upper),
            ("February", feb_years_of_sample, feb_dryday, feb_svrday,
            feb_prop, feb_ci_lower, feb_ci_upper),
            ("March", march_years_of_sample, march_dryday, march_svrday,
            march_prop, march_ci_lower, march_ci_upper),
            ("April", april_years_of_sample, april_dryday, april_svrday,
            april_prop, april_ci_lower, april_ci_upper),
            ("May", may_years_of_sample, may_dryday, may_svrday,
            may_prop, may_ci_lower, may_ci_upper),
            ("June", june_years_of_sample, june_dryday, june_svrday,
            june_prop, june_ci_lower, june_ci_upper),
        ]

    table = pd.DataFrame(
        [
            {
                "Month/group": label,
                "Years": years,
                "Avg dryline days": dry,
                "Avg SGP tornado days": severe,
                "SGP tornado | dryline": f"{prop:.1%}",
                "95% CI": f"({lower:.1%}, {upper:.1%})",
            }
            for label, years, dry, severe, prop, lower, upper in rows
        ]
    )

    print(f"\nENSO phase: {ph} Tornado")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


for ph in phases:
    feb_prop, feb_ci_lower, feb_ci_upper, _, _ = proportion(svr_summary[svr_summary['month'] == 2], 'any_dryline', 'all_hail_report_bool', phase=ph)
    phase_df = svr_summary[(svr_summary[ENSO_TYPE] == ph) & (svr_summary['month'] == 2)]
    feb_years_of_sample = len(phase_df['year'].unique())
    feb_dryday = len(phase_df[(phase_df['any_dryline']==True)]) / feb_years_of_sample
    feb_svrday = len(phase_df[(phase_df['all_hail_report_bool']==True)]) / feb_years_of_sample

    march_prop, march_ci_lower, march_ci_upper, _, _ = proportion(svr_summary[svr_summary['month'] == 3], 'any_dryline', 'all_hail_report_bool', phase=ph)
    phase_df = svr_summary[(svr_summary[ENSO_TYPE] == ph) & (svr_summary['month'] == 3)]
    march_years_of_sample = len(phase_df['year'].unique())
    march_dryday = len(phase_df[(phase_df['any_dryline']==True)]) / march_years_of_sample
    march_svrday = len(phase_df[(phase_df['all_hail_report_bool']==True)]) / march_years_of_sample

    april_prop, april_ci_lower, april_ci_upper, _, _ = proportion(svr_summary[svr_summary['month'] == 4], 'any_dryline', 'all_hail_report_bool', phase=ph)
    phase_df = svr_summary[(svr_summary[ENSO_TYPE] == ph) & (svr_summary['month'] == 4)]
    april_years_of_sample = len(phase_df['year'].unique())
    april_dryday = len(phase_df[(phase_df['any_dryline']==True)]) / april_years_of_sample
    april_svrday = len(phase_df[(phase_df['all_hail_report_bool']==True)]) / april_years_of_sample

    may_prop, may_ci_lower, may_ci_upper, _, _ = proportion(svr_summary[svr_summary['month'] == 5], 'any_dryline', 'all_hail_report_bool', phase=ph)
    phase_df = svr_summary[(svr_summary[ENSO_TYPE] == ph) & (svr_summary['month'] == 5)]
    may_years_of_sample = len(phase_df['year'].unique())
    may_dryday = len(phase_df[(phase_df['any_dryline']==True)]) / may_years_of_sample
    may_svrday = len(phase_df[(phase_df['all_hail_report_bool']==True)]) / may_years_of_sample

    june_prop, june_ci_lower, june_ci_upper, _, _ = proportion(svr_summary[svr_summary['month'] == 6], 'any_dryline', 'all_hail_report_bool', phase=ph)
    phase_df = svr_summary[(svr_summary[ENSO_TYPE] == ph) & (svr_summary['month'] == 6)]
    june_years_of_sample = len(phase_df['year'].unique())
    june_dryday = len(phase_df[(phase_df['any_dryline']==True)]) / june_years_of_sample
    june_svrday = len(phase_df[(phase_df['all_hail_report_bool']==True)]) / june_years_of_sample

    rows = [
            # ("Feb-June", spring_years_of_sample,spring_dryday, spring_svrday,
            # spring_prop, spring_ci_lower, spring_ci_upper),
            ("February", feb_years_of_sample, feb_dryday, feb_svrday,
            feb_prop, feb_ci_lower, feb_ci_upper),
            ("March", march_years_of_sample, march_dryday, march_svrday,
            march_prop, march_ci_lower, march_ci_upper),
            ("April", april_years_of_sample, april_dryday, april_svrday,
            april_prop, april_ci_lower, april_ci_upper),
            ("May", may_years_of_sample, may_dryday, may_svrday,
            may_prop, may_ci_lower, may_ci_upper),
            ("June", june_years_of_sample, june_dryday, june_svrday,
            june_prop, june_ci_lower, june_ci_upper),
        ]

    table = pd.DataFrame(
        [
            {
                "Month/group": label,
                "Years": years,
                "Avg dryline days": dry,
                "Avg SGP hail days": severe,
                "SGP hail | dryline": f"{prop:.1%}",
                "95% CI": f"({lower:.1%}, {upper:.1%})",
            }
            for label, years, dry, severe, prop, lower, upper in rows
        ]
    )

    print(f"\nENSO phase: {ph} Hail")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


for ph in phases:
    feb_prop, feb_ci_lower, feb_ci_upper, _, _ = proportion(svr_summary[svr_summary['month'] == 2], 'any_dryline', 'all_wind_report_bool', phase=ph)
    phase_df = svr_summary[(svr_summary[ENSO_TYPE] == ph) & (svr_summary['month'] == 2)]
    feb_years_of_sample = len(phase_df['year'].unique())
    feb_dryday = len(phase_df[(phase_df['any_dryline']==True)]) / feb_years_of_sample
    feb_svrday = len(phase_df[(phase_df['all_wind_report_bool']==True)]) / feb_years_of_sample

    march_prop, march_ci_lower, march_ci_upper, _, _ = proportion(svr_summary[svr_summary['month'] == 3], 'any_dryline', 'all_wind_report_bool', phase=ph)
    phase_df = svr_summary[(svr_summary[ENSO_TYPE] == ph) & (svr_summary['month'] == 3)]
    march_years_of_sample = len(phase_df['year'].unique())
    march_dryday = len(phase_df[(phase_df['any_dryline']==True)]) / march_years_of_sample
    march_svrday = len(phase_df[(phase_df['all_wind_report_bool']==True)]) / march_years_of_sample

    april_prop, april_ci_lower, april_ci_upper, _, _ = proportion(svr_summary[svr_summary['month'] == 4], 'any_dryline', 'all_wind_report_bool', phase=ph)
    phase_df = svr_summary[(svr_summary[ENSO_TYPE] == ph) & (svr_summary['month'] == 4)]
    april_years_of_sample = len(phase_df['year'].unique())
    april_dryday = len(phase_df[(phase_df['any_dryline']==True)]) / april_years_of_sample
    april_svrday = len(phase_df[(phase_df['all_wind_report_bool']==True)]) / april_years_of_sample

    may_prop, may_ci_lower, may_ci_upper, _, _ = proportion(svr_summary[svr_summary['month'] == 5], 'any_dryline', 'all_wind_report_bool', phase=ph)
    phase_df = svr_summary[(svr_summary[ENSO_TYPE] == ph) & (svr_summary['month'] == 5)]
    may_years_of_sample = len(phase_df['year'].unique())
    may_dryday = len(phase_df[(phase_df['any_dryline']==True)]) / may_years_of_sample
    may_svrday = len(phase_df[(phase_df['all_wind_report_bool']==True)]) / may_years_of_sample

    june_prop, june_ci_lower, june_ci_upper, _, _ = proportion(svr_summary[svr_summary['month'] == 6], 'any_dryline', 'all_wind_report_bool', phase=ph)
    phase_df = svr_summary[(svr_summary[ENSO_TYPE] == ph) & (svr_summary['month'] == 6)]
    june_years_of_sample = len(phase_df['year'].unique())
    june_dryday = len(phase_df[(phase_df['any_dryline']==True)]) / june_years_of_sample
    june_svrday = len(phase_df[(phase_df['all_wind_report_bool']==True)]) / june_years_of_sample

    rows = [
            # ("Feb-June", spring_years_of_sample,spring_dryday, spring_svrday,
            # spring_prop, spring_ci_lower, spring_ci_upper),
            ("February", feb_years_of_sample, feb_dryday, feb_svrday,
            feb_prop, feb_ci_lower, feb_ci_upper),
            ("March", march_years_of_sample, march_dryday, march_svrday,
            march_prop, march_ci_lower, march_ci_upper),
            ("April", april_years_of_sample, april_dryday, april_svrday,
            april_prop, april_ci_lower, april_ci_upper),
            ("May", may_years_of_sample, may_dryday, may_svrday,
            may_prop, may_ci_lower, may_ci_upper),
            ("June", june_years_of_sample, june_dryday, june_svrday,
            june_prop, june_ci_lower, june_ci_upper),
        ]

    table = pd.DataFrame(
        [
            {
                "Month/group": label,
                "Years": years,
                "Avg dryline days": dry,
                "Avg SGP wind days": severe,
                "SGP wind | dryline": f"{prop:.1%}",
                "95% CI": f"({lower:.1%}, {upper:.1%})",
            }
            for label, years, dry, severe, prop, lower, upper in rows
        ]
    )

    print(f"\nENSO phase: {ph} Wind")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

############################################################
############################################################ proportion of SVR days that are dryline-associated
############################################################ 5-panel
############################################################ 

#3 x 2. For each month, hazard is on the x-axis with bar of avg. number of days on right y-axis, and proportion with CI is on left y-axis
phases = [0, -1, 1]
haz = ['all_report_bool', 'all_tor_report_bool', 'all_hail_report_bool', 'all_wind_report_bool']
dryhaz = ['dry_report_bool', 'dry_tor_report_bool', 'dry_hail_report_bool', 'dry_wind_report_bool']
months = [2,3,4,5,6]

rows = []



for ph in phases:
    for hazard, dryhazard in zip(haz, dryhaz):
        for mo in months:
            month_df = svr_summary[svr_summary['month'] == mo]

            prop, lower, upper, _, n_total = proportion(month_df, hazard, dryhazard, phase=ph, toprint=False)

            phase_df = month_df[month_df[ENSO_TYPE] == ph]
            n_years = phase_df['year'].nunique()

            rows.append({
                "Month": mo,
                "phase": ph,
                "hazard": hazard,
                "proportion": prop,
                "lower": lower,
                "upper": upper,
                "n_years": n_years,
                "avg_svr_days": n_total / n_years if n_years > 0 else 0
            })

plot_data = pd.DataFrame(rows)

def plot_prop(ax, plot_data, hazard, dryhazard, phase):

    if phase == 0:
        offset = 0
        color = 'gray'
        fmt = 'X'
        markersize=13
    elif phase == -1:
        offset = -0.2
        color = 'royalblue'
        fmt = 'o'
        markersize=13
    elif phase == 1:
        offset = 0.2
        color = 'firebrick'
        fmt = 's'
        markersize=11

    for i, (hazard_name, dryhazard_name) in enumerate(zip(hazard, dryhazard)):
        data = plot_data[(plot_data['hazard'] == hazard_name) & (plot_data['phase'] == phase)]
        x = [i + 1 + offset] * len(data)
        ax.errorbar(x, data['proportion']*100, yerr=[(data['proportion'] - data['lower'])*100, (data['upper'] - data['proportion'])*100], fmt=fmt,
                    capsize=5, elinewidth=2.5, capthick=2.5, markersize=markersize, markeredgecolor='black', color=color, zorder=5)

    return

def plot_avg(ax, plot_data, hazard, dryhazard, phase, axt=None):

    if axt is None:
        axt = ax.twinx()

    if phase == 0:
        offset = 0
        color = 'gray'
    elif phase == -1:
        offset = -0.2
        color = 'royalblue'
    elif phase == 1:
        offset = 0.2
        color = 'firebrick'

    for i, (hazard_name, dryhazard_name) in enumerate(zip(hazard, dryhazard)):
        data = plot_data[(plot_data['hazard'] == hazard_name) & (plot_data['phase'] == phase)]
        x = [i + 1 + offset] * len(data)
        axt.bar(x, data['avg_svr_days'], color=color, width=0.2, alpha=0.33, zorder=2)


    if phase == 0:
        if ax==ax3 or ax==ax5:
            axt.set_ylabel('Avg Number of Days', fontsize=24)

        if plot_data.iloc[0]['Month'] == 2:
            axt.yaxis.set_major_locator(MultipleLocator((0.5)))
            axt.yaxis.set_minor_locator(MultipleLocator((0.25)))
        elif plot_data.iloc[0]['Month'] == 3:
            axt.yaxis.set_major_locator(MultipleLocator((1)))
            axt.yaxis.set_minor_locator(MultipleLocator((0.5)))
        elif plot_data.iloc[0]['Month'] == 4:
            axt.yaxis.set_major_locator(MultipleLocator((1)))
            axt.yaxis.set_minor_locator(MultipleLocator((0.5)))
        elif plot_data.iloc[0]['Month'] == 5:
            axt.yaxis.set_major_locator(MultipleLocator((2)))
            axt.yaxis.set_minor_locator(MultipleLocator((1)))
        elif plot_data.iloc[0]['Month'] == 6:
            axt.yaxis.set_major_locator(MultipleLocator((2)))
            axt.yaxis.set_minor_locator(MultipleLocator((1)))


        axt.tick_params(axis='y', which='major', labelsize=18, length=6, width=1.5)
        axt.tick_params(axis='y', which='minor', length=3, width=1)

    return axt


fig, ((ax1, ax2,ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 18))
axes = [ax1, ax2, ax3, ax4, ax5]
ax6.set_axis_off()

xlabs = ["All SVR", "Tornado", "Hail", "Wind"]

panellabs = ["a)", "b)", "c)", "d)", "e)"]
for i, ax in enumerate(axes):
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(xlabs, fontsize=18)
    ax.set_xlim(0.6, len(xlabs) + 0.4)
    ax.tick_params(axis='x', which='major', length=6, width=1.5)

    ax.text(0.02, 1.055, panellabs[i], transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='left', bbox=dict(facecolor='white', edgecolor='black'))

    ax.set_ylim(-1, 101)
    ax.yaxis.set_major_locator(MultipleLocator((20)))
    ax.yaxis.set_minor_locator(MultipleLocator((5)))
    ax.tick_params(axis='y', which='major', labelsize=18, length=6, width=1.5)
    ax.tick_params(axis='y', which='minor', length=3, width=1)

    ax.grid(True, axis='y', linewidth=0.5)

    if ax == ax1 or ax == ax4:
        ax.set_ylabel('Percentage', fontsize=24)
    if ax == ax4 or ax == ax5 or ax == ax3:
        ax.set_xlabel('Hazard Type', fontsize=24)

    

#Feb
plot_prop(ax1, plot_data[plot_data['Month'] == 2], haz, dryhaz, phase=0)
plot_prop(ax1, plot_data[plot_data['Month'] == 2], haz, dryhaz, phase=-1)
plot_prop(ax1, plot_data[plot_data['Month'] == 2], haz, dryhaz, phase=1)
axt = plot_avg(ax1, plot_data[plot_data['Month'] == 2], haz, dryhaz, phase=0)
axt = plot_avg(ax1, plot_data[plot_data['Month'] == 2], haz, dryhaz, phase=-1, axt=axt)
axt = plot_avg(ax1, plot_data[plot_data['Month'] == 2], haz, dryhaz, phase=1, axt=axt)

#March
plot_prop(ax2, plot_data[plot_data['Month'] == 3], haz, dryhaz, phase=0)
plot_prop(ax2, plot_data[plot_data['Month'] == 3], haz, dryhaz, phase=-1)
plot_prop(ax2, plot_data[plot_data['Month'] == 3], haz, dryhaz, phase=1)
axt = plot_avg(ax2, plot_data[plot_data['Month'] == 3], haz, dryhaz, phase=0)
axt = plot_avg(ax2, plot_data[plot_data['Month'] == 3], haz, dryhaz, phase=-1, axt=axt)
axt = plot_avg(ax2, plot_data[plot_data['Month'] == 3], haz, dryhaz, phase=1, axt=axt)

#April
plot_prop(ax3, plot_data[plot_data['Month'] == 4], haz, dryhaz, phase=0)
plot_prop(ax3, plot_data[plot_data['Month'] == 4], haz, dryhaz, phase=-1)
plot_prop(ax3, plot_data[plot_data['Month'] == 4], haz, dryhaz, phase=1)
axt = plot_avg(ax3, plot_data[plot_data['Month'] == 4], haz, dryhaz, phase=0)
axt = plot_avg(ax3, plot_data[plot_data['Month'] == 4], haz, dryhaz, phase=-1, axt=axt)
axt = plot_avg(ax3, plot_data[plot_data['Month'] == 4], haz, dryhaz, phase=1, axt=axt)

#May
plot_prop(ax4, plot_data[plot_data['Month'] == 5], haz, dryhaz, phase=0)
plot_prop(ax4, plot_data[plot_data['Month'] == 5], haz, dryhaz, phase=-1)
plot_prop(ax4, plot_data[plot_data['Month'] == 5], haz, dryhaz, phase=1)
axt = plot_avg(ax4, plot_data[plot_data['Month'] == 5], haz, dryhaz, phase=0)
axt = plot_avg(ax4, plot_data[plot_data['Month'] == 5], haz, dryhaz, phase=-1, axt=axt)
axt = plot_avg(ax4, plot_data[plot_data['Month'] == 5], haz, dryhaz, phase=1, axt=axt)

#June
plot_prop(ax5, plot_data[plot_data['Month'] == 6], haz, dryhaz, phase=0)
plot_prop(ax5, plot_data[plot_data['Month'] == 6], haz, dryhaz, phase=-1)
plot_prop(ax5, plot_data[plot_data['Month'] == 6], haz, dryhaz, phase=1)
axt = plot_avg(ax5, plot_data[plot_data['Month'] == 6], haz, dryhaz, phase=0)
axt = plot_avg(ax5, plot_data[plot_data['Month'] == 6], haz, dryhaz, phase=-1, axt=axt)
axt = plot_avg(ax5, plot_data[plot_data['Month'] == 6], haz, dryhaz, phase=1, axt=axt)


ax1.set_title("February", fontsize=26)
ax2.set_title("March", fontsize=26)
ax3.set_title("April", fontsize=26)
ax4.set_title("May", fontsize=26)
ax5.set_title("June", fontsize=26)

phase_styles = [
    ("Neutral", "gray", 'X', 13),
    ("La Niña", "royalblue", 'o', 13),
    ("El Niño", "firebrick", 's', 11),
]

handles = []
labels = []

for label, color, fmt, msize in phase_styles:
    bar = Patch(facecolor=color, edgecolor="none", alpha=0.33)

    # Empty data create a legend handle without plotting any points.
    marker_error = ax6.errorbar(
        [0], [0],
        yerr=1,
        fmt=fmt,
        color=color,
        markersize=msize,
        markeredgecolor='black',
        capsize=5,
        elinewidth=2.5,
        capthick=2.5,
    )

    marker_error.remove()  # Remove the temporary errorbar from the plot
    handles.append((bar, marker_error))
    labels.append(label)

def centered_legend_patch(
    legend, orig_handle, xdescent, ydescent, width, height, fontsize
):
    return Rectangle(
        (-xdescent, -ydescent / 2),
        width,
        height,
    )

ax6.legend(
    handles=handles,
    labels=labels,
    handler_map={
        tuple: HandlerTuple(ndivide=1),
        Patch: HandlerPatch(patch_func=centered_legend_patch),
        ErrorbarContainer: HandlerErrorbar(
            numpoints=1,
            yerr_size=1.2,
        ),
    },
    loc="center",
    fontsize=22,
    framealpha=1,
    handlelength=2.5,
    handleheight=3,
    labelspacing=1.2,
    title="ENSO Phase",
    title_fontsize=26,
)


plt.suptitle("Percentage of SVR Days which are Dryline-Associated", fontsize=32, y=0.99)
fig.tight_layout()
plt.savefig(f'{directory_path}/dryline_associated_svr_proportion.png', format='png', dpi=150, bbox_inches='tight') 


############################################################ 
############################################################
############################################################ Spatial 4-quadrant analysis of frequency and size
############################################################ 5-panel
############################################################ 
############################################################ 

#Idea: each panel is month with statistics; 6th panel is a map that denotes the domain quadrants
#similar style to "dryline_associated_svr_proportion": point and error bar is % associated with drylines; underlying bar is total SVR area
#run this twice: once on all-svr (for paper), and once on just tor+hail (for appendix?)

#don't use CI... not enough years to be reliable
#use a min/max of leave-one-out ratios

phases = [0, -1, 1]
months = [2,3,4,5,6]

def area_fraction_ci(df, quadrant_a, quadrant_b, n_boot=10_000, seed=67, is_all=True):

    if is_all==True:
        dry_col_a = f"dry_report_q{quadrant_a}_area"
        all_col_a = f"all_report_q{quadrant_a}_area"
        dry_col_b = f"dry_report_q{quadrant_b}_area"
        all_col_b = f"all_report_q{quadrant_b}_area"

        dry_cols = [dry_col_a, dry_col_b]
        all_cols = [all_col_a, all_col_b]
    else:
        dry_col_at = f"dry_tor_report_q{quadrant_a}_area"
        all_col_at = f"all_tor_report_q{quadrant_a}_area"
        dry_col_bt = f"dry_tor_report_q{quadrant_b}_area"
        all_col_bt = f"all_tor_report_q{quadrant_b}_area"

        dry_col_ah = f"dry_hail_report_q{quadrant_a}_area"
        all_col_ah = f"all_hail_report_q{quadrant_a}_area"
        dry_col_bh = f"dry_hail_report_q{quadrant_b}_area"
        all_col_bh = f"all_hail_report_q{quadrant_b}_area"

        dry_cols = [dry_col_at, dry_col_bt, dry_col_ah, dry_col_bh]
        all_cols = [all_col_at, all_col_bt, all_col_ah, all_col_bh]


    areas = df[dry_cols + all_cols].fillna(0)

    dry_col = "dry_area"
    all_col = "all_area"

    data = df[['year']].copy()
    data[dry_col] = areas[dry_cols].sum(axis=1)
    data[all_col] = areas[all_cols].sum(axis=1)

    # Retain only days with report area in this quadrant.
    data = data.loc[data[all_col] > 0].copy()

    if data.empty:
        raise ValueError(
            "No days have positive report area in this quadrant"
        )
    
    yearly = (
        data.groupby("year")[[dry_col, all_col]]
        .sum()
        .to_numpy(dtype=float)
    )

    annual_fractions = yearly[:, 0] / yearly[:, 1]

    remaining = yearly.sum(axis=0) - yearly
    loo_fraction = remaining[:, 0] / remaining[:, 1]
    loo_lower = loo_fraction.min()
    loo_upper = loo_fraction.max()


    total_area = yearly[:, 1].sum()
    if total_area <= 0:
        raise ValueError("No report area: the fraction is undefined")

    fraction = yearly[:, 0].sum() / total_area

    lower, upper = loo_lower, loo_upper



    return fraction, lower, upper, total_area, annual_fractions

rows = []
for ph in phases:
    for mo in months:
        subset = svr_summary[(svr_summary['month'] == mo) & (svr_summary[ENSO_TYPE] == ph)]
        n_years = subset['year'].nunique()
        fraction, lower, upper, total_area, annual_fractions = area_fraction_ci(subset, quadrant_a=2, quadrant_b=3, n_boot=10, is_all=True)
        # print(f"Phase: {ph}, Month: {mo}, Fraction: {fraction}, CI: ({lower}, {upper}), N Years: {n_years}, Avg Area: {total_area / n_years if n_years > 0 else 0}")

        rows.append({
            "Phase": ph,
            "Month": mo,
            "Fraction": fraction,
            "Lower LOO": lower,
            "Upper LOO": upper,
            "N Years": n_years,
            "Avg Area": total_area / n_years if n_years > 0 else 0,
            "Annual Fractions": annual_fractions
        })

plot_data_all_west = pd.DataFrame(rows)

rows = []
for ph in phases:
    for mo in months:
        subset = svr_summary[(svr_summary['month'] == mo) & (svr_summary[ENSO_TYPE] == ph)]
        n_years = subset['year'].nunique()
        fraction, lower, upper, total_area, annual_fractions = area_fraction_ci(subset, quadrant_a=1, quadrant_b=4, n_boot=10, is_all=True)
        # print(f"Phase: {ph}, Month: {mo}, Fraction: {fraction}, CI: ({lower}, {upper}), N Years: {n_years}, Avg Area: {total_area / n_years if n_years > 0 else 0}")

        rows.append({
            "Phase": ph,
            "Month": mo,
            "Fraction": fraction,
            "Lower LOO": lower,
            "Upper LOO": upper,
            "N Years": n_years,
            "Avg Area": total_area / n_years if n_years > 0 else 0,
            "Annual Fractions": annual_fractions
        })

plot_data_all_east = pd.DataFrame(rows)

rows = []
for ph in phases:
    for mo in months:
        subset = svr_summary[(svr_summary['month'] == mo) & (svr_summary[ENSO_TYPE] == ph)]
        n_years = subset['year'].nunique()
        fraction, lower, upper, total_area, annual_fractions = area_fraction_ci(subset, quadrant_a=2, quadrant_b=3, n_boot=10, is_all=False)
        # print(f"Phase: {ph}, Month: {mo}, Fraction: {fraction}, CI: ({lower}, {upper}), N Years: {n_years}, Avg Area: {total_area / n_years if n_years > 0 else 0}")

        rows.append({
            "Phase": ph,
            "Month": mo,
            "Fraction": fraction,
            "Lower LOO": lower,
            "Upper LOO": upper,
            "N Years": n_years,
            "Avg Area": total_area / n_years if n_years > 0 else 0,
            "Annual Fractions": annual_fractions
        })

plot_data_torhail_west = pd.DataFrame(rows)

rows = []
for ph in phases:
    for mo in months:
        subset = svr_summary[(svr_summary['month'] == mo) & (svr_summary[ENSO_TYPE] == ph)]
        n_years = subset['year'].nunique()
        fraction, lower, upper, total_area, annual_fractions = area_fraction_ci(subset, quadrant_a=1, quadrant_b=4, n_boot=10, is_all=False)
        # print(f"Phase: {ph}, Month: {mo}, Fraction: {fraction}, CI: ({lower}, {upper}), N Years: {n_years}, Avg Area: {total_area / n_years if n_years > 0 else 0}")

        rows.append({
            "Phase": ph,
            "Month": mo,
            "Fraction": fraction,
            "Lower LOO": lower,
            "Upper LOO": upper,
            "N Years": n_years,
            "Avg Area": total_area / n_years if n_years > 0 else 0,
            "Annual Fractions": annual_fractions
        })

plot_data_torhail_east = pd.DataFrame(rows)



def plot_area(ax, plot_data_west, plot_data_east, plot_data_torhail_west, plot_data_torhail_east, phase, axt=None):

    if axt is None:
        axt = ax.twinx()

    if phase == 0:
        offset = 0
        color = 'gray'
        fmt = 'X'
        markersize=13
    elif phase == -1:
        offset = -0.2
        color = 'royalblue'
        fmt='o'
        markersize=13
    elif phase == 1:
        offset = 0.2
        color = 'firebrick'
        fmt='s'
        markersize=11

    x = 0


    data_west = plot_data_west[plot_data_west['Phase'] == phase]
    data_east = plot_data_east[plot_data_east['Phase'] == phase]
    data_torhail_west = plot_data_torhail_west[plot_data_torhail_west['Phase'] == phase]
    data_torhail_east = plot_data_torhail_east[plot_data_torhail_east['Phase'] == phase]

    groups = [
    data_west,
    data_east,
    data_torhail_west,
    data_torhail_east,
    ]

    for position, group in enumerate(groups):
        if len(group) != 1:
            raise ValueError(
                "Expected exactly one summary row per month, phase, "
                "region, and hazard combination"
            )

        annual = np.asarray(
            group.iloc[0]["Annual Fractions"],
            dtype=float,
        )

        jitter = (
            np.linspace(-0.055, 0.055, annual.size)
            if annual.size > 1
            else np.zeros(annual.size)
        )

        ax.scatter(
            x + offset + position + jitter,
            annual * 100,
            s=35,
            marker=fmt,
            facecolors="none",
            edgecolors=color,
            linewidths=1.3,
            zorder=6,
        )

    s = 300
    if fmt == 'X':
        s = 400
    ax.scatter(x+offset, data_west['Fraction']*100, color=color, s=s, marker=fmt, edgecolors='black', zorder=5)
    ax.scatter(x+offset+1, data_east['Fraction']*100, color=color, s=s, marker=fmt, edgecolors='black', zorder=5)
    ax.scatter(x+offset+2, data_torhail_west['Fraction']*100, color=color, s=s, marker=fmt, edgecolors='black', zorder=5)
    ax.scatter(x+offset+3, data_torhail_east['Fraction']*100, color=color, s=s, marker=fmt, edgecolors='black', zorder=5)

    ax.vlines(x+offset, data_west['Lower LOO']*100, data_west['Upper LOO']*100, color=color, linewidth=3)
    ax.vlines(x+offset+1, data_east['Lower LOO']*100, data_east['Upper LOO']*100, color=color, linewidth=3)
    ax.vlines(x+offset+2, data_torhail_west['Lower LOO']*100, data_torhail_west['Upper LOO']*100, color=color, linewidth=3)
    ax.vlines(x+offset+3, data_torhail_east['Lower LOO']*100, data_torhail_east['Upper LOO']*100, color=color, linewidth=3)

    axt.bar(x+offset, data_west['Avg Area'], color=color, width=0.2, alpha=0.33, zorder=2)
    axt.bar(x+offset+1, data_east['Avg Area'], color=color, width=0.2, alpha=0.33, zorder=2)
    axt.bar(x+offset+2, data_torhail_west['Avg Area'], color=color, width=0.2, alpha=0.33, zorder=2)
    axt.bar(x+offset+3, data_torhail_east['Avg Area'], color=color, width=0.2, alpha=0.33, zorder=2)

    if phase == 0:
        if ax==ax3 or ax==ax5:
            axt.set_ylabel('Avg Area\n(number of grid cells)', fontsize=24)

        if ax==ax1:
            axt.yaxis.set_major_locator(MultipleLocator((2)))
            axt.yaxis.set_minor_locator(MultipleLocator((0.5)))
        elif ax==ax2:
            axt.yaxis.set_major_locator(MultipleLocator((5)))
            axt.yaxis.set_minor_locator(MultipleLocator((1)))
        elif ax==ax3:
            axt.yaxis.set_major_locator(MultipleLocator((10)))
            axt.yaxis.set_minor_locator(MultipleLocator((5)))
        elif ax==ax4:
            axt.yaxis.set_major_locator(MultipleLocator((20)))
            axt.yaxis.set_minor_locator(MultipleLocator((10)))
        elif ax==ax5:
            axt.yaxis.set_major_locator(MultipleLocator((20)))
            axt.yaxis.set_minor_locator(MultipleLocator((10)))


        axt.tick_params(axis='y', which='major', labelsize=18, length=6, width=1.5)
        axt.tick_params(axis='y', which='minor', length=3, width=1)


    return axt


fig, ((ax1, ax2,ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(28, 18))
axes = [ax1, ax2, ax3, ax4, ax5]
ax6.set_axis_off()

xlabs = ["All SVR\nWest", "All SVR\nEast", "Tornado+Hail\nWest", "Tornado+Hail\nEast"]

panellabs = ["a)", "b)", "c)", "d)", "e)"]
for i, ax in enumerate(axes):
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(xlabs, fontsize=18)
    ax.set_xlim(-0.6, 3.4)
    ax.tick_params(axis='x', which='major', length=6, width=1.5)

    ax.text(0.02, 1.063, panellabs[i], transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='left', bbox=dict(facecolor='white', edgecolor='black'))

    ax.set_ylim(-1, 101)
    ax.yaxis.set_major_locator(MultipleLocator((20)))
    ax.yaxis.set_minor_locator(MultipleLocator((5)))
    ax.tick_params(axis='y', which='major', labelsize=18, length=6, width=1.5)
    ax.tick_params(axis='y', which='minor', length=3, width=1)

    ax.grid(True, axis='y', linewidth=0.5)

    if ax == ax1 or ax == ax4:
        ax.set_ylabel('Percentage', fontsize=24)
    if ax == ax4 or ax == ax5 or ax == ax3:
        ax.set_xlabel('Hazard and Region', fontsize=24)

    ax.vlines(1.5, ymin=-1, ymax=101, colors='black', linestyles='dashed', linewidth=2)

    

#Feb
ax1t = plot_area(ax1, plot_data_all_west[plot_data_all_west['Month'] == 2], plot_data_all_east[plot_data_all_east['Month'] == 2], plot_data_torhail_west[plot_data_torhail_west['Month'] == 2], plot_data_torhail_east[plot_data_torhail_east['Month'] == 2], phase=0)
ax1t = plot_area(ax1, plot_data_all_west[plot_data_all_west['Month'] == 2], plot_data_all_east[plot_data_all_east['Month'] == 2], plot_data_torhail_west[plot_data_torhail_west['Month'] == 2], plot_data_torhail_east[plot_data_torhail_east['Month'] == 2], phase=1, axt=ax1t)
ax1t = plot_area(ax1, plot_data_all_west[plot_data_all_west['Month'] == 2], plot_data_all_east[plot_data_all_east['Month'] == 2], plot_data_torhail_west[plot_data_torhail_west['Month'] == 2], plot_data_torhail_east[plot_data_torhail_east['Month'] == 2], phase=-1, axt=ax1t)

#March
ax2t = plot_area(ax2, plot_data_all_west[plot_data_all_west['Month'] == 3], plot_data_all_east[plot_data_all_east['Month'] == 3], plot_data_torhail_west[plot_data_torhail_west['Month'] == 3], plot_data_torhail_east[plot_data_torhail_east['Month'] == 3], phase=0)
ax2t = plot_area(ax2, plot_data_all_west[plot_data_all_west['Month'] == 3], plot_data_all_east[plot_data_all_east['Month'] == 3], plot_data_torhail_west[plot_data_torhail_west['Month'] == 3], plot_data_torhail_east[plot_data_torhail_east['Month'] == 3], phase=1, axt=ax2t)
ax2t = plot_area(ax2, plot_data_all_west[plot_data_all_west['Month'] == 3], plot_data_all_east[plot_data_all_east['Month'] == 3], plot_data_torhail_west[plot_data_torhail_west['Month'] == 3], plot_data_torhail_east[plot_data_torhail_east['Month'] == 3], phase=-1, axt=ax2t)

#April
ax3t = plot_area(ax3, plot_data_all_west[plot_data_all_west['Month'] == 4], plot_data_all_east[plot_data_all_east['Month'] == 4], plot_data_torhail_west[plot_data_torhail_west['Month'] == 4], plot_data_torhail_east[plot_data_torhail_east['Month'] == 4], phase=0)
ax3t = plot_area(ax3, plot_data_all_west[plot_data_all_west['Month'] == 4], plot_data_all_east[plot_data_all_east['Month'] == 4], plot_data_torhail_west[plot_data_torhail_west['Month'] == 4], plot_data_torhail_east[plot_data_torhail_east['Month'] == 4], phase=1, axt=ax3t)
ax3t = plot_area(ax3, plot_data_all_west[plot_data_all_west['Month'] == 4], plot_data_all_east[plot_data_all_east['Month'] == 4], plot_data_torhail_west[plot_data_torhail_west['Month'] == 4], plot_data_torhail_east[plot_data_torhail_east['Month'] == 4], phase=-1, axt=ax3t)

#May
ax4t = plot_area(ax4, plot_data_all_west[plot_data_all_west['Month'] == 5], plot_data_all_east[plot_data_all_east['Month'] == 5], plot_data_torhail_west[plot_data_torhail_west['Month'] == 5], plot_data_torhail_east[plot_data_torhail_east['Month'] == 5], phase=0)
ax4t = plot_area(ax4, plot_data_all_west[plot_data_all_west['Month'] == 5], plot_data_all_east[plot_data_all_east['Month'] == 5], plot_data_torhail_west[plot_data_torhail_west['Month'] == 5], plot_data_torhail_east[plot_data_torhail_east['Month'] == 5], phase=1, axt=ax4t)
ax4t = plot_area(ax4, plot_data_all_west[plot_data_all_west['Month'] == 5], plot_data_all_east[plot_data_all_east['Month'] == 5], plot_data_torhail_west[plot_data_torhail_west['Month'] == 5], plot_data_torhail_east[plot_data_torhail_east['Month'] == 5], phase=-1, axt=ax4t)

#June
ax5t = plot_area(ax5, plot_data_all_west[plot_data_all_west['Month'] == 6], plot_data_all_east[plot_data_all_east['Month'] == 6], plot_data_torhail_west[plot_data_torhail_west['Month'] == 6], plot_data_torhail_east[plot_data_torhail_east['Month'] == 6], phase=0)
ax5t = plot_area(ax5, plot_data_all_west[plot_data_all_west['Month'] == 6], plot_data_all_east[plot_data_all_east['Month'] == 6], plot_data_torhail_west[plot_data_torhail_west['Month'] == 6], plot_data_torhail_east[plot_data_torhail_east['Month'] == 6], phase=1, axt=ax5t)
ax5t = plot_area(ax5, plot_data_all_west[plot_data_all_west['Month'] == 6], plot_data_all_east[plot_data_all_east['Month'] == 6], plot_data_torhail_west[plot_data_torhail_west['Month'] == 6], plot_data_torhail_east[plot_data_torhail_east['Month'] == 6], phase=-1, axt=ax5t)


ax1.set_title("February", fontsize=26)
ax2.set_title("March", fontsize=26)
ax3.set_title("April", fontsize=26)
ax4.set_title("May", fontsize=26)
ax5.set_title("June", fontsize=26)

#add a map for reference in ax6
llcrnrlat=28
llcrnrlon=360-106
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

#define SGP region
sgp = shapely.Polygon(((-105,35), (-105, 40), (-100,40), (-95, 40), (-95, 30), (-100, 30), (-105, 30)))
sgpx,sgpy = sgp.exterior.xy
sgpx,sgpy =m(sgpx,sgpy)

ax = ax6
ax.text(0.02, 1.063, "f)", transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='left', bbox=dict(facecolor='white', edgecolor='black'))

m.drawcoastlines(linewidth=1,color='k', ax=ax)
m.drawcountries(linewidth=1,color='k', ax=ax)
m.drawstates(linewidth=1, color='k', ax=ax)
m.drawmeridians(range(-180, 180, 5), labels = [0,0,0,1], linewidth=0.25, ax=ax, fontsize=12)
m.drawparallels(range(-10, 90, 5), labels = [1,0,0,0], linewidth=0.25, ax=ax, fontsize=12)

m.plot(sgpx, sgpy, color='#0072B2', linewidth=5, ax=ax, zorder=13)

dividing_line = shapely.LineString(((-100, 30), (-100, 40)))
dividing_linex, dividing_liney = dividing_line.exterior.xy if dividing_line.geom_type == 'Polygon' else dividing_line.xy
dividing_linex, dividing_liney = m(dividing_linex, dividing_liney)
m.plot(dividing_linex, dividing_liney, color='k', linewidth=5, linestyle='--', ax=ax, zorder=13)

text_xe, text_ye = m(-102.5, 36)
ax.text(text_xe, text_ye, "West", fontsize=20, fontweight='bold', color="black", ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', pad=5))
text_xe, text_ye = m(-97.5, 36)
ax.text(text_xe, text_ye, "East", fontsize=20, fontweight='bold', color="black", ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', pad=5))


phase_styles = [
    ("Neutral", "gray", 'X', 15),
    ("La Niña", "royalblue", 'o', 13),
    ("El Niño", "firebrick", 's', 11),
]

handles = []
labels = []

for label, color, fmt, msize in phase_styles:
    bar = Patch(facecolor=color, edgecolor="none", alpha=0.33)

    # Empty data create a legend handle without plotting any points.
    marker_error = ax6.errorbar(
        [0], [0],
        yerr=1,
        fmt=fmt,
        color=color,
        markersize=msize,
        elinewidth=2.5,
    )

    marker_error.remove()  # Remove the temporary errorbar from the plot
    handles.append((bar, marker_error))
    labels.append(label)

def centered_legend_patch(
    legend, orig_handle, xdescent, ydescent, width, height, fontsize
):
    return Rectangle(
        (-xdescent, -ydescent / 2),
        width,
        height,
    )

legend = ax6.legend(
    handles=handles,
    labels=labels,
    handler_map={
        tuple: HandlerTuple(ndivide=1),
        Patch: HandlerPatch(patch_func=centered_legend_patch),
        ErrorbarContainer: HandlerErrorbar(
            numpoints=1,
            yerr_size=1.2,
        ),
    },
    loc="center left",
    bbox_to_anchor=(0.92, 0.5),
    bbox_transform=ax6.transAxes,
    borderaxespad=0,
    fontsize=22,
    framealpha=1,
    handlelength=2.5,
    handleheight=3,
    labelspacing=1.2,
    title="ENSO Phase",
    title_fontsize=26,
)

legend.set_in_layout(False)


suptitle = plt.suptitle("Percentage of SVR Area Associated with the Dryline", fontsize=32, y=0.993)
fig.tight_layout()
plt.savefig(f'{directory_path}/dryline_associated_svr_area_quadrant.png', format='png', dpi=150, bbox_inches='tight', bbox_extra_artists=(legend,suptitle)) 





#################################################################
#################################################################
#################################################################
################################################################# final plot: bar chart; for each month what % of drylines which are assoicated with SVR are active/adjacent/quiescent
#################################################################
#################################################################
#################################################################

""" FOR PLOTTING WHAT % OF DRYLINE SVR DAYS COME FROM WHICH DRYLINES
svr_dry_onlyall = svr_summary[svr_summary['dry_report_bool']]
svr_dry_onlytorhail = svr_summary[(svr_summary['dry_tor_report_bool']) | (svr_summary['dry_hail_report_bool'])]

severe_df = svr_dry_onlyall
# severe_df = svr_dry_onlytorhail

dryline_types = [
    "cyclone_associated_dryline",
    "cyclone_adjacent_dryline",
    "quiescent_dryline",
]
type_labels = [
    "Cyclone-\nAssociated",
    "Cyclone-\nAdjacent",
    "Quiescent",
]
month_names = {
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
}
phase_styles = {
    -1: ("La Niña", "royalblue", -0.22),
    0: ("Neutral", "gray", 0.0),
    1: ("El Niño", "firebrick", 0.22),
}

# Every severe dryline day must belong to exactly one category.
flags = severe_df[dryline_types]
if flags.isna().any().any() or not flags.isin([True, False]).all().all():
    raise ValueError("Dryline categories must contain nonmissing Boolean values.")
if not flags.sum(axis=1).eq(1).all():
    raise ValueError("Each severe dryline day must have exactly one dryline type.")

rows = []

for month in month_names:
    for phase in phase_styles:
        exposure = svr_summary.loc[
            (svr_summary["month"] == month)
            & (svr_summary["enso_op"] == phase)
        ]
        n_years = exposure["year"].nunique()

        if n_years == 0:
            raise ValueError(
                f"No sampling years for month={month}, phase={phase}."
            )

        group = severe_df.loc[
            (severe_df["month"] == month)
            & (severe_df["enso_op"] == phase)
        ]
        n_total = len(group)

        for dryline_type in dryline_types:
            n_type = int(group[dryline_type].sum())

            if n_total > 0:
                prop = n_type / n_total
                lower, upper = proportion_confint(
                    count=n_type,
                    nobs=n_total,
                    alpha=0.05,
                    method="wilson",
                )
            else:
                # No severe dryline days: percentage is undefined, not zero.
                prop = lower = upper = np.nan

            rows.append({
                "month": month,
                "phase": phase,
                "dryline_type": dryline_type,
                "n_type": n_type,
                "n_total": n_total,
                "n_years": n_years,
                "percentage": 100 * prop,
                "lower": 100 * lower,
                "upper": 100 * upper,
                "avg_days": n_type / n_years,
            })

dryline_plot_data = pd.DataFrame(rows)
"""

month_names = {
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
}
dryline_types = [
    "cyclone_associated_dryline",
    "cyclone_adjacent_dryline",
    "quiescent_dryline",
]
type_labels = [
    "Cyclone-\nAssociated",
    "Cyclone-\nAdjacent",
    "Quiescent",
]

phase_order = [-1, 0, 1]

def build_dryline_severity_data(df, severe_col):
    flags = df[dryline_types]
    if flags.isna().any().any() or not flags.isin([True, False]).all().all():
        raise ValueError("Dryline categories must contain nonmissing Boolean values.")
    if flags.sum(axis=1).gt(1).any():
        raise ValueError("A day cannot have more than one dryline type.")

    severe = df[severe_col].astype(bool)
    unclassified = severe & flags.sum(axis=1).eq(0)
    if unclassified.any():
        raise ValueError(
            f"{unclassified.sum()} severe dryline days have no dryline type."
        )

    rows = []

    for month in month_names:
        for phase in phase_order:
            group = df.loc[(df["month"] == month) & (df[ENSO_TYPE] == phase)]
            n_years = group["year"].nunique()

            if n_years == 0:
                raise ValueError(
                    f"No sampling years for month={month}, phase={phase}."
                )

            for dryline_type in dryline_types:
                is_type = group[dryline_type]
                n_dryline = int(is_type.sum())
                n_severe = int((is_type & severe.loc[group.index]).sum())

                if n_dryline > 0:
                    prop = n_severe / n_dryline
                    lower, upper = proportion_confint(
                        count=n_severe,
                        nobs=n_dryline,
                        alpha=0.05,
                        method="wilson",
                    )
                else:
                    # No drylines of this type: percentage is undefined.
                    prop = lower = upper = np.nan

                rows.append({
                    "month": month,
                    "phase": phase,
                    "dryline_type": dryline_type,
                    "n_severe": n_severe,
                    "n_dryline": n_dryline,
                    "n_years": n_years,
                    "percentage": 100 * prop,
                    "lower": 100 * min(lower, prop),
                    "upper": 100 * max(upper, prop),
                    "avg_days": n_dryline / n_years,
                })

    return pd.DataFrame(rows)


svr_summary["dry_torhail_report_bool"] = (
    svr_summary["dry_tor_report_bool"] | svr_summary["dry_hail_report_bool"]
)

dryline_plot_data = build_dryline_severity_data(svr_summary, "dry_report_bool")


# PLOT SEPARATING BY ENSO
# fig, axs = plt.subplots(2, 3, figsize=(24, 18))
# axes = axs.ravel()[:5]
# legend_ax = axs.ravel()[5]
# legend_ax.set_axis_off()

# x = np.arange(1, 4)
# bar_axes = []

# # A common count scale makes bars comparable between monthly panels.
# count_max = max(1.0, 1.15 * dryline_plot_data["avg_days"].max())

# for i, (ax, (month, month_name)) in enumerate(
#     zip(axes, month_names.items())
# ):
#     axt = ax.twinx()
#     bar_axes.append(axt)

#     # Put the entire percentage axis above the bar axis.
#     ax.set_zorder(axt.get_zorder() + 1)
#     ax.patch.set_visible(False)

#     month_data = dryline_plot_data.loc[
#         dryline_plot_data["month"] == month
#     ]

#     for phase, (label, color, offset) in phase_styles.items():
#         data = (
#             month_data.loc[month_data["phase"] == phase]
#             .set_index("dryline_type")
#             .loc[dryline_types]
#         )
#         positions = x + offset

#         axt.bar(
#             positions,
#             data["avg_days"],
#             width=0.20,
#             color=color,
#             alpha=0.33,
#         )

#         valid = data["percentage"].notna()
#         values = data.loc[valid]

#         ax.errorbar(
#             positions[valid.to_numpy()],
#             values["percentage"].to_numpy(),
#             yerr=np.vstack([
#                 (values["percentage"] - values["lower"]).to_numpy(),
#                 (values["upper"] - values["percentage"]).to_numpy(),
#             ]),
#             fmt="o",
#             color=color,
#             markeredgecolor="black",
#             markersize=11,
#             capsize=5,
#             elinewidth=2.5,
#             capthick=2.5,
#             zorder=5,
#         )

#         if not valid.any():
#             ax.text(
#                 0.98, 0.97 - 0.055 * list(phase_styles).index(phase),
#                 f"{label}: no severe dryline days",
#                 transform=ax.transAxes,
#                 ha="right",
#                 va="top",
#                 color=color,
#                 fontsize=12,
#             )

#     ax.set_title(month_name, fontsize=26)
#     ax.text(
#         0.02, 1.055, f"{chr(97 + i)})",
#         transform=ax.transAxes,
#         fontsize=24,
#         fontweight="bold",
#         va="top",
#         bbox=dict(facecolor="white", edgecolor="black"),
#     )

#     ax.set_xticks(x)
#     ax.set_xticklabels(type_labels, fontsize=18)
#     ax.set_xlim(0.6, 3.4)
#     ax.set_ylim(-1, 101)
#     ax.yaxis.set_major_locator(MultipleLocator(20))
#     ax.yaxis.set_minor_locator(MultipleLocator(5))
#     ax.tick_params(axis="y", labelsize=18)
#     ax.grid(axis="y", linewidth=0.5)

#     axt.set_ylim(0, count_max)
#     axt.tick_params(axis="y", labelsize=18)

#     if i in (0, 3):
#         ax.set_ylabel("Percentage of severe dryline days", fontsize=22)
#     if i in (2, 4):
#         axt.set_ylabel("Avg severe dryline days per year", fontsize=22)
#     if i in (2, 3, 4):
#         ax.set_xlabel("Dryline Type", fontsize=22)

# phase_handles = [
#     Line2D(
#         [], [], linestyle="none", marker="o",
#         color=color, markeredgecolor="black",
#         markersize=11, label=label,
#     )
#     for label, color, _ in phase_styles.values()
# ]
# phase_legend = legend_ax.legend(
#     handles=phase_handles,
#     title="ENSO Phase",
#     loc="upper center",
#     fontsize=22,
#     title_fontsize=26,
# )
# legend_ax.add_artist(phase_legend)

# legend_ax.legend(
#     handles=[
#         Line2D(
#             [], [], linestyle="none", marker="o", color="black",
#             markersize=10, label="Percentage; error bars: 95% CI",
#         ),
#         Patch(
#             facecolor="gray", alpha=0.33,
#             label="Average severe dryline days/year",
#         ),
#     ],
#     loc="center",
#     fontsize=17,
# )

# fig.suptitle(
#     "Cyclone Association of Severe Dryline Days",
#     fontsize=32,
#     y=0.99,
# )
# fig.tight_layout(rect=(0, 0, 1, 0.96))
# fig.savefig(
#     "dryline_svr_cyclone_association.png",
#     dpi=150,
#     bbox_inches="tight",
# )


phase_order = [-1, 0, 1]
phase_labels = ["La Niña", "Neutral", "El Niño"]

type_styles = {
    "cyclone_associated_dryline": (
        "Cyclone-Associated", "darkorange", -0.22
    ),
    "cyclone_adjacent_dryline": (
        "Cyclone-Adjacent", "purple", 0.0
    ),
    "quiescent_dryline": (
        "Quiescent", "seagreen", 0.22
    ),
}

fig, axs = plt.subplots(2, 3, figsize=(24, 18))
axes = axs.ravel()[:5]
legend_ax = axs.ravel()[5]
legend_ax.set_axis_off()

x = np.arange(1, 4)
count_max = max(1.0, 1.15 * dryline_plot_data["avg_days"].max())

for i, (ax, (month, month_name)) in enumerate(
    zip(axes, month_names.items())
):
    axt = ax.twinx()

    # Keep percentage markers and error bars above the bars.
    ax.set_zorder(axt.get_zorder() + 1)
    ax.patch.set_visible(False)

    month_data = dryline_plot_data.loc[
        dryline_plot_data["month"] == month
    ]

    for dryline_type, (label, color, offset) in type_styles.items():
        data = (
            month_data.loc[
                month_data["dryline_type"] == dryline_type
            ]
            .set_index("phase")
            .loc[phase_order]
        )
        positions = x + offset

        axt.bar(
            positions,
            data["avg_days"],
            width=0.20,
            color=color,
            alpha=0.33,
        )

        valid = data["percentage"].notna()
        values = data.loc[valid]

        ax.errorbar(
            positions[valid.to_numpy()],
            values["percentage"].to_numpy(),
            yerr=np.vstack([
                (values["percentage"] - values["lower"]).to_numpy(),
                (values["upper"] - values["percentage"]).to_numpy(),
            ]),
            fmt="o",
            color=color,
            markeredgecolor="black",
            markersize=11,
            capsize=5,
            elinewidth=2.5,
            capthick=2.5,
            zorder=5,
        )

    # Annotate missing percentages once per phase, not once per type.
    for j, phase in enumerate(phase_order):
        phase_data = month_data.loc[month_data["phase"] == phase]
        if phase_data["n_dryline"].eq(0).all():
            ax.text(
                x[j], 0.97,
                "No dryline\ndays",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=12,
            )

    ax.set_title(month_name, fontsize=26)
    ax.text(
        0.02, 1.065, f"{chr(97 + i)})",
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        bbox=dict(facecolor="white", edgecolor="black"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels, fontsize=18)
    ax.set_xlim(0.6, 3.4)
    ax.set_ylim(-1, 101)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(axis="y", linewidth=0.5)

    axt.set_ylim(0, count_max)
    axt.tick_params(axis="y", labelsize=18)

    if i in (0, 3):
        ax.set_ylabel("Percentage of dryline days with severe", fontsize=22)
    if i in (2, 4):
        axt.set_ylabel("Avg dryline days per year", fontsize=22)
    if i in (2, 3, 4):
        ax.set_xlabel("ENSO Phase", fontsize=22)



handles = []
labels = []

for dryline_type, (label, color, offset) in type_styles.items():
    bar = Patch(facecolor=color, edgecolor="none", alpha=0.33)

    marker_error = legend_ax.errorbar(
        [0], [0],
        yerr=1,
        fmt="o",
        color=color,
        markersize=11,
        markeredgecolor="black",
        capsize=5,
        elinewidth=2.5,
        capthick=2.5,
    )
    marker_error.remove()

    handles.append((bar, marker_error))
    labels.append(label)

legend_ax.legend(
    handles=handles,
    labels=labels,
    handler_map={
        tuple: HandlerTuple(ndivide=1),
        Patch: HandlerPatch(patch_func=centered_legend_patch),
        ErrorbarContainer: HandlerErrorbar(
            numpoints=1,
            yerr_size=1.2,
        ),
    },
    loc="center",
    fontsize=22,
    framealpha=1,
    handlelength=2.5,
    handleheight=3,
    labelspacing=1.2,
    title="Cyclone Classification",
    title_fontsize=26,
)


fig.suptitle(
    "Percentage of Dryline Days Associated with Severe by Cyclone Classification",
    fontsize=32,
    y=0.98,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(
    f"{directory_path}/dryline_svr_fraction_by_cyclone_classification_by_phase.png",
    dpi=150,
    bbox_inches="tight",
)


###############################################################################################
dryline_plot_data = build_dryline_severity_data(svr_summary, "dry_torhail_report_bool")


dryline_types = [
    "cyclone_associated_dryline",
    "cyclone_adjacent_dryline",
    "quiescent_dryline",
]
type_labels = [
    "Cyclone-\nAssociated",
    "Cyclone-\nAdjacent",
    "Quiescent",
]
month_names = {
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
}
phase_styles = {
    -1: ("La Niña", "royalblue", -0.22),
    0: ("Neutral", "gray", 0.0),
    1: ("El Niño", "firebrick", 0.22),
}

# Every severe dryline day must belong to exactly one category.
# flags = severe_df[dryline_types]
# if flags.isna().any().any() or not flags.isin([True, False]).all().all():
#     raise ValueError("Dryline categories must contain nonmissing Boolean values.")
# if not flags.sum(axis=1).eq(1).all():
#     raise ValueError("Each severe dryline day must have exactly one dryline type.")

# rows = []

# for month in month_names:
#     for phase in phase_styles:
#         exposure = svr_summary.loc[
#             (svr_summary["month"] == month)
#             & (svr_summary["enso_op"] == phase)
#         ]
#         n_years = exposure["year"].nunique()

#         if n_years == 0:
#             raise ValueError(
#                 f"No sampling years for month={month}, phase={phase}."
#             )

#         group = severe_df.loc[
#             (severe_df["month"] == month)
#             & (severe_df["enso_op"] == phase)
#         ]
#         n_total = len(group)

#         for dryline_type in dryline_types:
#             n_type = int(group[dryline_type].sum())

#             if n_total > 0:
#                 prop = n_type / n_total
#                 lower, upper = proportion_confint(
#                     count=n_type,
#                     nobs=n_total,
#                     alpha=0.05,
#                     method="wilson",
#                 )
#             else:
#                 # No severe dryline days: percentage is undefined, not zero.
#                 prop = lower = upper = np.nan

#             rows.append({
#                 "month": month,
#                 "phase": phase,
#                 "dryline_type": dryline_type,
#                 "n_type": n_type,
#                 "n_total": n_total,
#                 "n_years": n_years,
#                 "percentage": 100 * prop,
#                 "lower": 100 * lower,
#                 "upper": 100 * upper,
#                 "avg_days": n_type / n_years,
#             })

# dryline_plot_data = pd.DataFrame(rows)

phase_order = [-1, 0, 1]
phase_labels = ["La Niña", "Neutral", "El Niño"]

type_styles = {
    "cyclone_associated_dryline": (
        "Cyclone-Associated", "darkorange", -0.22
    ),
    "cyclone_adjacent_dryline": (
        "Cyclone-Adjacent", "purple", 0.0
    ),
    "quiescent_dryline": (
        "Quiescent", "seagreen", 0.22
    ),
}

fig, axs = plt.subplots(2, 3, figsize=(24, 18))
axes = axs.ravel()[:5]
legend_ax = axs.ravel()[5]
legend_ax.set_axis_off()

x = np.arange(1, 4)
count_max = max(1.0, 1.15 * dryline_plot_data["avg_days"].max())

for i, (ax, (month, month_name)) in enumerate(
    zip(axes, month_names.items())
):
    axt = ax.twinx()

    # Keep percentage markers and error bars above the bars.
    ax.set_zorder(axt.get_zorder() + 1)
    ax.patch.set_visible(False)

    month_data = dryline_plot_data.loc[
        dryline_plot_data["month"] == month
    ]

    for dryline_type, (label, color, offset) in type_styles.items():
        data = (
            month_data.loc[
                month_data["dryline_type"] == dryline_type
            ]
            .set_index("phase")
            .loc[phase_order]
        )
        positions = x + offset

        axt.bar(
            positions,
            data["avg_days"],
            width=0.20,
            color=color,
            alpha=0.33,
        )

        valid = data["percentage"].notna()
        values = data.loc[valid]

        ax.errorbar(
            positions[valid.to_numpy()],
            values["percentage"].to_numpy(),
            yerr=np.vstack([
                (values["percentage"] - values["lower"]).to_numpy(),
                (values["upper"] - values["percentage"]).to_numpy(),
            ]),
            fmt="o",
            color=color,
            markeredgecolor="black",
            markersize=11,
            capsize=5,
            elinewidth=2.5,
            capthick=2.5,
            zorder=5,
        )

    # Annotate missing percentages once per phase, not once per type.
    for j, phase in enumerate(phase_order):
        phase_data = month_data.loc[month_data["phase"] == phase]
        if phase_data["n_dryline"].eq(0).all():
            ax.text(
                x[j], 0.97,
                "No dryline\ndays",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=12,
            )

    ax.set_title(month_name, fontsize=26)
    ax.text(
        0.02, 1.065, f"{chr(97 + i)})",
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        bbox=dict(facecolor="white", edgecolor="black"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels, fontsize=18)
    ax.set_xlim(0.6, 3.4)
    ax.set_ylim(-1, 101)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(axis="y", linewidth=0.5)

    axt.set_ylim(0, count_max)
    axt.tick_params(axis="y", labelsize=18)

    if i in (0, 3):
        ax.set_ylabel("Percentage of dryline days with severe", fontsize=22)
    if i in (2, 4):
        axt.set_ylabel("Avg drylinedays per year", fontsize=22)
    if i in (2, 3, 4):
        ax.set_xlabel("ENSO Phase", fontsize=22)



handles = []
labels = []

for dryline_type, (label, color, offset) in type_styles.items():
    bar = Patch(facecolor=color, edgecolor="none", alpha=0.33)

    marker_error = legend_ax.errorbar(
        [0], [0],
        yerr=1,
        fmt="o",
        color=color,
        markersize=11,
        markeredgecolor="black",
        capsize=5,
        elinewidth=2.5,
        capthick=2.5,
    )
    marker_error.remove()

    handles.append((bar, marker_error))
    labels.append(label)

legend_ax.legend(
    handles=handles,
    labels=labels,
    handler_map={
        tuple: HandlerTuple(ndivide=1),
        Patch: HandlerPatch(patch_func=centered_legend_patch),
        ErrorbarContainer: HandlerErrorbar(
            numpoints=1,
            yerr_size=1.2,
        ),
    },
    loc="center",
    fontsize=22,
    framealpha=1,
    handlelength=2.5,
    handleheight=3,
    labelspacing=1.2,
    title="Cyclone Classification",
    title_fontsize=26,
)


fig.suptitle(
    "Percentage of Dryline Days Associated with Severe by Cyclone Classification\nTornado and Hail only",
    fontsize=32,
    y=0.98,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(
    f"{directory_path}/dryline_svr_fraction_by_cyclone_classification_by_phase_torhail.png",
    dpi=150,
    bbox_inches="tight",
)


###############################################################################################
dryline_plot_data = build_dryline_severity_data(svr_summary, "dry_tor_report_bool")


dryline_types = [
    "cyclone_associated_dryline",
    "cyclone_adjacent_dryline",
    "quiescent_dryline",
]
type_labels = [
    "Cyclone-\nAssociated",
    "Cyclone-\nAdjacent",
    "Quiescent",
]
month_names = {
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
}
phase_styles = {
    -1: ("La Niña", "royalblue", -0.22),
    0: ("Neutral", "gray", 0.0),
    1: ("El Niño", "firebrick", 0.22),
}


phase_order = [-1, 0, 1]
phase_labels = ["La Niña", "Neutral", "El Niño"]

type_styles = {
    "cyclone_associated_dryline": (
        "Cyclone-Associated", "darkorange", -0.22
    ),
    "cyclone_adjacent_dryline": (
        "Cyclone-Adjacent", "purple", 0.0
    ),
    "quiescent_dryline": (
        "Quiescent", "seagreen", 0.22
    ),
}

fig, axs = plt.subplots(2, 3, figsize=(24, 18))
axes = axs.ravel()[:5]
legend_ax = axs.ravel()[5]
legend_ax.set_axis_off()

x = np.arange(1, 4)
count_max = max(1.0, 1.15 * dryline_plot_data["avg_days"].max())

for i, (ax, (month, month_name)) in enumerate(
    zip(axes, month_names.items())
):
    axt = ax.twinx()

    # Keep percentage markers and error bars above the bars.
    ax.set_zorder(axt.get_zorder() + 1)
    ax.patch.set_visible(False)

    month_data = dryline_plot_data.loc[
        dryline_plot_data["month"] == month
    ]

    for dryline_type, (label, color, offset) in type_styles.items():
        data = (
            month_data.loc[
                month_data["dryline_type"] == dryline_type
            ]
            .set_index("phase")
            .loc[phase_order]
        )
        positions = x + offset

        axt.bar(
            positions,
            data["avg_days"],
            width=0.20,
            color=color,
            alpha=0.33,
        )

        valid = data["percentage"].notna()
        values = data.loc[valid]

        ax.errorbar(
            positions[valid.to_numpy()],
            values["percentage"].to_numpy(),
            yerr=np.vstack([
                (values["percentage"] - values["lower"]).to_numpy(),
                (values["upper"] - values["percentage"]).to_numpy(),
            ]),
            fmt="o",
            color=color,
            markeredgecolor="black",
            markersize=11,
            capsize=5,
            elinewidth=2.5,
            capthick=2.5,
            zorder=5,
        )

    # Annotate missing percentages once per phase, not once per type.
    for j, phase in enumerate(phase_order):
        phase_data = month_data.loc[month_data["phase"] == phase]
        if phase_data["n_dryline"].eq(0).all():
            ax.text(
                x[j], 0.97,
                "No dryline\ndays",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=12,
            )

    ax.set_title(month_name, fontsize=26)
    ax.text(
        0.02, 1.065, f"{chr(97 + i)})",
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        bbox=dict(facecolor="white", edgecolor="black"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels, fontsize=18)
    ax.set_xlim(0.6, 3.4)
    ax.set_ylim(-1, 101)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(axis="y", linewidth=0.5)

    axt.set_ylim(0, count_max)
    axt.tick_params(axis="y", labelsize=18)

    if i in (0, 3):
        ax.set_ylabel("Percentage of dryline days with tornado", fontsize=22)
    if i in (2, 4):
        axt.set_ylabel("Avg drylinedays per year", fontsize=22)
    if i in (2, 3, 4):
        ax.set_xlabel("ENSO Phase", fontsize=22)



handles = []
labels = []

for dryline_type, (label, color, offset) in type_styles.items():
    bar = Patch(facecolor=color, edgecolor="none", alpha=0.33)

    marker_error = legend_ax.errorbar(
        [0], [0],
        yerr=1,
        fmt="o",
        color=color,
        markersize=11,
        markeredgecolor="black",
        capsize=5,
        elinewidth=2.5,
        capthick=2.5,
    )
    marker_error.remove()

    handles.append((bar, marker_error))
    labels.append(label)

legend_ax.legend(
    handles=handles,
    labels=labels,
    handler_map={
        tuple: HandlerTuple(ndivide=1),
        Patch: HandlerPatch(patch_func=centered_legend_patch),
        ErrorbarContainer: HandlerErrorbar(
            numpoints=1,
            yerr_size=1.2,
        ),
    },
    loc="center",
    fontsize=22,
    framealpha=1,
    handlelength=2.5,
    handleheight=3,
    labelspacing=1.2,
    title="Cyclone Classification",
    title_fontsize=26,
)


fig.suptitle(
    "Percentage of Dryline Days Associated with Severe by Cyclone Classification\nTornado only",
    fontsize=32,
    y=0.98,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(
    f"{directory_path}/dryline_svr_fraction_by_cyclone_classification_by_phase_tor.png",
    dpi=150,
    bbox_inches="tight",
)


###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

#binomail model with logic link (logistic regression)
#daily (of all dryline days) logistic model with ENSO phase and cyclone classification as predictors
do_model = False
if do_model == True:

    formula = f"severe ~ C({ENSO_TYPE}) * C(cyclone_classification)" 

    severe_col = "dry_report_bool"

    month_names = {
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
    }

    dryline_types = [
        "cyclone_associated_dryline",
        "cyclone_adjacent_dryline",
        "quiescent_dryline",
    ]

    # Restrict the analysis to dryline days in the study months.
    model_df = svr_summary.loc[
        svr_summary["any_dryline"].eq(True)
        & svr_summary["month"].isin(month_names),
        ["month", ENSO_TYPE, severe_col, *dryline_types],
    ].copy()

    if model_df.empty:
        raise ValueError("No dryline days are available for modeling.")

    if model_df.isna().any().any():
        raise ValueError("Model variables contain missing values.")

    if not model_df[severe_col].isin([True, False]).all():
        raise ValueError(f"{severe_col} must contain Boolean or 0/1 values.")

    if not model_df[ENSO_TYPE].isin([-1, 0, 1]).all():
        raise ValueError("ENSO phase must be -1, 0, or 1.")

    flags = model_df[dryline_types]
    if (
        not flags.isin([True, False]).all().all()
        or not flags.sum(axis=1).eq(1).all()
    ):
        raise ValueError("Each dryline day must have exactly one cyclone class.")

    model_df["severe"] = model_df[severe_col].astype(int)
    model_df["cyclone_class"] = flags.idxmax(axis=1)

    enso_term = f"C({ENSO_TYPE}, Treatment(reference=0))"
    cyclone_term = (
        "C(cyclone_class, Treatment(reference='quiescent_dryline'))"
    )

    test_formulas = {
        "full_model": f"severe ~ {enso_term} * {cyclone_term}",
        "No predictors": "severe ~ 1",
        "No ENSO": f"severe ~ {cyclone_term}",
        "No cyclone": f"severe ~ {enso_term}",
        "No interaction": f"severe ~ {enso_term} + {cyclone_term}",
    }

    tests = []
    monthly_models = {}

    for month, month_name in month_names.items():
        monthly_data = model_df.loc[model_df["month"] == month].copy()

        if monthly_data.empty:
            raise ValueError(f"{month_name}: no dryline days available.")

        if monthly_data["severe"].nunique() != 2:
            raise ValueError(
                f"{month_name}: both severe and nonsevere days are required."
            )

        if set(monthly_data[ENSO_TYPE]) != {-1, 0, 1}:
            raise ValueError(f"{month_name}: not all three ENSO phases are present.")

        if set(monthly_data["cyclone_class"]) != set(dryline_types):
            raise ValueError(
                f"{month_name}: not all three cyclone classes are present."
            )

        monthly_models[month_name] = {}

        for test_name, formula in test_formulas.items():
            model = smf.glm(
                formula=formula,
                data=monthly_data,
                family=sm.families.Binomial(),
                missing="raise",
            )

            if np.linalg.matrix_rank(model.exog) < model.exog.shape[1]:
                raise ValueError(
                    f"{month_name}, {test_name}: rank-deficient design. "
                    "Check for absent ENSO-by-cyclone combinations."
                )

            result = model.fit(maxiter=200)

            if (
                not result.converged
                or not np.isfinite(result.params).all()
                or not np.isfinite(result.llf)
            ):
                raise RuntimeError(
                    f"{month_name}, {test_name}: model fit is not reliable."
                )

            # Bernoulli GLM: one parameter per design column, no fitted dispersion.
            n_parameters = model.exog.shape[1]
            bic = -2 * result.llf + n_parameters * np.log(result.nobs)

            monthly_models[month_name][test_name] = result
            tests.append({
                "month": month_name,
                "test": test_name,
                "n_days": int(result.nobs),
                "n_severe": int(monthly_data["severe"].sum()),
                "n_parameters": n_parameters,
                "bic": bic,
            })

    tests_df = pd.DataFrame(tests)

    bic_table = (
        tests_df.pivot(index="month", columns="test", values="bic")
        .reindex(
            index=list(month_names.values()),
            columns=list(test_formulas),
        )
    )

    bic_relative_to_full_model = bic_table.subtract(
        bic_table["full_model"], axis=0
    )

    print("\nBIC: lower is preferred")
    print(bic_table.round(2).to_string())

    print("\nDelta BIC = candidate model BIC - full model BIC")
    print(bic_relative_to_full_model.round(2).to_string())

    print("\nPreferred model within each month:")
    print(bic_table.idxmin(axis=1).to_string())



# toprint=True
#Dryline vs SGP severe day
# print("Full Feb-June")
# proportion(svr_summary, 'any_dryline', 'all_report_bool', phase=None, toprint=toprint)
# print("Feb")
# proportion(svr_summary[svr_summary['month'] == 2], 'any_dryline', 'all_report_bool', phase=None, toprint=toprint)
# print("March")
# proportion(svr_summary[svr_summary['month'] == 3], 'any_dryline', 'all_report_bool', phase=None, toprint=toprint)
# print("April")
# proportion(svr_summary[svr_summary['month'] == 4], 'any_dryline', 'all_report_bool', phase=None, toprint=toprint)
# print("May")
# proportion(svr_summary[svr_summary['month'] == 5], 'any_dryline', 'all_report_bool', phase=None, toprint=toprint)
# print("June")
# proportion(svr_summary[svr_summary['month'] == 6], 'any_dryline', 'all_report_bool', phase=None, toprint=toprint)

#cyclone and SGP severe day
# print("Full Feb-June")
# proportion(svr_summary, 'sgp_cyclone_day', 'all_report_bool', phase=None, toprint=toprint)
# print("Feb")
# proportion(svr_summary[svr_summary['month'] == 2], 'sgp_cyclone_day', 'all_report_bool', phase=None, toprint=toprint)
# print("March")
# proportion(svr_summary[svr_summary['month'] == 3], 'sgp_cyclone_day', 'all_report_bool', phase=None, toprint=toprint)
# print("April")
# proportion(svr_summary[svr_summary['month'] == 4], 'sgp_cyclone_day', 'all_report_bool', phase=None, toprint=toprint)
# print("May")
# proportion(svr_summary[svr_summary['month'] == 5], 'sgp_cyclone_day', 'all_report_bool', phase=None, toprint=toprint)
# print("June")
# proportion(svr_summary[svr_summary['month'] == 6], 'sgp_cyclone_day', 'all_report_bool', phase=None, toprint=toprint)

#Dryline-associated severe and all SGP severe day
# print("Full Feb-June")
# proportion(svr_summary, 'all_report_bool', 'dry_report_bool', phase=None, toprint=toprint)
# print("Feb")
# proportion(svr_summary[svr_summary['month'] == 2], 'all_report_bool', 'dry_report_bool', phase=None, toprint=toprint)
# print("March")
# proportion(svr_summary[svr_summary['month'] == 3], 'all_report_bool', 'dry_report_bool', phase=None, toprint=toprint)
# print("April")
# proportion(svr_summary[svr_summary['month'] == 4], 'all_report_bool', 'dry_report_bool', phase=None, toprint=toprint)
# print("May")
# proportion(svr_summary[svr_summary['month'] == 5], 'all_report_bool', 'dry_report_bool', phase=None, toprint=toprint)
# print("June")
# proportion(svr_summary[svr_summary['month'] == 6], 'all_report_bool', 'dry_report_bool', phase=None, toprint=toprint)




#repeat for individual hazards
#tor
# condb = "all_tor_report_bool"
# print("Full Feb-June")
# odds_ratio(svr_summary, 'any_dryline', condb, phase=None, toprint=toprint)
# print("Feb")
# odds_ratio(svr_summary[svr_summary['month'] == 2], 'any_dryline', condb, phase=None, toprint=toprint)
# print("March")
# odds_ratio(svr_summary[svr_summary['month'] == 3], 'any_dryline', condb, phase=None, toprint=toprint)
# print("April")
# odds_ratio(svr_summary[svr_summary['month'] == 4], 'any_dryline', condb, phase=None, toprint=toprint)
# print("May")
# odds_ratio(svr_summary[svr_summary['month'] == 5], 'any_dryline', condb, phase=None, toprint=toprint)
# print("June")
# odds_ratio(svr_summary[svr_summary['month'] == 6], 'any_dryline', condb, phase=None, toprint=toprint)

# #hail
# condb = "all_hail_report_bool"
# print("Full Feb-June")
# odds_ratio(svr_summary, 'any_dryline', condb, phase=None, toprint=toprint)
# print("Feb")
# odds_ratio(svr_summary[svr_summary['month'] == 2], 'any_dryline', condb, phase=None, toprint=toprint)
# print("March")
# odds_ratio(svr_summary[svr_summary['month'] == 3], 'any_dryline', condb, phase=None, toprint=toprint)
# print("April")
# odds_ratio(svr_summary[svr_summary['month'] == 4], 'any_dryline', condb, phase=None, toprint=toprint)
# print("May")
# odds_ratio(svr_summary[svr_summary['month'] == 5], 'any_dryline', condb, phase=None, toprint=toprint)
# print("June")
# odds_ratio(svr_summary[svr_summary['month'] == 6], 'any_dryline', condb, phase=None, toprint=toprint)

# #wind
# condb = "all_wind_report_bool"
# print("Full Feb-June")
# odds_ratio(svr_summary, 'any_dryline', condb, phase=None, toprint=toprint)
# print("Feb")
# odds_ratio(svr_summary[svr_summary['month'] == 2], 'any_dryline', condb, phase=None, toprint=toprint)
# print("March")
# odds_ratio(svr_summary[svr_summary['month'] == 3], 'any_dryline', condb, phase=None, toprint=toprint)
# print("April")
# odds_ratio(svr_summary[svr_summary['month'] == 4], 'any_dryline', condb, phase=None, toprint=toprint)
# print("May")
# odds_ratio(svr_summary[svr_summary['month'] == 5], 'any_dryline', condb, phase=None, toprint=toprint)
# print("June")
# odds_ratio(svr_summary[svr_summary['month'] == 6], 'any_dryline', condb, phase=None, toprint=toprint)


print('Time to run: ' + str(datetime.utcnow() - starttime))