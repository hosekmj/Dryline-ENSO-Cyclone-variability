#create and run the GAM model and analysis
#gpd_plot

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
from scipy.stats import chi2_contingency
from scipy.signal import detrend
import matplotlib.patheffects as pe
from shapely.geometry import box, Point
from shapely.ops import transform
from pyproj import Transformer
import matplotlib.colors as colors
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import NullFormatter
from matplotlib.lines import Line2D

from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform
from sklearn.model_selection import ParameterSampler
from sklearn.calibration import calibration_curve

from scipy.special import expit, logit
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree

from contextlib import redirect_stdout
from pathlib import Path

#from shapely.geometry import LineString
import shapely

#import random

import xarray as xr

#import multiprocessing
#from itertools import repeat

#import imageio
#import glob

from pygam import LogisticGAM, te, l , f
from pygam.utils import OptimizationError


from netCDF4 import Dataset
import os
import argparse

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

if domain_type == "None":
    domain_type = None

directory_path = f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/{terrain_mask}/{ENSO_TYPE}"
if domain_type is not None:
    directory_path = f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/{terrain_mask}/{ENSO_TYPE}/{domain_type}"
os.makedirs(directory_path, exist_ok=True)

starttime = datetime.utcnow()

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
# west = -105
# north = 45
# south = 32
# east = -98

to_meters = Transformer.from_crs(
    "EPSG:4326", "EPSG:5070", always_xy=True
).transform

# lee_domain = transform(to_meters, box(west, south, east, north))

########################### read in data

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
all_drylines = pd.read_pickle('/raid/student/mhosek/era5_drylines/pkl/dryline_envi_1940_2020.pkl')
spring_drylines = all_drylines[(all_drylines['month'] >= 2) & (all_drylines['month'] <= 6)].copy()

#restrict all drylines to just the SGP (use median lat/lon to define whether it's "in" the SGP)
sgp_n = 40
sgp_s = 30
sgp_w = -105
sgp_e = -95
spring_drylines = spring_drylines[(spring_drylines['median_lon'] >= -105) & (spring_drylines['median_lon'] <= -95)]
spring_drylines = spring_drylines[(spring_drylines['median_lat'] >= 30) & (spring_drylines['median_lat'] <= 40)]

sgp_domain = transform(to_meters, box(sgp_w, sgp_s, sgp_e, sgp_n))

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


#first, limit the df to only days when there's a SGP cyclone
all_sgp_within = cyclones_df[cyclones_df['sgp_cyclone'].map(any)].copy(deep=True)

#create a new dataframe for model training that will contain only one cyclone per row
#if multiple cyclones are within the lee domain, chose the one with centroid within the lee domain
#if multiple cyclones have their centroid within the lee domain, choose the one with the lowest MSLP
model_df = pd.DataFrame()
lat = []
lon = []
year = []
month = []
mslp = []
size = []
max_wind = []
is_lee_cyclone = [] #bool if it's a lee cyclone or not
enso_phase = [] #use operational ENSO

dryline = [] #bool, if there's an SGP dryline

for i in range(len(all_sgp_within)):
    lee_within_mask = list(all_sgp_within.iloc[i]['sgp_cyclone']) #list of bools
    within_idx = [j for j, v in enumerate(lee_within_mask) if v] #index of where "true" is

    #if only one cyclone is within lee domain, keep that one (index 0 of cyclone stat list)
    if len(within_idx) == 1:
        k = 0

    else:        
        # if more than one, keep the one with a centroid within the sgp domain
        # if more than one (or none) has centroid within sgp domain, keep lowest MSLP

        row = all_sgp_within.iloc[i]
        centroid_points = [Point(*to_meters(row['lon'][cyclone_idx], row['lat'][cyclone_idx])) for cyclone_idx in within_idx]

        centroid_ks = [candidate_idx for candidate_idx, centroid in enumerate(centroid_points) if sgp_domain.covers(centroid)]
        
        if centroid_ks:
            k = min(centroid_ks, key=lambda c: row['sgp_mslp'][c])
        else:
            k = min(range(len(within_idx)), key=lambda c: row['sgp_mslp'][c])


    j = within_idx[k]
    lat.append(all_sgp_within.iloc[i]['lat'][j])
    lon.append(all_sgp_within.iloc[i]['lon'][j])
    year.append(all_sgp_within.iloc[i]['year'])
    month.append(all_sgp_within.iloc[i]['month'])
    mslp.append(all_sgp_within.iloc[i]['sgp_mslp'][k])
    size.append(all_sgp_within.iloc[i]['sgp_size'][k])
    max_wind.append(all_sgp_within.iloc[i]['sgp_maxwind'][k])
    is_lee_cyclone.append(all_sgp_within.iloc[i]['is_lee_cyclone'][j])
    enso_phase.append(all_sgp_within.iloc[i][ENSO_TYPE])
    dryline.append(all_sgp_within.iloc[i]['dryline'])


model_df['latitude'] = lat
model_df['longitude'] = lon
model_df['year'] = year
model_df['month'] = month
model_df['mslp'] = mslp
model_df['size'] = size
model_df['max_wind'] = max_wind
model_df['is_lee_cyclone'] = is_lee_cyclone
model_df['enso_phase'] = enso_phase
model_df['dryline'] = dryline

#restrict to only years on or after year cutoff
model_df = model_df[model_df['year'] >= YEAR_CUTOFF]

#remove any rows with missing values
# bad_thresh = -99999
# fulllen = len(model_df)
# model_df = model_df[model_df['size'] != bad_thresh]
fulllen = len(model_df)
model_df = model_df.dropna(subset=['size'])
print(f"Removed {fulllen - len(model_df)} out of {fulllen} rows with bad outermost closed isobar")

######################################################################## create the model and analytics

def format_interval(estimates, lower, upper, column, scale=1, decimals=2):
    return [
    (f"{scale * estimates.loc[label, column]:.{decimals}f} "
        f"({scale * lower.loc[label, column]:.{decimals}f}, "
        f"{scale * upper.loc[label, column]:.{decimals}f})") for label in estimates.index]

def create_gam(spatial_lam, linear_lam, enso_lam, lee_lam, month_lam=None):
    #initialize the GAM model given the lam parameters
    
    if month_lam is not None:
        terms = (
        te(0, 1, n_splines=6, lam=spatial_lam) #tensor for spatial
        + l(2, lam=linear_lam) #linear term for mslp
        + l(3, lam=linear_lam) #linear term for size
        + l(4, lam=linear_lam) #linear term for max_wind
        + f(5, lam=enso_lam) #enso categorical
        + f(6, lam=month_lam)) #month categorical
    else:
        terms = (
        te(0, 1, n_splines=6, lam=spatial_lam) #tensor for spatial
        + l(2, lam=linear_lam) #linear term for mslp
        + l(3, lam=linear_lam) #linear term for size
        + l(4, lam=linear_lam) #linear term for max_wind
        + f(5, lam=enso_lam) #enso categorical
        + f(6, lam=lee_lam)) #is lee cyclone categorical
    return LogisticGAM(terms, max_iter=500)

def fit_analysis(raw_data, best_parameters, outcomes):
    scaler = StandardScaler()
    fitted_data = raw_data.copy()
    fitted_data[:, 2:5] = scaler.fit_transform(raw_data[:, 2:5])
    model = create_gam(**best_parameters).fit(fitted_data, outcomes)
    differences = model.logs_.get("diffs", [])
    converged = (
        len(differences) > 0
        and np.isfinite(differences[-1])
        and differences[-1] < model.tol)
    if not converged:
        raise OptimizationError("Model did not converge")

    return model, scaler

def effect_table(model, scaler, reference, comparisons, predictors):
    rows = []
    for label, (settings_a, settings_b) in comparisons.items():
        scenarios = []
        for settings in [settings_a, settings_b]:
            scenario = reference.copy()
            for name, value in settings.items():
                scenario[:, predictors.index(name)] = value
            scenario[:,2:5] = scaler.transform(scenario[:,2:5])
            scenarios.append(scenario)
        scenario_a, scenario_b = scenarios
        prob_a = model.predict_proba(scenarios[0]).mean()
        prob_b = model.predict_proba(scenarios[1]).mean()
        log_odds_change = sum(model.partial_dependence(term=term, X=scenario_b) - model.partial_dependence(term=term, X=scenario_a) for term in range(len(model.terms)-1))
        rows.append({
            "comparison (b-a)": label,
            "prob_a": prob_a,
            "prob_b": prob_b,
            "difference": (prob_b - prob_a)*100,
            "Conditional_OR": float(np.exp(log_odds_change.mean()))
        })
    return pd.DataFrame(rows).set_index("comparison (b-a)")

def bootstrap_effects(raw, outcomes, years, reference, comparisons, predictors, best_parameters, n_bootstrap=10000):
    #run bootstrap 95% CI

    rng = np.random.default_rng(67)
    unique_years = np.unique(years)
    blocks = {year: np.flatnonzero(years==year) for year in unique_years}
    samples = []
    failed_fits = 0
    attempts = 0
    max_attempts = 2 * n_bootstrap

    while len(samples) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        selected = rng.choice(unique_years, len(unique_years), replace=True)
        indices = np.concatenate([blocks[year] for year in selected])
        sampled_outcome = outcomes[indices]

        if np.unique(sampled_outcome).size < 2: #skip if there's ever a sample with only non-dryline days (only really possible for Feb)
            failed_fits += 1
            continue

        try:
            model, scaler = fit_analysis(raw[indices], best_parameters, sampled_outcome)
            effect = effect_table(model, scaler, reference, comparisons, predictors)
            samples.append(effect)
        except OptimizationError:
            failed_fits += 1
            continue
        except Exception as error:
            failed_fits += 1
            print(f"Exception occurred during bootstrap fit: {error}")
            continue

    if len(samples) < n_bootstrap:
        raise RuntimeError(f"Failed to generate enough bootstrap samples: {len(samples)} out of {n_bootstrap}")

    print(f"Generated {len(samples)} bootstrap samples with {failed_fits} failed fits.")
    draws = pd.concat(samples, keys=range(n_bootstrap))
    return (draws.groupby(level=1).quantile(0.025), draws.groupby(level=1).quantile(0.975)) 

def run_gam(df, month=None):
    #given the appropriate cyclone dataframe and a month (int), run the GAM model and
    #print analytics and save out plots

    ########## GAM initialization and hyperparameter setup
    if month is None:
        term_names = ["Location", "MSLP", "Size", "Maximum wind", "ENSO", "month"]
        predictors = ['longitude', 'latitude', 'mslp', 'size', 'max_wind', 'enso_phase', 'month']
        X_train = df[predictors].to_numpy(dtype=float)
        distributions = {
          "spatial_lam": loguniform(1e-4, 1e3),
          "linear_lam": loguniform(1e-4, 1e3),
          "enso_lam": loguniform(1e-4, 1e3),
          "month_lam": loguniform(1e-4, 1e3)} #for determining the best parameters for the model

        filename = "allspring"
        figtitle = "All Spring"

        #B-A comparison
        comparisons = {"La Nina - El Nino": ({"enso_phase": 1}, {"enso_phase": -1}),
                "La Nina - Neutral": ({"enso_phase": 0}, {"enso_phase": -1}),
                "El Nino - Neutral": ({"enso_phase": 0}, {"enso_phase": 1})}

        for name in ["mslp", "size", "max_wind"]:
            column = predictors.index(name)
            mean, sd = X_train[:, column].mean(), X_train[:, column].std()
            comparisons[name + f" (mean): {name} (+1 SD (+{sd:.2f}))"] = ({name: mean}, {name: mean + sd})
            if name == "mslp":
                comparisons["mslp (mean): mslp (+5 hPa)"] = ({ "mslp": mean}, {"mslp": mean + 5})
                comparisons["mslp (mean): mslp (+10 hPa)"] = ({ "mslp": mean}, {"mslp": mean + 10})
    
        month_names = {2: "February", 3: "March", 4: "April", 5: "May", 6: "June"}
        for month_number in range(3, 7):
            comparisons[f"{month_names[month_number]} - February"] = ({"month": 2}, {"month": month_number})

        cyclone_count_threshold = 25

    else:
        term_names = ["Location", "MSLP", "Size", "Maximum wind", "ENSO", "Lee Cyclone"]
        predictors = ['longitude', 'latitude', 'mslp', 'size', 'max_wind', 'enso_phase', "is_lee_cyclone"]
        X_train = df[predictors].to_numpy(dtype=float)
        distributions = {
          "spatial_lam": loguniform(1e-4, 1e3),
          "linear_lam": loguniform(1e-4, 1e3),
          "enso_lam": loguniform(1e-4, 1e3),
          "lee_lam": loguniform(1e-4, 1e3)} #for determining the best parameters for the model

        #B-A comparison
        comparisons = {"La Nina - El Nino": ({"enso_phase": 1}, {"enso_phase": -1}),
                "La Nina - Neutral": ({"enso_phase": 0}, {"enso_phase": -1}),
                "El Nino - Neutral": ({"enso_phase": 0}, {"enso_phase": 1})}

        for name in ["mslp", "size", "max_wind"]:
            column = predictors.index(name)
            mean, sd = X_train[:, column].mean(), X_train[:, column].std()
            comparisons[name + f" (mean): {name} (+1 SD (+{sd:.2f}))"] = ({name: mean}, {name: mean + sd})
            if name == "mslp":
                comparisons["mslp (mean): mslp (+5 hPa)"] = ({ "mslp": mean}, {"mslp": mean + 5})
                comparisons["mslp (mean): mslp (+10 hPa)"] = ({ "mslp": mean}, {"mslp": mean + 10})

        comparisons["Lee Cyclone - Not Lee Cyclone"] = ({"is_lee_cyclone": 0}, {"is_lee_cyclone": 1})

        cyclone_count_threshold = 10

        if month == 2:
            filename = "feb"
            figtitle = "February"
        elif month == 3:
            filename = "march"
            figtitle = "March"
        elif month == 4:
            filename = "april"
            figtitle = "April"
        elif month == 5:
            filename = "may"
            figtitle = "May"
        elif month == 6:
            filename = "june"
            figtitle = "June"
        else:
            print("Month not recognized")
            return

    print(f"RUNNING GAM FOR: {filename}")

    y_train = df['dryline'].to_numpy()

    PARAMETER_NITER = 150
    parameter_sets = list(ParameterSampler(distributions, n_iter=PARAMETER_NITER, random_state=67)) #this selects n_iter random sets of hyperparameters

    fold_scalar = StandardScaler() #initialize scaler for this fold to standardize the MSLP/size/max wind
    groups = df['year'].to_numpy()
    cv = GroupKFold(n_splits=5)
    fold_data = []

    for fit_idx, valid_idx in cv.split(X_train, y_train, groups=groups):
        fit_data = X_train[fit_idx].copy()
        valid_data = X_train[valid_idx].copy()
        fit_data[:, 2:5] = fold_scalar.fit_transform(fit_data[:, 2:5])
        valid_data[:, 2:5] = fold_scalar.transform(valid_data[:, 2:5])
        fold_data.append((fit_idx, valid_idx, fit_data, valid_data))

    results = []
    for number, parameters in enumerate(parameter_sets):
        oof_probability = np.full(len(y_train), np.nan)
        for fit_idx, valid_idx, fit_data, valid_data in fold_data:
            try:
                candidate = create_gam(**parameters).fit(fit_data, y_train[fit_idx])
                differences = candidate.logs_.get("diffs", [])
                converged = (
                    len(differences) > 0
                    and np.isfinite(differences[-1])
                    and differences[-1] < candidate.tol)
                if not converged:
                    raise OptimizationError("Model did not converge")
                
                oof_probability[valid_idx] = candidate.predict_proba(valid_data)
            except OptimizationError as error:
                print(f"Skipping candidate {number} due to optimization error/lack of model convergence")
                oof_probability[valid_idx] = np.nan

        if not np.isfinite(oof_probability).all(): #if bad set of parameters, skip them
            print(f"Skipping candidate {number} due to invalid predictions.")
            results.append({**parameters, "log_loss": np.inf})
            continue
        loss = log_loss(y_train, oof_probability, labels=[0,1])
        results.append({**parameters, "log_loss": loss})
        # print(f"{number}/{len(parameter_sets)}: log loss = {loss:.5f}")

    #if all parameters were skipped, raise an error
    if all(result["log_loss"] == np.inf for result in results):
        raise ValueError("All parameter sets were skipped due to invalid predictions.")

    cv_results = pd.DataFrame(results).sort_values("log_loss")
    best_parameters = {
        name: float(cv_results.iloc[0][name])
        for name in distributions
    }
    print("Selected penalties:", best_parameters)

    ############## basic model evaluation with out-of-fold predictions
    oof_probability = np.full(len(y_train), np.nan)
    for fit_idx, valid_idx, fit_data, valid_data in fold_data:
        fold_model = create_gam(**best_parameters).fit(
            fit_data, y_train[fit_idx])
        oof_probability[valid_idx] = fold_model.predict_proba(valid_data)

    observed, predicted = calibration_curve(
        y_train, oof_probability, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(predicted, observed, "o-", color="black")
    ax.set_xlabel("Mean out-of-fold predicted probability")
    ax.set_ylabel("Observed dryline proportion")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.grid(visible=True)

    ax.set_title(f"{figtitle} Calibration Curve", fontsize=16)

    plt.savefig(f'{directory_path}/cyclone_dryline_gam_calibration_{filename}_{YEAR_CUTOFF}.png', format='png', dpi=100, bbox_inches='tight')

    oof_reference = np.full(len(y_train), np.nan)
    for fit_idx, valid_idx, _, _ in fold_data:
        oof_reference[valid_idx] = y_train[fit_idx].mean()

    brier = np.mean((oof_probability - y_train) ** 2)
    reference_brier = np.mean((oof_reference - y_train) ** 2)
    brier_skill = 1 - brier / reference_brier

    print(f"Out-of-fold Brier score: {brier:.4f}")
    print(f"Out-of-fold Brier skill: {brier_skill:.3f}")


    ################# Apply best parameters to the full training set
    final_scalar = StandardScaler() #initialize scaler for the full training set
    X_fit = X_train.copy()
    X_fit[:, 2:5] = final_scalar.fit_transform(X_fit[:, 2:5])
    final_gam_model = create_gam(**best_parameters).fit(X_fit, y_train)

    df['dryline_prediction'] = final_gam_model.predict_proba(X_fit)

    #deviance explained - model assessment
    fitted_probability = final_gam_model.predict_proba(X_fit)
    null_probability = np.full(len(y_train), y_train.mean())

    deviance_explained = 1 - (
        log_loss(y_train, fitted_probability, labels=[0, 1])
        / log_loss(y_train, null_probability, labels=[0, 1]))
    print(f"In-sample deviance explained: {deviance_explained:.3f}")
    #log-loss binomial deviance comparison


    ###################### evaluate relevance of individual predictors

    #adjusted odds ratio for a dryline given a cyclone   

    model, scaler = fit_analysis(X_train, best_parameters, y_train)
    estimates = effect_table(model, scaler, X_train, comparisons, predictors)
    obs_years = df['year'].to_numpy()
    n_bootstraps = 10000
    lower, upper = bootstrap_effects(X_train, y_train, obs_years, X_train, comparisons, predictors, best_parameters, n_bootstrap=n_bootstraps) #for testing

    #build df that holds this info for plotting
    adj_odds_ratio_results = pd.DataFrame({"adjusted_odds_ratio": estimates['Conditional_OR'], "ci_lower": lower['Conditional_OR'], "ci_upper": upper['Conditional_OR']})
    adj_odds_ratio_results.index.name = "comparison"

    summary = pd.DataFrame(index=estimates.index)
    summary["Probability A (mean %)"] = format_interval(estimates, lower, upper, "prob_a", scale=100, decimals=1)
    summary["Probability B (mean %)"] = format_interval(estimates, lower, upper, "prob_b", scale=100, decimals=1)
    summary["Difference B-A (pp)"] = format_interval(estimates, lower, upper, "difference", decimals=1)
    summary["Adjusted odds ratio B/A"] = format_interval(estimates, lower, upper, "Conditional_OR", decimals=2)

    print("\nAdjusted associations: estimate (95% bootstrap CI)")
    print(summary.to_string())
    print("\npp = percentage points; differences are B minus A.")
    print("Odds ratios compare B with A, holding other predictors constant.")
    #NOTE:
    # probabilities are mean predicted prob across all observations
    # adjusted odds ratio is the mean odds ratio across all observations (the odds ratio for each individual observation)
        #as currently constructed, this is consistent with conditional odds ratio. But if interaction terms are added, this may change

    term_variation = pd.Series({
        name: np.std(final_gam_model.partial_dependence(term=term, X=X_fit))
        for term, name in enumerate(term_names)
    }).sort_values(ascending=False)

    print("SD of fitted term contribution, in log-odds units:")
    print(term_variation)

    ############################## spatially-adjusted dryline probability map 

    #build a grid from all possible lee_within cyclone locations
    longitude = np.linspace(df['longitude'].min(), df['longitude'].max(), 100)
    latitude = np.linspace(df['latitude'].min(), df['latitude'].max(), 100)
    longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)
    locations = np.column_stack([longitude_grid.ravel(), latitude_grid.ravel()])

    grid_data = np.tile(X_fit[0], (len(locations), 1))
    grid_data[:, :2] = locations

    observed_spatial = final_gam_model.partial_dependence(term=0, X=X_fit)
    grid_spatial = final_gam_model.partial_dependence(term=0, X=grid_data)

    fitted_prob = np.clip(final_gam_model.predict_proba(X_fit), 1e-12, 1-1e-12)
    other_effects = logit(fitted_prob) - observed_spatial

    adjusted_prob = np.empty(len(locations))
    for start in range(0, len(locations), 100):
        stop = min(start + 100, len(locations))
        adjusted_prob[start:stop] = expit(grid_spatial[start:stop, None] + other_effects[None, :]).mean(axis=1)

    #this masks out locations outside the convex hull of observed centroids
    support = Delaunay(X_fit[:,:2]).find_simplex(locations) >= 0
    prob_map = np.ma.masked_where(~support.reshape(longitude_grid.shape), adjusted_prob.reshape(longitude_grid.shape))

    #this masks out locations without sufficient observational support -- must be more than X cyclones centroids within 200km radius
    observed_x, observed_y = to_meters(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    grid_x, grid_y = to_meters(locations[:,0], locations[:,1])
    tree = cKDTree(np.column_stack([observed_x, observed_y]))
    neighbors = tree.query_ball_point(np.column_stack([grid_x, grid_y]), r=200_000) #200km radius for observational support

    local_counts = np.array([len(indices) for indices in neighbors])

    well_supported = (support & (local_counts >= cyclone_count_threshold)) #25 for all spring, 10 for individual month
    prob_map = np.ma.masked_where(~well_supported.reshape(longitude_grid.shape), adjusted_prob.reshape(longitude_grid.shape))


    fig, ax = plt.subplots(figsize=(10, 10))
    m.drawcoastlines(linewidth=0.5,color='k', ax=ax)
    m.drawcountries(linewidth=0.5,color='k', ax=ax)
    m.drawstates(linewidth=0.25, color='k', ax=ax)
    m.drawmeridians(range(-180, 180, 5), labels = [0,0,0,1], linewidth=0.25, ax=ax)
    m.drawparallels(range(-10, 90, 5), labels = [1,0,0,0], linewidth=0.25, ax=ax)

    lon_grid, lat_grid = m(longitude_grid, latitude_grid)

    col = ['#fff5f0','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#a50f15','#67000d']
    custom_cmap = colors.LinearSegmentedColormap.from_list('custom_cmap', col)
    bounds = np.arange(0, 105, 5)
    norm = colors.BoundaryNorm(bounds, custom_cmap.N)

    surface = m.pcolormesh(lon_grid, lat_grid, prob_map*100, shading='auto', cmap=custom_cmap, norm=norm, ax=ax)
    cbar = plt.colorbar(surface, ax=ax, orientation='vertical', fraction=0.1, shrink=0.97, pad=0.025)

    # m.scatter(
    #     train["longitude"].to_numpy(), train["latitude"].to_numpy(), latlon=True,
    #     s=2, color="black", alpha=0.08, linewidths=0, ax=ax
    # )

    cbar.set_label('Adjusted Dryline Probability (%)', fontsize=16)
    ax.set_title(f"Cyclone Location and Adjusted\nDryline Probability, {figtitle}", fontsize=24)

    #add the domain bounds
    #define SGP region
    sgp = shapely.Polygon(((-105,35), (-105, 40), (-100,40), (-95, 40), (-95, 30), (-100, 30), (-105, 30)))
    sgpx,sgpy = sgp.exterior.xy
    sgpx,sgpy =m(sgpx,sgpy)

    #define SGP buffer zone
    # sgp_buffer = sgp.buffer(1.0)  # buffer of 1 degree
    # sgp_buffer_x, sgp_buffer_y = sgp_buffer.exterior.xy
    # sgp_buffer_x, sgp_buffer_y = m(sgp_buffer_x, sgp_buffer_y)

    #define the lee-cyclone region
    # lc = shapely.Polygon(((-105,32), (-105, 45), (-98, 45), (-98, 32), (-105,32)))
    # lc = shapely.Polygon(((-106,32), (-106, 41), (-109, 43), (-109, 47.5), (-106, 47.5), (-100, 47.5), (-98, 47.5), (-95, 40), (-95, 32), (-106,32)))
    lc = shapely.Polygon(((-106,32), (-106, 41), (-109, 43), (-109, 47.5), (-106, 47.5), (-100, 47.5), (-100, 47.5), (-98, 41), (-98, 32), (-106,32)))
    lcx, lcy = lc.exterior.xy
    lcx, lcy = m(lcx, lcy)

    m.plot(sgpx, sgpy, color='#0072B2', linewidth=7, ax=ax, zorder=13)
    m.plot(lcx, lcy, color='darkgreen', linewidth=5, ax=ax, zorder=40)

    #read in the terrain map for masking
    hght_array = xr.open_dataset("/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/ERA5_terrain_high.nc") #geopotential height
    eralons = hght_array.longitude.values
    eralats = hght_array.latitude.values
    #restrict the latitude and longitude
    reslon = []
    for j in range(len(eralons)):
        if eralons[j] >= 240 and eralons[j] <= 300:
            reslon.append(eralons[j])

    reslat = []
    for j in range(len(eralats)):
        if eralats[j] >= 20 and eralats[j] <= 55:
            reslat.append(eralats[j])

    hght_array = hght_array.sel(longitude=reslon)
    hght_array = hght_array.sel(latitude=reslat)

    eralons = hght_array.longitude.values
    eralats = hght_array.latitude.values
    x,y = np.meshgrid(eralons, eralats)
    x,y = m(x,y)

    hght = hght_array.z.values[0] / 9.80665 #convert geopotential height to height (m)

    terrain_mask = np.where(hght>=1750, 1.1, 0)
    m.contourf(x,y, terrain_mask, [1,2], colors=['gray'])

    plt.savefig(f'{directory_path}/adjusted_dryline_probability_map_{filename}_{YEAR_CUTOFF}.png', format='png', dpi=100, bbox_inches='tight')


    #save out the data necessary to recreate the calibration curve and map plots for a later multi-plot
    np.savez_compressed(f"{directory_path}/gam_plot_data_{filename}_{YEAR_CUTOFF}.npz",
                        cal_predicted=predicted,
                        cal_observed=observed,
                        oof_probability=oof_probability,
                        y_train=y_train,
                        brier=brier,
                        bss=brier_skill,
                        lon=longitude,
                        lat=latitude,
                        adjusted_prob=adjusted_prob.reshape(longitude_grid.shape),
                        well_supported=well_supported.reshape(longitude_grid.shape))


    return adj_odds_ratio_results

#TO DO: RUN THESE WITH DIFFERENT ENSO DEFINITION TO TEST SENSITIVITY
#TO DO: RUN THIS OVER ONLY 1979+ TO TEST SENSITIVITY

# startgamtime = datetime.utcnow()
# log_path = Path(__file__).resolve().parent / "allspring_gam_model.log"
# with log_path.open("w", buffering=1) as log_file, redirect_stdout(log_file):
#     run_gam(model_df) #all spring
# print("Allspring finished; Time to run: " + str(datetime.utcnow() - startgamtime))

run_model = True

if run_model == True:
    print("Running GAM")
    monthly_results = {}

    startgamtime = datetime.utcnow()
    log_path = Path(__file__).resolve().parent / f"{directory_path}/feb_gam_model_{YEAR_CUTOFF}.log"
    with log_path.open("w", buffering=1) as log_file, redirect_stdout(log_file):
        testing = model_df[model_df['month']==2]
        monthly_results["feb"] = run_gam(testing, month=2) #feb
    print("February finished; Time to run: " + str(datetime.utcnow() - startgamtime))

    startgamtime = datetime.utcnow()
    log_path = Path(__file__).resolve().parent / f"{directory_path}/march_gam_model_{YEAR_CUTOFF}.log"
    with log_path.open("w", buffering=1) as log_file, redirect_stdout(log_file):
        testing = model_df[model_df['month']==3]
        monthly_results["march"] = run_gam(testing, month=3) #march
    print("March finished; Time to run: " + str(datetime.utcnow() - startgamtime))

    startgamtime = datetime.utcnow()
    log_path = Path(__file__).resolve().parent / f"{directory_path}/april_gam_model_{YEAR_CUTOFF}.log"
    with log_path.open("w", buffering=1) as log_file, redirect_stdout(log_file):
        testing = model_df[model_df['month']==4]
        monthly_results["april"] = run_gam(testing, month=4) #april
    print("April finished; Time to run: " + str(datetime.utcnow() - startgamtime))

    startgamtime = datetime.utcnow()
    log_path = Path(__file__).resolve().parent / f"{directory_path}/may_gam_model_{YEAR_CUTOFF}.log"
    with log_path.open("w", buffering=1) as log_file, redirect_stdout(log_file):
        testing = model_df[model_df['month']==5]
        monthly_results["may"] = run_gam(testing, month=5) #may
    print("May finished; Time to run: " + str(datetime.utcnow() - startgamtime))

    startgamtime = datetime.utcnow()
    log_path = Path(__file__).resolve().parent / f"{directory_path}/june_gam_model_{YEAR_CUTOFF}.log"
    with log_path.open("w", buffering=1) as log_file, redirect_stdout(log_file):
        testing = model_df[model_df['month']==6]
        monthly_results["june"] = run_gam(testing, month=6) #june
    print("June finished; Time to run: " + str(datetime.utcnow() - startgamtime))

    #save out
    pd.to_pickle(monthly_results, f"{directory_path}/monthly_gam_results_{YEAR_CUTOFF}.pkl")

    print("GAM saved out; not yet plotted")

################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################ Make plots with GAM statistics
################################################################################################################ hard code these in from .log
################################################################################################################    so we don't have to re-run GAM all the time
################################################################################################################
################################################################################################################
#want to make a 6-panel plot by test which has the adjusted odds ratio with error bars for 95% CI for each month

plot_model = True

if plot_model ==  True:
    print("Plotting GAM results")

    if YEAR_CUTOFF == 1940:
        monthly_results = pd.read_pickle(f"{directory_path}/monthly_gam_results.pkl")
    else:
        monthly_results = pd.read_pickle(f"{directory_path}/monthly_gam_results_{YEAR_CUTOFF}.pkl")
    month_keys = ["feb", "march", "april", "may", "june"]

    def monthly_or_values(comparison):
        month_keys = ["feb", "march", "april", "may", "june"]

        rows = []

        for month_key in month_keys:
            results = monthly_results[month_key]

            matches = [label for label in results.index if label == comparison or label.startswith(comparison)]

            if len(matches) != 1:
                raise ValueError(f"Expected exactly one match for comparison '{comparison}' in month '{month_key}', but found {len(matches)}")
            row = results.loc[matches[0]]
            rows.append(row)

        estimate = np.array([row['adjusted_odds_ratio'] for row in rows])
        lower = np.array([row['ci_lower'] for row in rows])
        upper = np.array([row['ci_upper'] for row in rows])

        return estimate, [estimate - lower, upper - estimate]

    #organization: each test gets its own list, ordered from Feb-June, with separate lists for 95th-5th percentiles


    # la_minus_el = [2.77, 1.05, 1.01, 0.96, 0.99]
    # la_minus_el_95 = [5.39, 1.08, 1.04, 1.12, 1.02]
    # la_minus_el_5 = [1.41, 1.01, 0.98, 0.83, 0.95]

    # la_minus_nu = [2.01, 1.02, 0.99, 0.91, 0.99]
    # la_minus_nu_95 = [3.95, 1.07, 1.04, 1.13, 1.05]
    # la_minus_nu_5 = [1.06, 0.98, 0.94, 0.74, 0.94]

    # el_minus_nu = [0.73, 0.98, 0.98, 0.95, 1.01]
    # el_minus_nu_95 = [1.50, 1.02, 1.01, 1.11, 1.04]
    # el_minus_nu_5 = [0.37, 0.94, 0.95, 0.81, 0.97]

    # mslp_minus_sd = [0.62, 0.74, 0.87, 0.46, 0.50]
    # mslp_minus_sd_95 = [0.87, 0.86, 1.05, 0.60, 0.59]
    # mslp_minus_sd_5 = [0.42, 0.63, 0.71, 0.34, 0.41]

    mslp_sd = [6.50, 6.37, 5.62, 5.07, 4.43] #hPa

    # mslp_minus_10 = [0.48, 0.63, 0.78, 0.22, 0.21]
    # mslp_minus_10_95 = [0.81, 0.79, 1.09, 0.36, 0.30]
    # mslp_minus_10_5 = [0.26, 0.49, 0.55, 0.12, 0.13]

    # size_minus_sd = [1.15, 1.13, 1.23, 1.24, 1.10]
    # size_minus_sd_95 = [1.60, 1.34, 1.45, 1.49, 1.28]
    # size_minus_sd_5 = [0.85, 0.97, 1.05, 1.04, 0.94]

    size_sd = [243.28, 236.10, 227.53, 222.44, 220.46] #km^2

    # wind_minus_sd = [0.97, 0.93, 1.23, 0.92, 0.95]
    # wind_minus_sd_95 = [1.30, 1.08, 1.50, 1.13, 1.09]
    # wind_minus_sd_5 = [0.71, 0.80, 1.03, 0.76, 0.83]

    wind_sd = [8.83, 8.74, 8.04, 7.49, 6.80] #kts?

    #calculate the mean mslp/size/wind for each month
    months = np.arange(2, 7, 1) # February to June

    mslp_mean = []
    size_mean = []
    wind_mean = []

    for m in months:
        testing = model_df[model_df['month']==m]
        mslp_mean.append(testing['mslp'].mean())
        size_mean.append(testing['size'].mean())
        wind_mean.append(testing['max_wind'].mean())
        print(f"Month: {m}, MSLP Mean: {mslp_mean[-1]}, Size Mean: {size_mean[-1]}, Wind Mean: {wind_mean[-1]}")


    ######################################### make the plot
    fig, ((ax1,ax2), (ax3,ax4), (ax5,ax6), (ax7,ax8)) = plt.subplots(nrows=4,ncols=2,figsize=(24,24))
    axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]

    #panel a: La Nina - El Nino
    values, errors = monthly_or_values("La Nina - El Nino")
    ax1.errorbar(months, values, yerr=errors, fmt='o',
                markerfacecolor="royalblue", markeredgecolor="black",
                markersize=12, ecolor="royalblue", capsize=12,
                capthick=2.5, elinewidth=2.5)

    values, errors = monthly_or_values("La Nina - Neutral")
    ax3.errorbar(months, values, yerr=errors, fmt='o',
                markerfacecolor="royalblue", markeredgecolor="black",
                markersize=12, ecolor="royalblue", capsize=12,
                capthick=2.5, elinewidth=2.5)

    values, errors = monthly_or_values("El Nino - Neutral")
    ax5.errorbar(months, values, yerr=errors, fmt='o',
                markerfacecolor="royalblue", markeredgecolor="black",
                markersize=12, ecolor="royalblue", capsize=12,
                capthick=2.5, elinewidth=2.5)

    values, errors = monthly_or_values("Lee Cyclone - Not Lee Cyclone")
    ax7.errorbar(months, values, yerr=errors, fmt='o',
                markerfacecolor="royalblue", markeredgecolor="black",
                markersize=12, ecolor="royalblue", capsize=12,
                capthick=2.5, elinewidth=2.5)


    values, errors = monthly_or_values("mslp (mean): mslp (+1 SD")
    ax2.errorbar(months, values, yerr=errors, fmt='o',
                markerfacecolor="royalblue", markeredgecolor="black",
                markersize=12, ecolor="royalblue", capsize=12,
                capthick=2.5, elinewidth=2.5)

    values, errors = monthly_or_values("mslp (mean): mslp (+10 hPa")
    ax4.errorbar(months, values, yerr=errors, fmt='o',
                markerfacecolor="royalblue", markeredgecolor="black",
                markersize=12, ecolor="royalblue", capsize=12,
                capthick=2.5, elinewidth=2.5)


    values, errors = monthly_or_values("size (mean): size (+1 SD")
    ax6.errorbar(months, values, yerr=errors, fmt='o',
                markerfacecolor="royalblue", markeredgecolor="black",
                markersize=12, ecolor="royalblue", capsize=12,
                capthick=2.5, elinewidth=2.5)

    values, errors = monthly_or_values("max_wind (mean): max_wind (+1 SD")
    ax8.errorbar(months, values, yerr=errors, fmt='o',
                markerfacecolor="royalblue", markeredgecolor="black",
                markersize=12, ecolor="royalblue", capsize=12,
                capthick=2.5, elinewidth=2.5)

    # ax1.errorbar(months, la_minus_el, yerr=[np.array(la_minus_el)-np.array(la_minus_el_5), np.array(la_minus_el_95)-np.array(la_minus_el)], fmt='o', markerfacecolor="royalblue", markeredgecolor="black", markersize=12, ecolor="royalblue", capsize=12, capthick=2.5, elinewidth=2.5)
    # ax3.errorbar(months, la_minus_nu, yerr=[np.array(la_minus_nu)-np.array(la_minus_nu_5), np.array(la_minus_nu_95)-np.array(la_minus_nu)], fmt='o', markerfacecolor="royalblue", markeredgecolor="black", markersize=12, ecolor="royalblue", capsize=12, capthick=2.5, elinewidth=2.5)
    # ax5.errorbar(months, el_minus_nu, yerr=[np.array(el_minus_nu)-np.array(el_minus_nu_5), np.array(el_minus_nu_95)-np.array(el_minus_nu)], fmt='o', markerfacecolor="royalblue", markeredgecolor="black", markersize=12, ecolor="royalblue", capsize=12, capthick=2.5, elinewidth=2.5)

    # ax2.errorbar(months, mslp_minus_sd, yerr=[np.array(mslp_minus_sd)-np.array(mslp_minus_sd_5), np.array(mslp_minus_sd_95)-np.array(mslp_minus_sd)], fmt='o', markerfacecolor="royalblue", markeredgecolor="black", markersize=12, ecolor="royalblue", capsize=12, capthick=2.5, elinewidth=2.5)
    # # ax4.errorbar(months, mslp_minus_10, yerr=[np.array(mslp_minus_10)-np.array(mslp_minus_10_5), np.array(mslp_minus_10_95)-np.array(mslp_minus_10)], fmt='o', markerfacecolor="royalblue", markeredgecolor="black", markersize=12, ecolor="royalblue", capsize=12, capthick=2.5, elinewidth=2.5)
    # ax4.errorbar(months, size_minus_sd, yerr=[np.array(size_minus_sd)-np.array(size_minus_sd_5), np.array(size_minus_sd_95)-np.array(size_minus_sd)], fmt='o', markerfacecolor="royalblue", markeredgecolor="black", markersize=12, ecolor="royalblue", capsize=12, capthick=2.5, elinewidth=2.5)
    # ax6.errorbar(months, wind_minus_sd, yerr=[np.array(wind_minus_sd)-np.array(wind_minus_sd_5), np.array(wind_minus_sd_95)-np.array(wind_minus_sd)], fmt='o', markerfacecolor="royalblue", markeredgecolor="black", markersize=12, ecolor="royalblue", capsize=12, capthick=2.5, elinewidth=2.5)


    #add in the avg and SD info for each month of axes 2,4,6
    #MOVED TO TABLE
    # for i in range(len(mslp_mean)):
    #     ax2.text(months[i], 0.1, f"Avg: {mslp_mean[i]:.2f}\nSD: {mslp_sd[i]:.2f}", fontsize=12, ha='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    #     ax4.text(months[i], 20, f"Avg: {size_mean[i]:.2f}\nSD: {size_sd[i]:.2f}", fontsize=12, ha='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    #     ax6.text(months[i], 0.2, f"Avg: {wind_mean[i]:.2f}\nSD: {wind_sd[i]:.2f}", fontsize=12, ha='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))


    ax1.set_title('La Ni\u00f1a \N{MINUS SIGN} El Ni\u00f1o', fontsize=28)
    ax2.set_title('Avg. Min MSLP \N{MINUS SIGN}\n Avg. Min MSLP +1 Standard Deviation', fontsize=28)
    ax3.set_title('La Ni\u00f1a \N{MINUS SIGN} Neutral', fontsize=28)
    ax6.set_title(f'Avg. Area \N{MINUS SIGN}\n Avg. Area +1 Standard Deviation', fontsize=28)
    ax5.set_title('El Ni\u00f1o \N{MINUS SIGN} Neutral', fontsize=28)
    ax8.set_title(f'Avg. Max 10m Wind \N{MINUS SIGN}\n Avg. Max 10m Wind +1 Standard Deviation', fontsize=28)
    ax7.set_title('Lee Cyclone \N{MINUS SIGN} Non-Lee Cyclone', fontsize=28)
    ax4.set_title("Avg. Min MSLP \N{MINUS SIGN}\n Avg. Min MSLP +10 hPa", fontsize=28)


    ax7.set_xlabel('Month', fontsize=24)
    ax8.set_xlabel('Month', fontsize=24)

    ax1.set_ylabel('Adjusted Odds Ratio\n(log)', fontsize=24)
    ax3.set_ylabel('Adjusted Odds Ratio\n(log)', fontsize=24)
    ax5.set_ylabel('Adjusted Odds Ratio\n(log)', fontsize=24)
    ax7.set_ylabel('Adjusted Odds Ratio\n(log)', fontsize=24)

    month_labels = ["February", "March", "April", "May", "June"]
    month_positions = months
    for ax in axes:
        ax.set_xticks(month_positions)
        if ax != ax7 and ax != ax8:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(month_labels)
            ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.tick_params(axis='y', which='both', length=6, width=1.5)


        if ax == ax1 or ax == ax3 or ax == ax5 or ax == ax7:
            ax.set_yscale('log')
            ax.set_ylim(0.3, 6)
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.set_yticks([0.5, 1, 5])
        elif ax == ax4:
            ax.set_yscale('log')
            ax.set_ylim(0.035, 2)
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.set_yticks([.05, 0.1, 0.5, 1])
        else:
            ax.set_yscale('log')
            ax.set_ylim(0.2, 2)
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.set_yticks([0.5, 1, 2])
            ax.yaxis.set_minor_formatter(NullFormatter())

        # if ax == ax4:
        #     ax.set_yscale('log')
        #     ax.set_ylim(0.5, 60)
        #     ax.yaxis.set_major_formatter(ScalarFormatter())
        #     ax.set_yticks([0.5, 1, 5, 10, 50])
        # elif ax==ax1:
        #     ax.set_yscale('log')
        #     ax.set_ylim(0.5, 10)
        #     ax.yaxis.set_major_formatter(ScalarFormatter())
        #     ax.set_yticks([0.5, 1, 5, 10])
        # else:
        #     ax.set_yscale('log')
        #     ax.set_ylim(0.1, 5)
        #     ax.yaxis.set_major_formatter(ScalarFormatter())
        #     ax.set_yticks([0.1, 0.5, 1, 5])


        ax.axhline(1, color="black", linestyle='--', linewidth=1.5)

        ax.grid(visible=True, axis='y', which='both')

    panel_labels = ["a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)"]
    for i, ax in enumerate(axes):
        if ax != ax8:
            ax.text(0.05, 1.125, panel_labels[i], transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right', bbox=dict(facecolor='white', edgecolor='black'))
        else:
            ax.text(0.05, 1.2, panel_labels[i], transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right', bbox=dict(facecolor='white', edgecolor='black'))

    plt.subplots_adjust(wspace=0.1, hspace=0.25)
    plt.suptitle("Dryline Adjusted Odds Ratios from\n Monthly Cyclone Logistic Generalized Additive Model", fontsize=34, y=0.965)
    plt.savefig(f'{directory_path}/monthly_gam_adjusted_odds_ratios_{YEAR_CUTOFF}.png', format='png', dpi=150, bbox_inches='tight') 


    llcrnrlat=27.5 
    llcrnrlon=360-110 
    urcrnrlat=50 
    urcrnrlon=360-90
    lat_0=35
    lon_0=360-98
    latin1=25
    latin2=25
    m = Basemap(projection='lcc', llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
                urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon,
                lat_1=latin1, lat_2=latin2, lat_0=lat_0, lon_0=lon_0,
                resolution='h', area_thresh=10000, fix_aspect=True)

    #make the multi-panel adjusted probability and calibration plots
    month_keys = ["feb", "march", "april", "may", "june"]
    month_titles = ["February", "March", "April", "May", "June"]
    if YEAR_CUTOFF == 1940:
        plot_data = {k: np.load(f"{directory_path}/gam_plot_data_{k}.npz") for k in month_keys}
    else:
        plot_data = {k: np.load(f"{directory_path}/gam_plot_data_{k}_{YEAR_CUTOFF}.npz") for k in month_keys}

    #make the plots -- first start with map
    fig, ((ax1,ax2), (ax3,ax4), (ax5,ax6)) = plt.subplots(nrows=3,ncols=2,figsize=(10,20))
    axes = [ax1,ax2,ax3,ax4,ax5]
    labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']

    for k, ax in zip(month_keys, axes):
        d = plot_data[k]

        lon_g, lat_g = np.meshgrid(d['lon'], d['lat'])
        prob_map = np.ma.masked_where(~d['well_supported'], d['adjusted_prob'])

        x, y = m(lon_g, lat_g)

        m.drawcoastlines(linewidth=0.5,color='k', ax=ax)
        m.drawcountries(linewidth=0.5,color='k', ax=ax)
        m.drawstates(linewidth=0.25, color='k', ax=ax)
        m.drawmeridians(range(-180, 180, 5), labels = [0,0,0,1], linewidth=0.25, ax=ax)
        m.drawparallels(range(-10, 90, 5), labels = [1,0,0,0], linewidth=0.25, ax=ax)
    
        col = ['#fff5f0','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#a50f15','#67000d']
        custom_cmap = colors.LinearSegmentedColormap.from_list('custom_cmap', col)
        bounds = np.arange(0, 105, 5)
        norm = colors.BoundaryNorm(bounds, custom_cmap.N)
    
        surface = m.pcolormesh(x, y, prob_map*100, shading='auto', cmap=custom_cmap, norm=norm, ax=ax)
        print(f"{month_titles[month_keys.index(k)]}: {np.nanmax(prob_map*100)}")
        # cbar = plt.colorbar(surface, ax=ax, orientation='vertical', fraction=0.1, shrink=0.97, pad=0.025)
    
        # m.scatter(
        #     train["longitude"].to_numpy(), train["latitude"].to_numpy(), latlon=True,
        #     s=2, color="black", alpha=0.08, linewidths=0, ax=ax
        # )
    
        # cbar.set_label('Adjusted Dryline Probability (%)', fontsize=16)
        ax.set_title(f"{month_titles[month_keys.index(k)]}", fontsize=24)
    
        #add the domain bounds
        #define SGP region
        sgp = shapely.Polygon(((-105,35), (-105, 40), (-100,40), (-95, 40), (-95, 30), (-100, 30), (-105, 30)))
        sgpx,sgpy = sgp.exterior.xy
        sgpx,sgpy =m(sgpx,sgpy)

        #define SGP buffer zone
        sgp_buffer = sgp.buffer(1.0)  # buffer of 1 degree
        sgp_buffer_x, sgp_buffer_y = sgp_buffer.exterior.xy
        sgp_buffer_x, sgp_buffer_y = m(sgp_buffer_x, sgp_buffer_y)
    
        #define the lee-cyclone region
        # lc = shapely.Polygon(((-105,32), (-105, 45), (-98, 45), (-98, 32), (-105,32)))
        # lc = shapely.Polygon(((-106,32), (-106, 41), (-109, 43), (-109, 47.5), (-106, 47.5), (-100, 47.5), (-98, 47.5), (-95, 40), (-95, 32), (-106,32)))
        lc = shapely.Polygon(((-106,32), (-106, 41), (-109, 43), (-109, 47.5), (-106, 47.5), (-100, 47.5), (-100, 47.5), (-98, 41), (-98, 32), (-106,32)))
        lcx, lcy = lc.exterior.xy
        lcx, lcy = m(lcx, lcy)
    
        m.plot(sgpx, sgpy, color='#0072B2', linewidth=7, ax=ax, zorder=13)
        m.plot(lcx, lcy, color='darkgreen', linewidth=4, ax=ax, zorder=40)
    
        #read in the terrain map for masking
        hght_array = xr.open_dataset("/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/ERA5_terrain_high.nc") #geopotential height
        eralons = hght_array.longitude.values
        eralats = hght_array.latitude.values
        #restrict the latitude and longitude
        reslon = []
        for j in range(len(eralons)):
            if eralons[j] >= 240 and eralons[j] <= 300:
                reslon.append(eralons[j])
    
        reslat = []
        for j in range(len(eralats)):
            if eralats[j] >= 20 and eralats[j] <= 55:
                reslat.append(eralats[j])
    
        hght_array = hght_array.sel(longitude=reslon)
        hght_array = hght_array.sel(latitude=reslat)
    
        eralons = hght_array.longitude.values
        eralats = hght_array.latitude.values
        x,y = np.meshgrid(eralons, eralats)
        x,y = m(x,y)
    
        hght = hght_array.z.values[0] / 9.80665 #convert geopotential height to height (m)
    
        terrain_mask = np.where(hght>=1750, 1.1, 0)
        m.contourf(x,y, terrain_mask, [1,2], colors=['gray'], ax=ax)

        ax.text(0.095, 1.075, labels[month_keys.index(k)], transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right', bbox=dict(facecolor='white', edgecolor='black'))


    ax6.set_axis_off()
    domain_handles = [
        Line2D([], [], color='#0072B2', linewidth=7,
               label='SGP\ndomain'),
        Line2D([], [], color='darkgreen', linewidth=4,
               label='Lee cyclone\ndomain'),
    ]

    ax6.legend(
        handles=domain_handles,
        loc='center left',
        bbox_to_anchor=(0.02, 0.55),
        bbox_transform=ax6.transAxes,
        frameon=True,
        edgecolor='black',
        framealpha=1.0,
        fontsize=16,
        handlelength=1.5,
        labelspacing=1.5,
        borderaxespad=0,
    )

    cbar_ax = ax6.inset_axes([0.60, 0.15, 0.08, 0.8])
    cbar = fig.colorbar(
        surface,
        cax=cbar_ax,
        orientation='vertical',
        ticks=np.arange(0, 101, 20),
    )
    cbar.ax.set_xlabel(
        'Adjusted Dryline\nProbability (%)',
        labelpad=10,
        fontsize=16,
    )
    # cbar_ax = ax6.inset_axes([0.42, 0.15, 0.08, 0.8])
    # cbar = fig.colorbar(surface, cax=cbar_ax, orientation='vertical', ticks=np.arange(0, 101, 20))
    # cbar.ax.set_xlabel('Adjusted Dryline\nProbability (%)', labelpad=10, fontsize=16)

    plt.suptitle('Adjusted Dryline Probability\nby Cyclone Location', fontsize=28, y=0.945)
    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    plt.savefig(f'{directory_path}/adjusted_dryline_probability_map_{YEAR_CUTOFF}.png', format='png', dpi=100, bbox_inches='tight')


    #plot the calibration curve
    fig, ((ax1,ax2), (ax3,ax4), (ax5,ax6)) = plt.subplots(nrows=3,ncols=2,figsize=(12,12))
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']

    for k, ax in zip(month_keys, axes):
        d = plot_data[k]

        observed, predicted = d['cal_predicted'], d['cal_observed']

        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.plot(predicted, observed, "o-", color="black")
        if ax == ax5 or ax == ax4:
            ax.set_xlabel("Mean out-of-fold predicted probability")
        if ax == ax1 or ax == ax3 or ax == ax5:
            ax.set_ylabel("Observed dryline proportion")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.grid(visible=True)

        ax.text(0.095, 0.96, labels[month_keys.index(k)], transform=ax.transAxes, fontsize=15, fontweight='bold', va='top', ha='right', bbox=dict(facecolor='white', edgecolor='black'))


        ax.set_title(f"{month_titles[month_keys.index(k)]}", fontsize=24)

    ax6.set_axis_off()
    plt.suptitle("GAM Calibration Curve by Month", fontsize=28, y=0.945)
    plt.savefig(f'{directory_path}/cyclone_dryline_gam_calibration_{YEAR_CUTOFF}.png', format='png', dpi=100, bbox_inches='tight')



print('Time to run: ' + str(datetime.utcnow() - starttime))