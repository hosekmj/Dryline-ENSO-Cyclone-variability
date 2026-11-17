# script that will run the sensitivity analysis.

#gpd_plot

#can run from shell if I so choose

import os
import numpy as np
import pandas as pd
import subprocess
import sys
from datetime import datetime, timezone, timedelta
import xarray as xr
import glob

import yaml

from pathlib import Path

startcodetime = datetime.now(timezone.utc)

run_log_dir = (
    Path(__file__).resolve().parent
    / "logs"
    / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
)
run_log_dir.mkdir(parents=True, exist_ok=False)

def run_logged(cmd, script_name, terrain_mask, enso_type, year=None, domain_type=None):
    if year is not None:
        log_dir = run_log_dir / f"t={terrain_mask}" / enso_type / f"year={year}"
    else:
        log_dir = run_log_dir / f"t={terrain_mask}" / enso_type

    if domain_type is not None:
        log_dir = log_dir / f"domain={domain_type}"

    # log_dir = run_log_dir / f"t={terrain_mask}" / enso_type
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{script_name}.log"

    print(f"Running {script_name}; log: {log_path}", flush=True)

    with log_path.open("x", encoding="utf-8") as log_file:
        print(f"Command: {cmd}", file=log_file, flush=True)
        subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

def cytrack_terrain_sensitivity(terrain_mask, enso_type='op_enso'):
    #terrain mask: int: 1500, 1750, 2000
    #enso_type: str: 'op_enso', '3mo_enso', '1mo_enso

    
    expected_pickle = f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/all_cyclones_simplified_t={terrain_mask}.pkl"
    if not os.path.exists(expected_pickle):
        print(f"Expected pickle not found for terrain mask {terrain_mask}, running preprocessing.")
    
        #build the cyclone database, if not already
        cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/build_cyclone_database.py {terrain_mask}'
        run_logged(cmd, "build_cyclone_database", terrain_mask, enso_type)
        # subprocess.run(cmd, shell=True, check=True)

        #simplify the cyclone database
        cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/simplify_cyclone_database.py {terrain_mask}'
        run_logged(cmd, "simplify_cyclone_database", terrain_mask, enso_type)
        # subprocess.run(cmd, shell=True, check=True)

    #cyclone heatmap
    cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/cyclone_summary_plots.py {terrain_mask}'
    run_logged(cmd, "cyclone_summary_plots", terrain_mask, enso_type)
    # subprocess.run(cmd, shell=True, check=True)


    #timeseries
    cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/timeseries.py {terrain_mask} {enso_type}'
    run_logged(cmd, "timeseries", terrain_mask, enso_type)
    # subprocess.run(cmd, shell=True, check=True)

    #summary_analysis
    cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/summary_analysis.py {terrain_mask} {enso_type} 1940'
    run_logged(cmd, "summary_analysis", terrain_mask, enso_type)
    # subprocess.run(cmd, shell=True, check=True)


    #svr_summary analysis
    if enso_type=="op_enso":
        enso_type_svr = "enso_op"
    elif enso_type=="3mo_enso":
        enso_type_svr = "enso_3mo"
    elif enso_type=="1mo_enso":
        enso_type_svr = "enso_1mo"
    else:
        raise ValueError(f"Unsupported ENSO type: {enso_type}")
    
    cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/svr_summary_analysis.py {terrain_mask} {enso_type_svr}'
    run_logged(cmd, "svr_summary_analysis", terrain_mask, enso_type)
    # subprocess.run(cmd, shell=True, check=True)

    #gam_model
    cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/gam_model.py {terrain_mask} {enso_type} 1940'
    run_logged(cmd, "gam_model", terrain_mask, enso_type)
    # subprocess.run(cmd, shell=True, check=True)
    

    #cyclone composite -- will need ENSO read_in too
    cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/cyclone_composite.py {terrain_mask} {enso_type} 1940'
    run_logged(cmd, "cyclone_composite", terrain_mask, enso_type)
    # subprocess.run(cmd, shell=True, check=True)


    return


def enso_sensitivity(enso_type='op_enso'):
    #enso_type: str: 'op_enso', '3mo_enso', '1mo_enso'

    terrain_mask = 1750

    #timeseries
    # cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/timeseries.py {terrain_mask} {enso_type}'
    # run_logged(cmd, "timeseries", terrain_mask, enso_type)
    # subprocess.run(cmd, shell=True, check=True)

    #summary_analysis
    cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/summary_analysis.py {terrain_mask} {enso_type} 1940'
    run_logged(cmd, "summary_analysis", terrain_mask, enso_type)
    # subprocess.run(cmd, shell=True, check=True)


    #svr_summary analysis
    if enso_type=="op_enso":
        enso_type_svr = "enso_op"
    elif enso_type=="3mo_enso":
        enso_type_svr = "enso_3mo"
    elif enso_type=="1mo_enso":
        enso_type_svr = "enso_1mo"
    else:
        raise ValueError(f"Unsupported ENSO type: {enso_type}")
    
    cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/svr_summary_analysis.py {terrain_mask} {enso_type_svr}'
    run_logged(cmd, "svr_summary_analysis", terrain_mask, enso_type)
    # subprocess.run(cmd, shell=True, check=True)

    #gam_model
    cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/gam_model.py {terrain_mask} {enso_type} 1940'
    run_logged(cmd, "gam_model", terrain_mask, enso_type)
    # subprocess.run(cmd, shell=True, check=True)

    #cyclone composite -- will need ENSO read_in too
    cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/cyclone_composite.py {terrain_mask} {enso_type} 1940'
    run_logged(cmd, "cyclone_composite", terrain_mask, enso_type)
    # subprocess.run(cmd, shell=True, check=True)

    #dryline enso composite
    cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/dryline_composite.py {terrain_mask} {enso_type} 1940'
    run_logged(cmd, "dryline_composite", terrain_mask, enso_type)
    # subprocess.run(cmd, shell=True, check=True)

    return 



def year_sensitivity(enso_type='op_enso', terrain_mask=1750, year_cutoff=1979):
    #enso_type: str: 'op_enso', '3mo_enso', '1mo_enso'



    
    #summary_analysis
    cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/summary_analysis.py {terrain_mask} {enso_type} {year_cutoff}'
    run_logged(cmd, "summary_analysis", terrain_mask, enso_type, year=year_cutoff)
    # subprocess.run(cmd, shell=True, check=True)

    #gam_model
    cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/gam_model.py {terrain_mask} {enso_type} {year_cutoff}'
    run_logged(cmd, "gam_model", terrain_mask, enso_type, year=year_cutoff)
    # subprocess.run(cmd, shell=True, check=True)
    
    #cyclone composite -- will need ENSO read_in too
    cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/cyclone_composite.py {terrain_mask} {enso_type} {year_cutoff}'
    run_logged(cmd, "cyclone_composite", terrain_mask, enso_type, year=year_cutoff)
    # subprocess.run(cmd, shell=True, check=True)

    #dryline enso composite
    cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/dryline_composite.py {terrain_mask} {enso_type} {year_cutoff}'
    run_logged(cmd, "dryline_composite", terrain_mask, enso_type, year=year_cutoff)
    # subprocess.run(cmd, shell=True, check=True)

    return 

def domain_sensitivity(enso_type='op_enso', terrain_mask=1750, domain_type='no_buffer'):
    #terrain mask: int: 1500, 1750, 2000
    #enso_type: str: 'op_enso', '3mo_enso', '1mo_enso

    
    expected_pickle = f"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/all_cyclones_simplified_t={terrain_mask}_{domain_type}.pkl"
    if not os.path.exists(expected_pickle):
        print(f"Expected pickle not found for terrain mask {domain_type}, running preprocessing.")
    
        #build the cyclone database, if not already
        cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/build_cyclone_database.py {terrain_mask} {domain_type}'
        run_logged(cmd, "build_cyclone_database", terrain_mask, enso_type, domain_type=domain_type)
        # subprocess.run(cmd, shell=True, check=True)

        #simplify the cyclone database
        cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/revision_1/simplify_cyclone_database.py {terrain_mask} {domain_type}'
        run_logged(cmd, "simplify_cyclone_database", terrain_mask, enso_type, domain_type=domain_type)
        # subprocess.run(cmd, shell=True, check=True)

    #timeseries
    cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/timeseries.py {terrain_mask} {enso_type} {domain_type}'
    run_logged(cmd, "timeseries", terrain_mask, enso_type, domain_type=domain_type)
    # subprocess.run(cmd, shell=True, check=True)

    #summary_analysis
    cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/summary_analysis.py {terrain_mask} {enso_type} 1940 {domain_type}'
    run_logged(cmd, "summary_analysis", terrain_mask, enso_type, domain_type=domain_type)
    # subprocess.run(cmd, shell=True, check=True)


    #gam_model
    cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/gam_model.py {terrain_mask} {enso_type} 1940 {domain_type}'
    run_logged(cmd, "gam_model", terrain_mask, enso_type, domain_type=domain_type)
    # subprocess.run(cmd, shell=True, check=True)
    
    
    #cyclone composite -- will need ENSO read_in too
    cmd = f'conda run -n gpd_plot python -W ignore /home/michael.hosek/Documents/ERA5_drylines/enso_paper/revision_1/cyclone_composite.py {terrain_mask} {enso_type} 1940 {domain_type}'
    run_logged(cmd, "cyclone_composite", terrain_mask, enso_type, domain_type=domain_type)
    # subprocess.run(cmd, shell=True, check=True)


    return

# run sensitivity analysis on cyclone terrain masking
cytrack_terrain_sensitivity(terrain_mask=1750, enso_type='op_enso') #what I'm using

cytrack_terrain_sensitivity(terrain_mask=1500, enso_type='op_enso')
cytrack_terrain_sensitivity(terrain_mask=2000, enso_type='op_enso')

enso_sensitivity(enso_type='op_enso') #baseline

enso_sensitivity(enso_type='3mo_enso')
enso_sensitivity(enso_type='1mo_enso')

year_sensitivity(enso_type='op_enso', terrain_mask=1750, year_cutoff=1995)

domain_sensitivity(enso_type='op_enso', terrain_mask=1750, domain_type='no_buffer')
domain_sensitivity(enso_type='op_enso', terrain_mask=1750, domain_type='only_centroid')

print('Time to run full: ' + str(datetime.now(timezone.utc) - startcodetime))