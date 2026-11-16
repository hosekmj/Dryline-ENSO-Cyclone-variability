#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#run this using run_CyTRACK.sh
#bigbang4, cytrack

import numpy as np
import cytrack
import configobj
from configobj import ConfigObj

#tahnks stackoverflow: https://stackoverflow.com/questions/33280545/how-to-keep-the-quotes-string-with-configobj-in-python
class MyConfigObj(ConfigObj):
    def __init__(self, *args, **kwargs):
        ConfigObj.__init__(self, *args, **kwargs)
    def _unquote(self, value):
        return value
    def _quote(self, value, multiline=True):
        return value




args = cytrack.read_args()
if args.cytrack_help:
    cytrack.help()
elif args.get_template:
    cytrack.get_cytrack_inputs_template()
else:
    #run multiple times, modifying the input file each time
    #end in early July, to buffer the end of June. Start in late Jan, to buffer the start of Feb.
    startmonth = 1 
    startday = 20
    endmonth = 7
    endday = 10
    startyear = np.arange(1940, 2021, 1)
    endyear = np.arange(1940, 2021, 1)

    # startyear = [1945]
    # endyear = [1945]
    
    starthour = 0
    endhour = 18

    terrain_hght = 2000 #1000, 1250, 1500, 1750, 2000
    
    
    #for running multiple years
    for i in range(len(startyear)):
        # conf = configobj.ConfigObj('CyTRACK_config.cfg')
        conf = MyConfigObj('CyTRACK_config.cfg')
        
        conf['verbose'] = "True"
        conf['cyclone_type'] = '"EC"'
        conf['source'] = '"ERA5"'
        conf["path_data_source"]='"/raid/mhosek/cytrack/ERA5_data/"'
        conf['era_file_prefix'] = '"mod_uvmslp_ERA5"' 
        conf['era_date_file_name'] = '"yyyymmdd_hh"'
        conf['checking_upper_levels_parameters'] = "False"
        conf['max_dist'] = 500
        conf['model_res'] = 28
        conf['search_region'] = '"NA"'
        conf['search_limits'] = [-112.5,25,-80,52.5] #[lonmin,latmin,lonmax,latmax] 
        ################################### these are the ones that will change
        conf['begin_year'] = f'"{startyear[i]}"'
        conf['begin_month'] = f'"{startmonth:02d}"'
        conf['begin_day'] = f'"{startday:02d}"'
        conf['begin_hour'] = f'"{starthour:02d}"'
        conf['end_year'] = f'"{endyear[i]}"'
        conf['end_month'] = f'"{endmonth:02d}"'
        conf['end_day'] = f'"{endday:02d}"'
        conf['end_hour'] = f'"{endhour:02d}"'
        #################################
        conf['path_out'] = f'"/home/michael.hosek/Documents/ERA5_drylines/enso_paper/CyTRACK/"'
        conf['tmp_dir'] = '"./"'
        conf['remove_tmp_dir'] = "True"
        conf['max_wind_speed_threshold'] = None
        conf['outer_wind_speed_threshold'] = None
        conf['filter_center_threshold'] = 1000 #was 1000
        conf['dist_threshold'] = 1000 #was 1000
        conf['critical_outer_radius'] = 100
        conf['dr_res'] = 100
        conf['d_ang'] = 10
        conf['rout'] = 2000
        conf['terrain_filter'] = terrain_hght
        conf['intensity_threshold'] = None
        conf['dt_lifetime'] = 24
        conf['vorticity_threshold'] = 1.45e-5
        conf['min_slp_threshold'] = 1015
        conf['great_circle_distance'] = 6.5
        conf['dmslp_great_circle_distance'] = 200
        conf['radius_for_msw'] = 250
        conf['prev_days'] = 14 #was 14 -- NEED TO CHECK THIS BECAUSE SHOULD USE FULL ANOMALY
        conf['mslp_anomaly_threshold'] = -2.5
        conf['minimum_distance_travelled'] = 500
        conf['max_westward_distance'] = 400 #km, new parameter

        conf.write()
    
    
        cytrack.get_cytrack_main(args.parameterfile)
    '''
    
    # for running a single year with multiprocess
    conf = MyConfigObj('CyTRACK_config.cfg')
    
    conf['verbose'] = "True"
    conf['cyclone_type'] = '"EC"'
    conf['source'] = '"ERA5"'
    conf["path_data_source"]='"/raid/student/mhosek/cytrack/ERA5_data"'
    conf['era_file_prefix'] = '"uvmslp_ERA5"'
    conf['era_date_file_name'] = '"yyyymmdd_hh"'
    conf['checking_upper_levels_parameters'] = "False"
    conf['max_dist'] = 500
    conf['model_res'] = 28
    conf['search_region'] = '"NA"'
    conf['search_limits'] = [-106,30,-80,50] #[lonmin,latmin,lonmax,latmax] 
    ################################### these are the ones that will change
    conf['begin_year'] = f'"{startyear}"'
    conf['begin_month'] = f'"{startmonth:02d}"'
    conf['begin_day'] = f'"{startday:02d}"'
    conf['begin_hour'] = f'"{starthour:02d}"'
    conf['end_year'] = f'"{endyear}"'
    conf['end_month'] = f'"{endmonth:02d}"'
    conf['end_day'] = f'"{endday:02d}"'
    conf['end_hour'] = f'"{endhour:02d}"'
    #################################
    conf['path_out'] = '"./"'
    conf['tmp_dir'] = '"./"'
    conf['remove_tmp_dir'] = "True"
    conf['max_wind_speed_threshold'] = None
    conf['outer_wind_speed_threshold'] = None
    conf['filter_center_threshold'] = 1000
    conf['dist_threshold'] = 1000
    conf['critical_outer_radius'] = 100
    conf['dr_res'] = 100
    conf['d_ang'] = 10
    conf['rout'] = 2000
    conf['terrain_filter'] = 1750
    conf['intensity_threshold'] = None
    conf['dt_lifetime'] = 24
    conf['vorticity_threshold'] = 1.45e-5
    conf['min_slp_threshold'] = 1015
    conf['great_circle_distance'] = 6.5
    conf['dmslp_great_circle_distance'] = 200
    conf['radius_for_msw'] = 250
    conf['prev_days'] = 14 #was 14
    conf['mslp_anomaly_threshold'] = -2.5
    conf['minimum_distance_travelled'] = 500

    conf.write()


    cytrack.get_cytrack_main(args.parameterfile)
    '''

#%% TO RUN WITH MULTIPROCESSING:
    
    
'''
args = cytrack.read_args()
if args.cytrack_help:
    cytrack.help()
elif args.get_template:
    cytrack.get_cytrack_inputs_template()
else:
    cytrack.get_cytrack_main(args.parameterfile)
'''