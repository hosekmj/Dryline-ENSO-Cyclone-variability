#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#use this script to run CyTRACK
#command: sh run_CyTRACK.sh

#the number after "-n" represents the number of processors used
#Was 12 -- can do this when only inputing one config file, but if running with modifying config files, only works on one
#probably due to the different processess accessing the half-written file at different times

#run in cytrack environment
#bigbang4 

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mpirun -n 1 python run_CyTRACK.py -pf CyTRACK_config.cfg

    
    
