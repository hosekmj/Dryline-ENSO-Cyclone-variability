#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#use this script to run CyTRACK
#command: sh run_CyTRACK.sh

#the number after "-n" represents the number of processors used
#only tested with 1

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mpirun -n 1 python run_CyTRACK.py -pf CyTRACK_config.cfg

    
    
