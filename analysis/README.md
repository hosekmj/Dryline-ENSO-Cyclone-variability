Folder containing analysis and sensitivity scripts.

run_sensitivity.py: This script runs the other analysis scripts, and enables the user to easily modify what the data the analysis is run over (e.g., changing the CyTRACK terrain threshold, changing the earliest analysis year, modifying the ENSO definition, modifying the definition of SGP cyclone). Some scripts are not included, as they are not relevant to this sensitivity test process.

associate_drylines_severe_enso.py: Associate gridded severe storm reports and warnings to drylines.
example_plot.py: generate an example plot with FrontFinder AI front predictions and identified cyclones overlaid with ERA5 reanalysis.
determine_ENSO.py: build the ENSO dataset for use in the rest of analysis.

dryline_envi_1940_2020.pkl: summary pickle file of the FrontFinder AI dryline predictions

cyclone_composite.py, dryline_composite.py: generate composite anomalies of cyclone and dryline frequency by ENSO phase.
domain_plot.py: generate figure for the domains

gam_model.py: create and analyze a generalized additive model of dryline probability for all SGP cyclones.
summary_analysis.py: create figures for the primary dryline--ENSO--cyclone comparison
svr_summary_figures.py: create figures pertaining to the severe analysis for the dryline--ENSO--cyclone comparison

timeseries.py: create a timeseries of monthly and seasonal frequency of drylines and cyclones.

