cytrack_modified_source_code: source code for the CyTRACK package, with some minor modifications to add a maximum westward-propagation threshold and other tweaks to get it running on my machine.

CyTRACK_config.cfg: config file used in the running of CyTRACK
run_CyTRACK.py: the script which runs the cyclone tracking, run using the shell script run_CyTRACK.sh

create_cytrack_mslp_climo.py: create an 81-year centered 31-day climatology for use in CyTRACK

build_cyclone_database.py: Take the CyTRACK output and organize it into a by-timestep database
simplify_cyclone_database.py: Create a by-cyclone database from the by-timestep database built above
