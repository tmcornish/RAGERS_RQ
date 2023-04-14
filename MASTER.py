#####################################################################################################
# Master script for running different stages of the RAGERS analysis.
#
#####################################################################################################

# Import necessary packages/modules
import os, sys
import my_functions as mf



####################################
#### RELEVANT PATHS & VARIABLES ####
####################################

PATH_RAGERS = '/home/cornisht/RAGERS/'
PATH_SCRIPTS = PATH_RAGERS + '/Scripts/Analysis_pipeline/'
PATH_CATS = PATH_RAGERS + '/Catalogues/'
PATH_PLOTS = PATH_RAGERS + '/Plots/'
PATH_DATA = PATH_RAGERS + '/Data/'


##################
#### SETTINGS ####
##################

#toggle `switches' for determining which scripts to run
rq_sample = True			#select the sample of radio-quiet massive galaxies
number_counts = True		#construct number counts

##################

settings = [rq_sample, number_counts]
proc_names = ['Selecting RQ sample', 'Constructing number counts']
run_str = [
	f'./Select_radio_quiet_sample.sh {PATH_RAGERS} {PATH_CATS} {PATH_DATA} {PATH_PLOTS}',
	f'python Submm_number_counts.py {PATH_RAGERS} {PATH_CATS} {PATH_PLOTS}']

print(mf.colour_string(mf.string_important('PROCESSES TO RUN')+'\n', 'olive'))
setting_str = []
for se, pn in zip(settings, proc_names):
	if se:
		setting_str.append(pn)
print('\n'.join(setting_str)+'\n')




#########################
#### RUNNING SCRIPTS ####
#########################

for se, pn, rs in zip(settings, proc_names, run_str):
	if se:
		print(mf.colour_string(mf.string_important(pn.upper())+'\n', 'orange')+'\n')
		os.system(rs)