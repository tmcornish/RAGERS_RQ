#####################################################################################################
# Master script for running different stages of the RAGERS analysis.
#
#####################################################################################################

# Import necessary packages/modules
import os, sys
from . import rq_general as rqg


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
	'./Select_radio_quiet_sample.sh',
	'python Submm_number_counts.py']

print(rqg.colour_string(rqg.string_important('PROCESSES TO RUN')+'\n', 'cyan'))
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
		print(rqg.colour_string(rqg.string_important(pn.upper())+'\n', 'orange')+'\n')
		os.system(rs)