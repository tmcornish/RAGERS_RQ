#####################################################################################################
# Master script for running different stages of the RAGERS analysis.
#
#####################################################################################################

# Import necessary packages/modules
import os, sys
import general as gen


##################
#### SETTINGS ####
##################

#toggle `switches' for determining which scripts to run
recreate_S19 = False		#recreate the results (completeness, number counts) from Simpson+19
rq_sample = False			#select the sample of radio-quiet massive galaxies
number_counts = True		#construct number counts

##################

settings = [recreate_S19, rq_sample, number_counts]
proc_names = [
	'Recreating Simpson+19 results', 
	'Selecting RQ sample', 
	'Constructing number counts']
run_str = [
	'python Recreate_S19_number_counts.py',
	'python Select_radio_quiet_sample.py',
	'python Submm_number_counts.py']

print(gen.colour_string(gen.string_important('PROCESSES TO RUN')+'\n', 'cyan'))
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
		print(gen.colour_string(gen.string_important(pn.upper())+'\n', 'orange')+'\n')
		os.system(rs)