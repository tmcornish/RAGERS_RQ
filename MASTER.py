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
recreate_S19_comp = False	#recreate the completeness results from Simpson+19
random_datasets = True		#generate random flux densities (and completeness values) from the S2COSMOS catalogue
recreate_S19_nc = False		#recreate the number counts from Simpson+19
rq_sample = False			#select the sample of radio-quiet massive galaxies
calc_numcounts = False		#calculate number counts for 
numcounts_mega = False		#number counts mega script
apertures_test = False		#test the effect of different aperture sizes on the number counts

##################

settings = [
	recreate_S19_comp,
	random_datasets,
	recreate_S19_nc, 
	rq_sample,
	calc_numcounts,
	numcounts_mega,
	apertures_test
	]
proc_names = [
	'Recreating Simpson+19 completeness',
	'Generating random datasets',
	'Recreating Simpson+19 number counts', 
	'Selecting RQ sample', 
	'Constructing number counts',
	'Constructing number counts (mega script)',
	'Test different apertures'
	]
run_str = [
	'python Recreate_S19_completeness.py',
	'python Generate_random_datasets.py',
	'python Recreate_S19_number_counts.py',
	'python Select_radio_quiet_sample.py',
	'python Calculate_numcounts.py',
	'python Submm_number_counts.py',
	'python Test_search_radius.py'
	]

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