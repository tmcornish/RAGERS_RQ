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
update_vla_cat = False		#estimates L_500MHz for VLA-COSMOS galaxies
recreate_S19_comp = False	#recreate the completeness results from Simpson+19
random_datasets = False		#generate random flux densities (and completeness values) from the S2COSMOS catalogue
recreate_S19_nc = False		#recreate the number counts from Simpson+19
rq_sample = True			#select the sample of radio-quiet massive galaxies
calc_numcounts = False		#calculate number counts
plot_areas = False			#plot the areas searched for a given sample of RQ galaxies
fit_schechter = False		#run MCMC to fit Schechter functions to the results
param_tables = False			#format best-fit results from MCMC into a LaTeX-style table
plot_numcounts = False		#plot the number counts
contour_plot = False			#make contour plots for the Schechter fit parameters
sig_test = False				#find density required to detect signal
density_hists = False		#construct histograms comparing the SMG density in different environments
backup = False				#backup selected directories to a folder marked with the current date
#numcounts_mega = False		#number counts mega script
#apertures_test = False		#test the effect of different aperture sizes on the number counts

##################

settings = [
	update_vla_cat,
	recreate_S19_comp,
	random_datasets,
	recreate_S19_nc, 
	rq_sample,
	calc_numcounts,
	plot_areas,
	fit_schechter,
	param_tables,
	plot_numcounts,
	contour_plot,
	sig_test,
	density_hists,
	backup
	#numcounts_mega,
	#apertures_test
	]
proc_names = [
	'Estimating L_500MHz for VLA-COSMOS galaxies',
	'Recreating Simpson+19 completeness',
	'Generating random datasets',
	'Recreating Simpson+19 number counts', 
	'Selecting RQ sample', 
	'Constructing number counts',
	'Plotting search areas',
	'Fitting Schechter functions',
	'Making LaTeX tables of best-fit parameters',
	'Plotting number counts',
	'Making contour plots',
	'Finding densities required to detect signal',
	'Constructing SMG density histograms',
	'Back up results'
	#'Constructing number counts (mega script)',
	#'Test different apertures'
	]
run_str = [
	'python Update_VLA_COSMOS_cat.py',
	'python Recreate_S19_completeness.py',
	'python Generate_random_datasets.py',
	'python Recreate_S19_number_counts.py',
	'python Select_radio_quiet_sample.py',
	'python Calculate_numcounts.py',
	'python Plot_search_areas.py',
	'python Fit_schechter_funcs.py',
	'python Make_schechter_param_tables.py',
	'python Plot_numcounts.py',
	'python Contour_plots.py', 
	'python Significance_test.py',
	'python Density_histograms.py',
	'python Backup_results.py'
	#'python Submm_number_counts.py',
	#'python Test_search_radius.py'
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