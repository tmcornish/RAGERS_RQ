############################################################################################################
# A script for backing up results from selected directories.
############################################################################################################

from datetime import date
import os
import general as gen

##################
#### SETTINGS ####
##################

#choice of things to back up
numcounts = True
numcount_dists = True
plots = True
params = True
posteriors = True
settings = [
	numcounts,
	numcount_dists,
	plots,
	params,
	posteriors
]

#print the chosen settings to the Terminal
print(gen.colour_string('CHOSEN SETTINGS:', 'white'))
settings_print = [
	'Back up number counts: ',
	'Back up number count distributions: ',
	'Back up figures: ',
	'Back up Schechter parameters: ',
	'Back up Schechter posteriors: '
]
for i in range(len(settings_print)):
	if settings[i]:
		settings_print[i] += 'y'
	else:
		settings_print[i] += 'n'

print(gen.colour_string('\n'.join(settings_print), 'white'))



#######################################################
###############    START OF SCRIPT    #################
#######################################################


#retrieve today's date and format it as 'yyyymmdd'
today = date.today()
today_str = today.strftime(f'{today.year}{today.month}{today.day}')

#create a list of directories
dirs = []
if numcounts:
	dirs.append(gen.PATH_CATS + 'Number_counts/')
if numcount_dists:
	dirs.append(gen.PATH_SIMS + 'Differential_numcount_dists/')
	dirs.append(gen.PATH_SIMS + 'Cumulative_numcount_dists/')
if plots:
	dirs.append(gen.PATH_PLOTS)
if params:
	dirs.append(gen.PATH_CATS + 'Schechter_params/')
if posteriors:
	dirs.append(gen.PATH_SIMS + 'Schechter_posteriors/')


for D in dirs:
	D_new = D + today_str + '/'
	os.system(f'mkdir -p {D_new}')
	os.system(f'mv {D}*.npz '+ D_new)
	os.system(f'mv {D}*.pdf '+ D_new)
	os.system(f'mv {D}*.png '+ D_new)

