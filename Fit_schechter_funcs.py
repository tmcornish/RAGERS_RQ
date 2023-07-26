############################################################################################################
# A script for fitting Schechter functions to the sub-mm number counts calculated in the previous step of 
# the pipeline.
############################################################################################################

#import modules/packages
import os, sys
import general as gen
import plotstyle as ps
import numcounts as nc
import stats
import emcee
import glob
import numpy as np



##################
#### SETTINGS ####
##################

settings_print = []

#number of walkers and iterations to use in MCMC fitting
nwalkers = 100
niter = 100000
#offsets for the initial walker positions from the initial guess values
offsets_init = [10., 0.01, 0.01]
#initial guesses for the parameters
popt_initial = [5000., 3., 1.6]

settings_print.append(f'Number of walkers to use for MCMC: {nwalkers}')
settings_print.append(f'Number of iterations to use for MCMC: {niter}')

#minimum flux density allowed when fitting (mJy)
Smin = gen.Smin
settings_print.append(f'Flux density limit for bin inclusion: {Smin} mJy')

#datasets to which Schechter functions will be fitted
data_to_fit = ['ALL'] #[f'zbin{i}' for i in range(1,5)] + [f'Mbin{i}' for i in range(1,4)]


print(gen.colour_string('\n'.join(settings_print), 'white'))


###################
#### FUNCTIONS ####
###################

def perform_fits(xbins, ybins, cumulative=False):

	#retrieve the bin heights and uncertainties
	y, ey_lo, ey_hi = ybins
	#symmetrise the uncertainties
	ey = (ey_lo + ey_hi) / 2.

	#fit to the differential number counts, excluding any bins below the the flux density limit
	masks = nc.mask_numcounts(xbins, y, limits=False, exclude_all_zero=False, Smin=Smin)
	if not cumulative:
		popt, epopt_lo, epopt_hi, sampler = nc.fit_schechter_mcmc(
			xbins[masks[0]],
			y[masks[0]],
			ey[masks[0]],
			nwalkers,
			niter,
			popt_initial,
			offsets_init,
			return_sampler=True)
	else:
		popt, epopt_lo, epopt_hi, sampler = nc.fit_cumulative_mcmc(
			xbins[masks[0]],
			y[masks[0]],
			ey[masks[0]],
			nwalkers,
			niter,
			popt_initial,
			offsets_init,
			return_sampler=True)
	#add the best-fit values and uncertainties to the dictionary in a 2D array
	params = np.array([popt, epopt_lo, epopt_hi])
	#add the sampler flatchain to the posteriors dictionary
	post = sampler.flatchain

	return params, post

#######################################################
###############    START OF SCRIPT    #################
#######################################################

#relevant paths
PATH_CATS = gen.PATH_CATS
PATH_SIMS = gen.PATH_SIMS
PATH_COUNTS = PATH_CATS + 'Number_counts/'

#make a directories for the outputs of this script if they don't already exist
PATH_PARAMS = PATH_CATS + 'Schechter_params/'
PATH_POSTS = PATH_SIMS + 'Schechter_posteriors/'
for P in [PATH_PARAMS, PATH_POSTS]:
	if not os.path.exists(P):
		os.system(f'mkdir -p {P}')


#############################
#### BLANK-FIELD RESULTS ####
#############################

#files containing the blank-field results
nc_file = PATH_COUNTS + 'Differential_with_errs_bf.npz'
cc_file = PATH_COUNTS + 'Cumulative_with_errs_bf.npz'
#destination files for the best-fit parameters and uncertainties
nc_params_file = PATH_PARAMS + f'Differential_bf.npz'
cc_params_file = PATH_PARAMS + f'Cumulative_bf.npz'
#set up dictionaries for these results
nc_params_dict, cc_params_dict = {}, {}

#destination files for the posterior distributions of each parameter
nc_post_file = PATH_POSTS + f'Differential_bf.npz'
cc_post_file = PATH_POSTS + f'Cumulative_bf.npz'

#see if these data have been fitted previously
if not os.path.exists(cc_post_file):
	print(gen.colour_string('Fitting to blank-field results...', 'purple'))
	#set up dictionaries for these results
	nc_post_dict, cc_post_dict = {}, {}

	#load the data for the differential and cumulative number counts
	nc_data = np.load(nc_file)
	cc_data = np.load(cc_file)

	#retrieve the bin edges and calculate the bin centres
	bin_edges = nc_data['bin_edges']
	bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.

	print(gen.colour_string('S2COSMOS', 'orange'))

	#perform the fits and add the best-fit values and uncertainties to the dictionary in a 2D array
	nc_params_dict['S2COSMOS'], nc_post_dict['S2COSMOS'] = perform_fits(bin_centres, nc_data['S2COSMOS'])
	cc_params_dict['S2COSMOS'], cc_post_dict['S2COSMOS'] = perform_fits(bin_edges[:-1], cc_data['S2COSMOS'], cumulative=True)

	#save the created dictionaries in the destination files
	np.savez_compressed(nc_params_file, **nc_params_dict)
	np.savez_compressed(cc_params_file, **cc_params_dict)
	np.savez_compressed(nc_post_file, **nc_post_dict)
	np.savez_compressed(cc_post_file, **cc_post_dict)

####################
#### RQ RESULTS ####
####################

print(gen.colour_string('Fitting to RQ results...', 'purple'))

#get the radii used
radii = gen.r_search_all
#list the files containing results to plot
nc_files = [PATH_COUNTS + f'Differential_with_errs_{r:.1f}am_{gen.n_gal}{gen.gal_type}.npz' for r in radii]
cc_files = [PATH_COUNTS + f'Cumulative_with_errs_{r:.1f}am_{gen.n_gal}{gen.gal_type}.npz' for r in radii]

#cycle through the radii
for r, ncf, ccf in zip(radii, nc_files, cc_files):

	print(gen.colour_string(f'Search radius = {r:.1f} arcminute', 'orange'))

	#destination files for the best-fit parameters and uncertainties
	nc_params_file = PATH_PARAMS + f'Differential_{r:.1f}am_{gen.n_gal}{gen.gal_type}.npz'
	cc_params_file = PATH_PARAMS + f'Cumulative_{r:.1f}am_{gen.n_gal}{gen.gal_type}.npz'
	#set up dictionaries for these results
	nc_params_dict, cc_params_dict = {}, {}

	#destination files for the posterior distributions of each parameter
	nc_post_file = PATH_POSTS + f'Differential_{r:.1f}am_{gen.n_gal}{gen.gal_type}.npz'
	cc_post_file = PATH_POSTS + f'Cumulative_{r:.1f}am_{gen.n_gal}{gen.gal_type}.npz'
	#set up dictionaries for these results
	nc_post_dict, cc_post_dict = {}, {}

	#load the data for the differential and cumulative number counts
	nc_data = np.load(ncf)
	cc_data = np.load(ccf)

	#retrieve the bin edges and calculate the bin centres
	bin_edges = nc_data['bin_edges']
	bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.

	#cycle through the different datasets
	for k in data_to_fit:
		
		print(gen.colour_string(k, 'blue'))

		#perform the fits and add the best-fit values and uncertainties to the dictionary in a 2D array
		nc_params_dict[k], nc_post_dict[k] = perform_fits(bin_centres, nc_data[k])
		cc_params_dict[k], cc_post_dict[k] = perform_fits(bin_edges[:-1], cc_data[k], cumulative=True)


	#save the created dictionaries in the destination files
	np.savez_compressed(nc_params_file, **nc_params_dict)
	np.savez_compressed(cc_params_file, **cc_params_dict)
	np.savez_compressed(nc_post_file, **nc_post_dict)
	np.savez_compressed(cc_post_file, **cc_post_dict)

