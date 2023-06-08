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

#number of walkers and iterations to use in MCMC fitting
nwalkers = 100
niter = 10000
#offsets for the initial walker positions from the initial guess values
offsets_init = [10., 0.01, 0.01]
#initial guesses for the parameters
popt_initial = [5000., 3., 1.6]

#minimum flux density allowed when fitting (mJy)
Smin = gen.Smin

#datasets to which Schechter functions will be fitted
data_to_fit = [f'zbin{i}' for i in range(1,5)] + [f'Mbin{i}' for i in range(1,4)] + ['ALL']

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

#list the files containing results to plot
nc_files = sorted(glob.glob(PATH_COUNTS+'Differential_numcounts_and_errs*.npz'))
cc_files = sorted(glob.glob(PATH_COUNTS+'Cumulative_numcounts_and_errs*.npz'))
#get the radii used
radii = [float(s.split('_')[-1][:3]) for s in nc_files]

#cycle through the radii
for r, ncf, ccf in zip(radii, nc_files, cc_files):

	print(gen.colour_string(f'Search radius = {r:.1f} arcminute', 'purple'))

	#destination files for the best-fit parameters and uncertainties
	nc_params_file = PATH_PARAMS + f'Differential_{r:.1f}am.npz'
	cc_params_file = PATH_PARAMS + f'Cumulative_{r:.1f}am.npz'
	#set up dictionaries for these results
	nc_params_dict, cc_params_dict = {}, {}

	#destination files for the posterior distributions of each parameter
	nc_post_file = PATH_POSTS + f'Differential_{r:.1f}am.npz'
	cc_post_file = PATH_POSTS + f'Cumulative_{r:.1f}am.npz'
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

		#retrieve the bin heights and uncertainties
		y_nc, ey_nc_lo, ey_nc_hi = nc_data[k]
		y_cc, ey_cc_lo, ey_cc_hi = cc_data[k]
		#symmetrise the uncertainties
		ey_nc = (ey_nc_lo + ey_nc_hi) / 2.
		ey_cc = (ey_cc_lo + ey_cc_hi) / 2.

		#fit to the differential number counts, excluding any bins below the the flux density limit
		nc_masks = nc.mask_numcounts(bin_centres, y_nc, limits=False, exclude_all_zero=False, Smin=Smin)
		popt_nc, epopt_lo_nc, epopt_hi_nc, sampler_nc = nc.fit_schechter_mcmc(
			bin_centres[nc_masks[0]],
			y_nc[nc_masks[0]],
			ey_nc[nc_masks[0]],
			nwalkers,
			niter,
			popt_initial,
			offsets_init,
			return_sampler=True)
		#add the best-fit values and uncertainties to the dictionary in a 2D array
		nc_params_dict[k] = np.array([popt_nc, epopt_lo_nc, epopt_hi_nc])
		#add the sampler flatchain to the posteriors dictionary
		nc_post_dict[k] = sampler_nc.flatchain

		#fit to the cumulative number counts, excluding any bins below the the flux density limit
		cc_masks = nc.mask_numcounts(bin_edges[:-1], y_cc, limits=False, exclude_all_zero=False, Smin=Smin)
		popt_cc, epopt_lo_cc, epopt_hi_cc, sampler_cc = nc.fit_schechter_mcmc(
			bin_edges[:-1][nc_masks[0]],
			y_cc[nc_masks[0]],
			ey_cc[nc_masks[0]],
			nwalkers,
			niter,
			popt_initial,
			offsets_init,
			return_sampler=True)
		#add the best-fit values and uncertainties to the dictionary in a 2D array
		cc_params_dict[k] = np.array([popt_cc, epopt_lo_cc, epopt_hi_cc])
		#add the sampler flatchain to the posteriors dictionary
		cc_post_dict[k] = sampler_cc.flatchain

	#save the created dictionaries in the destination files
	np.savez_compressed(nc_params_file, **nc_params_dict)
	np.savez_compressed(cc_params_file, **cc_params_dict)
	np.savez_compressed(nc_post_file, **nc_post_dict)
	np.savez_compressed(cc_post_file, **cc_post_dict)

