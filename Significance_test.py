############################################################################################################
# A script for determining the density of sources required to detect an overdensity using the method
# implemented in this pipeline.
############################################################################################################

'''
Plan for the script:
	- load blank field parameters
	- load in number counts data (one radius at a time)
	- generate 10^4 versions of the number counts using uncertainties on bin heights
	- use blank-field schechter parameters to create PDF for randomly drawing sources
	- randomly draw N sources from this PDF, 10^4 times
	- recalculate number counts including new sources for each realisation
	- combine results from all realisations into final bin heights with uncertainties
	- fit to the results
	- calculate ratio of N0 from new fit to blank field, with uncertainties
	- if ratio is <1sigma significant, increase N; if it is >1sigma, take midpoint between N and previous value of N (use N/2 if first iteration)
	- repeat until convergence reached
	- convert final N into surface density (divide by pi*R^2)
'''

'''
Alternative plan:
	- load blank field parameters
	- cycle through radii
	- use blank-field schechter parameters to create PDF for randomly drawing sources
	- cycle through n_iter iterations
	- each iteration, randomly draw N sources from the S2COSMOS catalogue using the blank-field PDF
	- for each source, retrieve the randomly generated flux densities
	- calculate number counts for each iteration
	- combine results from all iterations into final bin heights with uncertainties
	- fit to the results
	- calculate ratio of N0 from new fit to blank field, with uncertainties
	- if ratio is <1sigma significant, increase N; if it is >1sigma, take midpoint between N and previous value of N (use N/2 if first iteration)
	- repeat until convergence reached
	- convert final N into surface density (divide by pi*R^2)
'''

import os,sys
import general as gen
import numcounts as nc
import stats
import numpy as np
import plotstyle as ps
from astropy.table import Table
from multiprocessing import Pool, cpu_count
from functools import partial


def number_counts(S, bin_edges, weights, incl_poisson=True, cumulative=False):
	'''
	Constructs differential number counts for a given set of flux densities. Optionally also
	randomises the flux densities according to their uncertainties to get an estimate of the
	uncertainties on each bin (otherwise just assumes Poissonian uncertainties).

	Parameters
	----------
	S: array-like 
		Flux density values to be binned.

	bin_edges: array_like
		Edges of the bins to be used for constructing the number counts (includes rightmost edge).

	A: float or array-like
		Area probed for all flux density bins, or for each individual bin.

	idx: int
		Integer indicator to help keep track of processes if using multiprocessing.

	incl_poisson: bool
		Whether to include Poissonian uncertainties in the errorbars. 

	cumulative: bool
		If True, returns cumulative counts instead of differential.

	Returns
	----------
	N: array-like
		Differential number counts.

	eN_lo, eN_hi: array-like
		Lower and upper uncertainties on each bin height.

	'''

	#calculate the bin widths
	dS = bin_edges[1:] - bin_edges[:-1]

	#bin the sources in each dataset
	counts = np.array([np.histogram(S[i], bin_edges)[0] for i in range(len(S))])
	if cumulative:
		counts = np.cumsum(counts[:,::-1], axis=1)[:,::-1]
	N_rand = counts * weights
		
	#take the median values to be the true values and use the 16th and 84th percentiles to estiamte the uncertainties
	N16, N, N84 = np.percentile(N_rand, q=[stats.p16, 50, stats.p84], axis=0)
	eN_lo = N - N16
	eN_hi = N84 - N
		
	#includes Poissonian uncertainties - NOTE: currently treats the uncertainties as if they're
	#symmetric, which isn't necessarily true. Workarounds exist (see e.g. Barlow 2003) but they
	#give similar enough results for this to be okay for now.
	if incl_poisson:
		counts_med = np.median(counts, axis=0)
		eN_lo_p, eN_hi_p = np.array(stats.poisson_errs_1sig(counts_med)) * weights
		eN_lo = np.sqrt(eN_lo ** 2. + eN_lo_p ** 2.)
		eN_hi = np.sqrt(eN_hi ** 2. + eN_hi_p ** 2.)

	return N, eN_lo, eN_hi



def collate_results(data, nmax, nsim):
	'''
	Takes the results for a given set of galaxies and combines them in such a way
	that doesn't overload the memory of the device running it.

	Parameters
	----------
	data: array
		The outputs from running number_counts many times; contains the heights and uncertainties
		for each bin from each iteration.

	nmax: int
		The maximum number of simualted datasets that can be analysed at any one time.

	nsim: int
		The number of simulated values to create for each bin.


	Returns
	----------
	N: array-like
		Differential number counts.

	eN_lo, eN_hi: array-like
		Lower and upper uncertainties on each bin height.
	'''

	#retrieve the data for this key, and the weights for each bin
	ndata = len(data)
	#determine how many iterations required after nmax is enforces
	n_iter = int(np.ceil(ndata / nmax))

	results_final = []
	for n in range(n_iter):
		results_iter = []
		#generate distributions from the results for each sample
		for i in range(n*nmax, min((n+1)*nmax,ndata)):
			dists_now = [stats.random_asymmetric_gaussian(data[i][0][j], data[i][1][j], data[i][2][j], nsim) for j in range(len(data[i][0]))]
			results_iter.append(dists_now)
			del dists_now
		#concatenate the results along the second axis to get distributions for each bin across all samples in this clump
		results_iter = np.concatenate(results_iter, axis=1)
		#calculate the median and uncertainties
		N, eN_lo, eN_hi = stats.vals_and_errs_from_dist(results_iter, axis=1)
		#generate a new distribution using these parameters
		dists_iter = [stats.random_asymmetric_gaussian(N[j], eN_lo[j], eN_hi[j], nsim) for j in range(len(N))]
		results_final.append(dists_iter)
		del dists_iter

	#concatenate the results along the second axis to get distributions for each bin across all samples
	results_final = np.concatenate(results_final, axis=1)
	#calculate the median and uncertainties
	N, eN_lo, eN_hi = stats.vals_and_errs_from_dist(results_iter, axis=1)

	#return the key and the final results
	return N, eN_lo, eN_hi


def perform_fits(xbins, ybins, nwalkers=100, niter=10000, offsets_init=[10.,0.01,0.01,], popt_initial=[5000.,3.,1.6], cumulative=False):

	#retrieve the bin heights and uncertainties
	y, ey_lo, ey_hi = ybins
	#symmetrise the uncertainties
	ey = (ey_lo + ey_hi) / 2.

	#fit to the differential number counts, excluding any bins below the the flux density limit
	masks = nc.mask_numcounts(xbins, y, limits=False, exclude_all_zero=False, Smin=gen.Smin)
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




if __name__ == '__main__':
	
	##################
	#### SETTINGS ####
	##################

	#toggle `switches' for additional functionality
	make_plots = True			#make plots comparing the real number counts with the counts required for a signal
	cumulative = False			#use cumulative counts instead of differential
	settings = [
		make_plots,
		cumulative
	]
	#print the chosen settings to the Terminal
	print(gen.colour_string('CHOSEN SETTINGS:', 'white'))
	settings_print = [
		'Make plots comparing the real number counts with the counts required for a signal: ',
		'Use cumulative counts instead of differential: '
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

	nsamples = 1000			#number of iterations to run per search radius
	ncpu = cpu_count()-1	#number of CPUs to use when multiprocessing

	#string to use for differential vs cumulative counts
	if cumulative:
		count_type = 'Cumulative'
	else:
		count_type = 'Differential'

	#directories containing the number counts and best-fit parameters, and simulated datasets
	PATH_COUNTS = gen.PATH_CATS + 'Number_counts/'
	PATH_PARAMS = gen.PATH_CATS + 'Schechter_params/'
	PATH_SIMS = gen.PATH_SIMS
	#directory in which the simulated number counts will be stored
	PATH_RESULTS = gen.PATH_CATS + 'Significance_tests/'
	PATH_POSTS = gen.PATH_SIMS + 'Schechter_posteriors/Significance_tests/'
	PATH_PARAMS_NEW = PATH_PARAMS + 'Significance_tests/'
	for PATH in [PATH_RESULTS, PATH_POSTS, PATH_PARAMS_NEW]: 
		if not os.path.exists(PATH):
				os.system(f'mkdir -p {PATH}')

	#directory for plots if told to make them
	PATH_PLOTS = gen.PATH_PLOTS + 'Significance_tests/'
	#make the directory if it doesn't exist
	if make_plots and not os.path.exists(PATH_PLOTS):
		os.system(f'mkdir -p {PATH_PLOTS}')

	#load the best-fit parameters for the blank field (S2COSMOS)
	params_file_bf = PATH_PARAMS + f'{count_type}_bf.npz'
	params_bf = np.load(params_file_bf)['S2COSMOS']

	#retrieve the flux densities of sources in the S2COSMOS catalogue
	data_submm = Table.read(gen.S19_cat)
	S850, *_ = gen.get_relevant_cols_S19(data_submm, main_only=gen.main_only)
	#assign numbers to each source based on their position in the table
	idx_submm = np.arange(len(S850))

	#retrieve the randomly generated flux densities for the S2COSMOS sources
	rand_s2c_file = PATH_SIMS + 'S2COSMOS_randomised_S850.npz'
	data_rand = np.load(rand_s2c_file)
	S850_rand = data_rand['S850_rand']

	#calculate the bin widths used for the number counts
	dS = gen.bin_edges[1:] - gen.bin_edges[:-1]

	#use the blank-field Schechter parameters to create a probability distribution as a function of flux density
	P = nc.schechter_model(S850, params_bf[0])
	#limit source selection to above the flux density limit used for fitting Schechter functions
	P[S850 < gen.Smin] = 0.
	#normalise the Schechter function so that the integral is equal to 1
	P = np.asarray(P / P.sum())

	#cycle through the radii used for making the number counts
	for r in gen.r_search_all:
		print(gen.colour_string(f'{r:.1f} arcminute', 'orange'))

		#calculate the area of the aperture (in deg^-2)
		A = np.pi * (r/60.) ** 2.
		#calculate the weights to be applied to each bin
		if cumulative:
			weights = 1. / A
		else:
			weights = 1. / (A * dS)

		#number of simulated sources to use for generating number counts
		nsim_old = 0
		nsim = int(10 * r ** 2.)
		nsignal = np.inf

		#names of the files containing the results from all iterations
		results_file = PATH_RESULTS + f'{count_type}_with_errs_{r:.1f}am.npz'
		#destination file for the best-fit parameters and uncertainties
		params_file = PATH_PARAMS_NEW + f'{count_type}_{r:.1f}am.npz'
		#destination files for the posterior distributions of each parameter
		post_file = PATH_POSTS + f'{count_type}_{r:.1f}am.npz'

		#see if these files exist and load the results if they do
		dicts_all = []
		for file in [results_file, params_file, post_file]:
			if os.path.exists(file):
				data = np.load(file)
				dicts_all.append(dict(zip(data.files, [data[f] for f in data.files])))
			else:
				dicts_all.append({})
		results_dict, params_dict, post_dict = dicts_all

		#add the bin edges and weights to the results dictionary
		results_dict['bin_edges'] = gen.bin_edges
		results_dict['weights'] = weights

		while True:

			print(gen.colour_string(f'Trying with {nsim} galaxies...', 'cyan'))

			#test for convergence
			if nsim_old == nsim:
				break

			#if this radius and nsim has been run previously, load the results to determine if it gave a signal
			if f'nsig_{nsim}gals' in results_dict:
				if results_dict[f'nsig_{nsim}gals'] > 1.:
					nsim_new = (nsim_old + nsim) // 2
					nsim_old = nsim
					nsignal = nsim
					nsim = nsim_new
				else:
					nsim_old = nsim
					if nsignal == np.inf:
						nsim *= 2
					else:
						nsim = (nsim + nsignal) // 2
				continue
		

			#use the blank-field probability distribution to randomly select nsim S2COSMOS sources
			idx_sel = np.random.choice(idx_submm, (nsamples, nsim), p=P)

			#create a partial version of the number counts function
			number_counts_p = partial(number_counts, bin_edges=gen.bin_edges, weights=weights, incl_poisson=True, cumulative=cumulative)

			with Pool(cpu_count()-1) as pool:
				results_rand = np.asarray(pool.map(number_counts_p, [S850_rand[:,idx_sel[i]] for i in range(nsamples)]))

			#combine the results from all iterations into one
			results_final = collate_results(results_rand, 50, 10000)

			#add the results to the dictionary
			results_dict[f'{nsim}gals'] = np.array(results_final)
			print(np.array(results_final))

			#fit a function to the results
			if cumulative:
				X = gen.bin_edges[:-1]
			else:
				X = (gen.bin_edges[:-1] + gen.bin_edges[1:]) / 2.
			params, post = perform_fits(X, results_final, cumulative=cumulative)
			params_dict[f'{nsim}gals'] = params
			post_dict[f'{nsim}gals'] = post

			#calculate the ratio of the N0 parameter to that of the blank field (with uncertainty) and subtract 1
			N0_rand = stats.random_asymmetric_gaussian(params[0][0], params[1][0], params[2][0], 10000)
			N0_bf_rand = stats.random_asymmetric_gaussian(params_bf[0][0], params_bf[1][0], params_bf[2][0], 10000)
			delta = N0_rand / N0_bf_rand - 1.
			#calculate the median and 1sigma percentiles
			deltamed, edelta_lo, edelta_hi = stats.vals_and_errs_from_dist(delta)

			#see if the result is >1sigma greater than 0
			sig = deltamed / edelta_lo
			results_dict[f'nsig_{nsim}gals'] = sig
			if sig > 1:
					nsim_new = (nsim_old + nsim) // 2
					nsim_old = nsim
					nsignal = nsim
					nsim = nsim_new
			else:
				nsim_old = nsim
				if nsignal == np.inf:
					nsim *= 2
				else:
					nsim = (nsim + nsignal) // 2

		#save the various dictionaries
		np.savez_compressed(results_file, **results_dict)
		np.savez_compressed(params_file, **params_dict)
		np.savez_compressed(post_file, **post_dict)



