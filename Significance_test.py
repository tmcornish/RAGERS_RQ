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
from memory_profiler import memory_usage
import psutil


def number_counts(j, Srand, idx_sel, bin_edges, weights, incl_poisson=True, cumulative=False):
	'''
	Constructs differential number counts for a given set of flux densities. Optionally also
	randomises the flux densities according to their uncertainties to get an estimate of the
	uncertainties on each bin (otherwise just assumes Poissonian uncertainties).

	Parameters
	----------
	j: int
		Which row of idx_sel to use for this iteration.

	Srand: array
		Randomly drawn flux density values to be binned.

	idx_sel: array
		Randomly drawn indices corresponding to the specific flux densities to use for this iteration.

	bin_edges: array_like
		Edges of the bins to be used for constructing the number counts (includes rightmost edge).

	weights: array-like
		Weights to apply to each bin.

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

	#retrieve only the desired flux densities
	S = Srand[:, idx_sel[j]]
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



def collate_results(data, nsim):
	'''
	Takes the results for a given set of galaxies and combines them in such a way
	that doesn't overload the memory of the device running it.

	Parameters
	----------
	data: array
		The outputs from running number_counts many times; contains the heights and uncertainties
		for each bin from each iteration.

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
	
	dists = []
	#generate distributions from the results for each sample
	for i in range(ndata):
		dists.append([stats.random_asymmetric_gaussian(data[i][0][j], data[i][1][j], data[i][2][j], nsim) for j in range(len(data[i][0]))])
	#concatenate the results along the second axis to get distributions for each bin across all samples
	dists = np.concatenate(dists, axis=1)
	#calculate the median and uncertainties
	N, eN_lo, eN_hi = stats.vals_and_errs_from_dist(dists, axis=1)

	#return the key and the final results
	return N, eN_lo, eN_hi

'''
def perform_fits(xbins, ybins, permut, popt_initial=[5000., 3., 1.6], offsets_init=[10.,0.01,0.01], cumulative=False):

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
	
	return N0, eN0_lo, eN0_hi
'''

def fit_N0(xbins, ybins, permut, cumulative=False):

	#retrieve the bin heights and uncertainties
	y, ey_lo, ey_hi = ybins
	#symmetrise the uncertainties
	ey = (ey_lo + ey_hi) / 2.

	#calculate model values for each permutation of parameter combinations
	if cumulative:
		ymodel = np.array([nc.cumulative_model(xbins, permut[i]) for i in range(len(permut))])
	else:
		ymodel = np.array([nc.schechter_model(xbins, permut[i]) for i in range(len(permut))])

	#calculate chi^2
	chi2 = np.sum(((ymodel - y)/ey) ** 2., axis=1)

	#find the minimum chi^2 and where it occurs
	idx_min = np.argmin(chi2)
	chi2_min = chi2[idx_min]
	#retrieve the corresponding parameters
	N0 = permut[idx_min][0]

	#find the solutions satisfying chi2 < chi2_min + 1 to get uncertainties on N0
	chi2_mask = chi2 <= chi2_min + 1.
	N0_masked = permut[:,0][chi2_mask]
	eN0_lo = N0 - N0_masked.min()
	eN0_hi = N0_masked.max() - N0

	return N0, eN0_lo, eN0_hi


if __name__ == '__main__':
	
	##################
	#### SETTINGS ####
	##################

	#toggle `switches' for additional functionality
	make_plots = False			#make plots comparing the real number counts with the counts required for a signal
	cumulative = True			#use cumulative counts instead of differential
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
	ncpu_avail = cpu_count()-1	#number of CPUs to use when multiprocessing

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
	if not os.path.exists(PATH_RESULTS):
		os.system(f'mkdir -p {PATH_RESULTS}')
	'''
	PATH_POSTS = gen.PATH_SIMS + 'Schechter_posteriors/Significance_tests/'
	PATH_PARAMS_NEW = PATH_PARAMS + 'Significance_tests/'
	for PATH in [PATH_RESULTS, PATH_POSTS, PATH_PARAMS_NEW]: 
		if not os.path.exists(PATH):
				os.system(f'mkdir -p {PATH}')
	'''

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

	#get the bin edges and centres used for the number counts
	bin_edges = gen.bin_edges
	bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.
	#calculate the bin widths used for the number counts
	dS = gen.bin_edges[1:] - gen.bin_edges[:-1]

	#calculate the blank field estimates
	if cumulative:
		X = bin_edges[:-1]
		bf_est = nc.cumulative_model(bin_edges[:-1], params_bf[0])
	else:
		X = bin_centres
		bf_est = nc.schechter_model(bin_centres, params_bf[0])
	#index of the first bin above the flux limit
	idx_lim = np.argwhere(X >= gen.Smin).min()

	#use the blank-field Schechter parameters to create a probability distribution as a function of flux density
	P = nc.schechter_model(S850, params_bf[0])
	#limit source selection to within the bins used
	P[S850 < gen.bin_edges.min()] = 0.
	#normalise the Schechter function so that the integral is equal to 1
	P = np.asarray(P / P.sum())

	'''
	#initial parameter guesses for Schechter fitting (use blank field as starting point)
	popt_init = params_bf[0]
	#minimum and maximum bound of the priors to use when fitting
	priors_min = [10., 1., -1.]
	priors_max = [1000000., 10., 6.]
	'''


	#parameter permutations to try when fitting
	N0_range = np.arange(10, 200010, 10)
	S0_range = np.full(len(N0_range), params_bf[0][1])
	gamma_range = np.full(len(N0_range), params_bf[0][2])
	permut = np.array([N0_range, S0_range, gamma_range]).T

	#create lists for the minimum number of sources and corresponding number density for each radius
	#Nmin_all, density_all = [], []

	#filename for the table containing the minimum number of galaxies required for a signal 
	results_tab_file = PATH_RESULTS + f'{count_type}_min_gals_for_signal.txt'
	if os.path.exists(results_tab_file):
		t_results = Table.read(results_tab_file, format='ascii')
	else:
		t_results = Table(names=['r', 'Nmin', 'surface_density'])


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
		nsim_min = 0
		nsim_old = 0
		nsim = int(10 * r ** 2.)
		nsim_max = np.inf

		#names of the files containing the results from all iterations
		results_file = PATH_RESULTS + f'{count_type}_with_errs_{r:.1f}am.npz'
		#destination file for the best-fit parameters and uncertainties
		#params_file = PATH_PARAMS_NEW + f'{count_type}_{r:.1f}am.npz'
		#destination files for the posterior distributions of each parameter
		#post_file = PATH_POSTS + f'{count_type}_{r:.1f}am.npz'

		'''
		#see if these files exist and load the results if they do
		dicts_all = []
		#for file in [results_file, params_file, post_file]:
		for file in [results_file, params_file]:
			if os.path.exists(file):
				data = np.load(file)
				dicts_all.append(dict(zip(data.files, [data[f] for f in data.files])))
			else:
				dicts_all.append({})
		results_dict, params_dict, post_dict = dicts_all
		'''
		#if the results file exists, load it
		if os.path.exists(results_file):
			data = np.load(results_file)
			results_dict = dict(zip(data.files, [data[f] for f in data.files]))
		else:
			results_dict = {}


		#add the bin edges and weights to the results dictionary
		results_dict['bin_edges'] = gen.bin_edges
		results_dict['weights'] = weights

		#set up a table for containing the minimum required galaxies for each radius
		results_file_r = PATH_RESULTS + f'{count_type}_sig_test_results_{r:.1f}am.txt'
		if os.path.exists(results_file_r):
			t_results_r = Table.read(results_file_r, format='ascii')
		else:
			t_results_r = Table(names=['r', 'N', 'surface_density', 'N0', 'eN0_lo', 'eN0_hi', 'nsigma'])

		while True:

			print(gen.colour_string(f'Trying with {nsim} galaxies...', 'cyan'))

			#test for convergence
			if nsim_old == nsim:
				t_results.add_row([r, nsim_max, nsim_max/A])
				break

			#if this radius and nsim has been run previously, load the results to determine if it gave a signal
			if f'nsig_{nsim}gals' in results_dict:
				if results_dict[f'nsig_{nsim}gals'] > 1.:
					print('Lowering number of galaxies...')
					nsim_new = (nsim_min + nsim) // 2
					nsim_old = nsim
					nsim_max = nsim
					nsim = nsim_new
				else:
					print('Increasing number of galaxies...')
					nsim_old = nsim
					nsim_min = nsim
					if nsim_max == np.inf:
						nsim *= 2
					else:
						nsim = (nsim + nsim_max) // 2
				continue
		

			#use the blank-field probability distribution to randomly select nsim S2COSMOS sources
			idx_sel = np.random.choice(idx_submm, (nsamples, nsim), p=P)

			#create a partial version of the number counts function
			number_counts_p = partial(number_counts, Srand=S850_rand, idx_sel=idx_sel, bin_edges=gen.bin_edges, weights=weights, incl_poisson=True, cumulative=cumulative)

			#if using the Linux machine, need to conserve memory to avoid the Pool hanging
			if gen.pf == 'Linux':
				#calculate max memory usage when running number_counts to decide how many processes to run
				print(gen.colour_string('Testing memory usage of number_counts function...', 'purple'))
				mem_usage = memory_usage((number_counts_p, (0,)))
				mem_usage = max(mem_usage) / 1000
				print(gen.colour_string(f'Memory used by number_counts (GB): {mem_usage}', 'blue'))
				mem_avail = (psutil.virtual_memory().available + psutil.swap_memory().free) / (1000**3)
				print(gen.colour_string(f'Available memory (GB): {mem_avail}', 'blue'))
				ncpu = min(int(mem_avail / mem_usage), ncpu_avail)
			else:
				ncpu = ncpu_avail

			print(gen.colour_string(f'Assigning {ncpu} CPUs to run number_counts.', 'blue'))

			with Pool(ncpu) as pool:
				results_rand = np.asarray(pool.map(number_counts_p, range(nsamples)))

			#combine the results from all iterations into one
			results_final = collate_results(results_rand, 10000)

			#add the results to the dictionary
			results_dict[f'{nsim}gals'] = np.array(results_final)
			print(np.array(results_final))

			'''
			#use the results to change the initial guess for N0; use first bin above flux limit
			corr_factor = results_final[0][idx_lim] / bf_est[idx_lim]
			popt_init[0] *= corr_factor
			
			popt_N0 = popt_init[0] * corr_factor
			#calculate difference from old initial guess
			dN0 = popt_N0 - popt_init[0]
			#shift the prior bounds accordingly
			priors_min[0] *= corr_factor
			priors_max[0] *= corr_factor
			#update the initial guess
			popt_init[0] = popt_N0
			'''



			N0, eN0_lo, eN0_hi = fit_N0(X, results_final, permut=permut, cumulative=cumulative)
			'''
			#append these values to the lists
			#params_dict[f'{nsim}gals'] = params
			#post_dict[f'{nsim}gals'] = post
			'''

			#calculate the ratio of the N0 parameter to that of the blank field (with uncertainty) and subtract 1
			N0_rand = stats.random_asymmetric_gaussian(N0, eN0_lo, eN0_hi, 10000)
			N0_bf_rand = stats.random_asymmetric_gaussian(params_bf[0][0], params_bf[1][0], params_bf[2][0], 10000)
			delta = (N0_rand / N0_bf_rand) - 1.
			#calculate the median and 1sigma percentiles
			deltamed, edelta_lo, edelta_hi = stats.vals_and_errs_from_dist(delta)


			#see if the result is >1sigma greater than 0
			sig = deltamed / edelta_lo
			results_dict[f'nsig_{nsim}gals'] = sig

			#add a row to the Table summarising the results
			t_results_r.add_row([r, nsim, nsim/A, N0, eN0_lo, eN0_hi, sig])

			print(f'Significance of signal: {sig}')

			if sig > 1:
				print('Lowering number of galaxies...')
				nsim_new = (nsim_min + nsim) // 2
				nsim_old = nsim
				nsim_max = nsim
				nsim = nsim_new
			else:
				print('Increasing number of galaxies...')
				nsim_old = nsim
				nsim_min = nsim
				if nsim_max == np.inf:
					nsim *= 2
				else:
					nsim = (nsim + nsim_max) // 2

			#save the various dictionaries
			np.savez_compressed(results_file, **results_dict)
			#np.savez_compressed(params_file, **params_dict)
			#np.savez_compressed(post_file, **post_dict)


		#append the converged-upon number of galaxies and correpsonding surface density to the lists
		#Nmin_all.append(nsim_max)
		#density_all.append(nsim_max / A)

		t_results_r.write(results_file_r, format='ascii', overwrite=True)


	#put the results from each radius in a table and save to a file
	t_results.write(results_tab_file, format='ascii', overwrite=True)

