############################################################################################################
# A script for determining the density of sources required to detect an overdensity using the method
# implemented in this pipeline.
############################################################################################################


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
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import quad


def shuffle_along_axis(a, axis):
	'''
	Shuffles an array along the specified axis.
	'''
	idx = np.random.rand(*a.shape).argsort(axis=axis)
	return np.take_along_axis(a,idx,axis=axis)



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
	#shuffle the array to ensure maximum randomness
	S = shuffle_along_axis(S, axis=0)
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

	nsamples = 100			#number of iterations to run per search radius
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
	S850, eS850_lo, eS850_hi, *_ = gen.get_relevant_cols_S19(data_submm, main_only=gen.main_only)
	#assign numbers to each source based on their position in the table
	idx_submm = np.arange(len(S850))

	#retrieve the blank-field number counts
	data_bf_file = PATH_COUNTS + f'{count_type}_with_errs_bf.npz'
	data_bf = np.load(data_bf_file)
	bin_edges_bf = data_bf['bin_edges']
	bin_centres_bf = (bin_edges_bf[1:] + bin_edges_bf[:-1]) / 2.
	dS_bf = bin_edges_bf[1:] - bin_edges_bf[:-1]
	data_s2c = data_bf['S2COSMOS']
	weights_bf = data_bf['w_S2COSMOS']

	#retrieve the blank-field MCMC outputs
	post_bf_file = PATH_SIMS + f'Schechter_posteriors/{count_type}_bf.npz'
	post_bf = np.load(post_bf_file)
	post_s2c = post_bf['S2COSMOS']

	#get the bin edges and centres used for the number counts
	bin_edges = gen.bin_edges#[gen.bin_edges >= gen.Smin]
	bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.
	#calculate the bin widths used for the number counts
	dS = bin_edges[1:] - bin_edges[:-1]
	nbins = len(dS)


	#calculate the blank field estimates
	if cumulative:
		X = bin_edges[:-1]
		bf_est = nc.cumulative_model(X, params_bf[0])
	else:
		X = bin_centres
		bf_est = nc.schechter_model(X, params_bf[0])

	idx_bins = np.digitize(S850, bin_edges)
	eS850_lo_med = [np.median(eS850_lo[idx_bins == i]) for i in range(1,nbins+1)]
	eS850_hi_med = [np.median(eS850_hi[idx_bins == i]) for i in range(1,nbins+1)]
	#calulate probabilities for selecting each bin centre sing the blank-field Schechter function
	P = nc.schechter_model(bin_centres, params_bf[0])
	#normalise the Schechter function so that the integral is equal to 1
	P = np.asarray(P / P.sum())
	#use the uncertainties to generate random flux densities
	S850_rand = np.array([stats.random_asymmetric_gaussian(bin_centres[i], eS850_lo_med[i], eS850_hi_med[i], 10000) for i in range(nbins)]).T


	#parameter permutations to try when fitting
	N0_range = np.arange(10, 200010, 10)
	S0_range = np.full(len(N0_range), params_bf[0][1])
	gamma_range = np.full(len(N0_range), params_bf[0][2])
	permut = np.array([N0_range, S0_range, gamma_range]).T

	#get the number of RAGERS galaxies
	t_analogues = Table.read(gen.PATH_CATS + 'RAGERS_COSMOS2020_matches_Mstar_z_rq.fits')
	n_ragers = len(np.unique(t_analogues['RAGERS_ID']))

	#filename for the table containing the minimum number of galaxies required for a signal 
	results_tab_file = PATH_RESULTS + f'{count_type}_min_gals_for_signal.txt'
	t_results = Table(names=['r', 'Nmin', 'surface_density', 'delta', 'edelta_lo', 'edelta_hi'])


	#cycle through the radii used for making the number counts
	for r in gen.r_search_all:
		print(gen.colour_string(f'{r:.1f} arcminute', 'orange'))

		#calculate the area of the aperture (in deg^-2)
		A = n_ragers * np.pi * (r/60.) ** 2.
		#calculate the weights to be applied to each bin
		if cumulative:
			weights = 1. / A
		else:
			weights = 1. / (A * dS)

		#number of simulated sources to use for generating number counts
		nsim_min = 0
		nsim_old = 0
		nsim = n_ragers * int(5 * r ** 2.)
		nsim_max = np.inf

		#names of the files containing the results from all iterations
		results_file = PATH_RESULTS + f'{count_type}_with_errs_{r:.1f}am.npz'

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
				#convert the convergent density into the `delta' typically used in the literature
				if cumulative:
					#for cumulative counts, compare the faintest bin with that of the blank field
					mask_bright_bf = bin_edges_bf[:-1] >= gen.Smin
					N_bf, eN_bf_lo, eN_bf_hi = data_s2c.T[mask_bright_bf][0]
					N_bf_rand = stats.random_asymmetric_gaussian(N_bf, eN_bf_lo, eN_bf_hi, 10000)

					#get the corresponding bin from the simulated number counts
					mask_bright = bin_edges[:-1] >= gen.Smin
					N_sim, eN_sim_lo, eN_sim_hi = results_dict[f'{nsim_max}gals'].T[mask_bright][0]
					N_sim_rand = stats.random_asymmetric_gaussian(N_sim, eN_sim_lo, eN_sim_hi, 10000)

					
					#calcualte delta
					delta_rand = (N_sim_rand / N_bf_rand) - 1.
					deltamed, edelta_lo, edelta_hi = stats.vals_and_errs_from_dist(delta_rand)

				else:

					mask_bright_bf = bin_centres_bf >= gen.Smin
					n_bf, en_bf_lo, en_bf_hi = data_s2c[:,mask_bright_bf]
					w_bf = dS_bf[mask_bright_bf]
					#generate random values for each bin and divide by the weights
					N_bf_rand = np.array([stats.random_asymmetric_gaussian(n_bf[i], en_bf_lo[i], en_bf_hi[i], 10000) / w_bf[i] for i in range(len(n_bf))])
					#take the sum for each bin
					N_bf_rand = np.sum(N_bf_rand, axis=0)

					#get the simulated number counts
					mask_bright = bin_centres >= gen.Smin
					n_sim, en_sim_lo, en_sim_hi = results_dict[f'{nsim_max}gals'][:,mask_bright]
					w_sim = dS[mask_bright]
					#generate random values for each bin and divide by the weights
					N_sim_rand = np.array([stats.random_asymmetric_gaussian(n_sim[i], en_sim_lo[i], en_sim_hi[i], 10000) / w_sim[i] for i in range(len(n_sim))])
					#take the sum for each bin
					N_sim_rand = np.sum(N_sim_rand, axis=0)

					
					#calculate delta
					delta_rand = (N_sim_rand / N_bf_rand) - 1.
					deltamed, edelta_lo, edelta_hi = stats.vals_and_errs_from_dist(delta_rand)


				t_results.add_row([r, nsim_max, nsim_max/A, deltamed, edelta_lo, edelta_hi])
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
			idx_sel = np.random.choice(range(nbins), (nsamples, nsim), p=P)

			#create a partial version of the number counts function
			number_counts_p = partial(number_counts, Srand=S850_rand, idx_sel=idx_sel, bin_edges=bin_edges, weights=weights, incl_poisson=True, cumulative=cumulative)

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


			N0, eN0_lo, eN0_hi = fit_N0(X, results_final, permut=permut, cumulative=cumulative)

			#calculate the ratio of the N0 parameter to that of the blank field (with uncertainty) and subtract 1
			N0_rand = stats.random_asymmetric_gaussian(N0, eN0_lo, eN0_hi, 10000)
			N0_bf_rand = stats.random_asymmetric_gaussian(params_bf[0][0], params_bf[1][0], params_bf[2][0], 10000)
			Q = (N0_rand / N0_bf_rand) - 1.
			#calculate the median and 1sigma percentiles
			Qmed, eQ_lo, eQ_hi = stats.vals_and_errs_from_dist(Q)


			#see if the result is >1sigma greater than 0
			sig = Qmed / eQ_lo
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

		t_results_r.write(results_file_r, format='ascii', overwrite=True)


	#put the results from each radius in a table and save to a file
	t_results.write(results_tab_file, format='ascii', overwrite=True)

