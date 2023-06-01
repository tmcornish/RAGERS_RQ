############################################################################################################
# A script for calculating and the sub-mm number counts in the environments of radio-quiet counterparts
# to the RAGERS radio-loud galaxies.
############################################################################################################

#import modules/packages
import os, sys
from astropy.table import Table, vstack
from astropy.io import fits
import general as gen
import numpy as np
import numcounts as nc
import stats
import astrometry as ast
import multiprocessing as mp
from multiprocessing import Pool, cpu_count, freeze_support
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy import units as u
from itertools import repeat
import tqdm
from memory_profiler import memory_usage
import psutil
from functools import partial


def number_counts(
	S850_rand,
	S850_bin_edges,
	comp_rand,
	data_rq,
	RAGERS_info,
	coords_submm,
	r_search,
	zbin_edges,
	Mbin_edges,
	independent_rq,
	PATH_NC_DISTS,
	PATH_CC_DISTS,
	idx
	):

	'''
	Calculates the number counts for groups of RQ galaxies, as well as in different redshift and 
	stellar mass bins. 

	Parameters
	----------
	S850_rand: array
		Array of flux densities to be binned for the number counts. If 2D, assumed to be the result
		of creating simulated datasets by randomly generating flux densities from the catalogue values
		and uncertainties.

	S850_bin_egdes: array
		Edges of the flux density bins.

	comp_rand: array 
		Completeness values corresponding to each flux density value in S850_rand.

	data_rq: Table
		Table of data for the radio-quiet galaxies that have been matched in stellar mass and redshift
		with the radio-loud RAGERS sample.

	RAGERS_info: list
		List containing three Columns: the IDs, redshifts and stellar masses of each RAGERS RL galaxy.

	coords_submm: SkyCoord array
		Array of SkyCoords for each submm source in the S2COSMOS catalogue.

	zbin_edges: array
		Edges of the redshift bins.

	Mbin_edges: array
		Edegs of the stellar mass bins.

	independent_rq: bool
		Whether to treat each RQ galaxy independently from each other.

	PATH_NC_DISTS: str
		Path of the directory in which the differential number count distributions will be stored.

	PATH_CC_DISTS: str
		Path of the directory in which the cumulative number count distributions will be stored.

	idx: int
		Integer indicator to help keep track of processes if using multiprocessing.


	Returns
	-------
	nc_name: str
		Path of the dictionary containing the differential number count distributions for this iteration.

	cc_name: str
		Path of the dictionary containing the cumulative number count distributions for this iteration.
	'''
	
	print(f'Started task {idx+1} ({mp.current_process().name}).', flush=True)


	#set up dictionaries for the results
	nc_dict = {'bin_edges' : S850_bin_edges} 
	cc_dict = {'bin_edges' : S850_bin_edges}

	#convert search radius to degrees
	R_deg = r_search / 60.
	#area of aperture (sq. deg)
	A_sqdeg = np.pi * R_deg ** 2.

	#retrieve information about the RAGERS galaxies
	RAGERS_IDs, RAGERS_zs, RAGERS_Ms = RAGERS_info

	#redshift bins for the RL galaxies
	zbin_centres = (zbin_edges[:-1] + zbin_edges[1:]) / 2.
	zbin_widths = np.diff(zbin_edges)

	#stellar mass bins for the RL galaxies
	Mbin_centres = (Mbin_edges[:-1] + Mbin_edges[1:]) / 2.
	Mbin_widths = np.diff(Mbin_edges)

	#make a note of the dimensions of S850_rand
	ndim = gen.get_ndim(S850_rand)

	###########################################################
	#### RANDOMLY SELECTING RQ GALAXIES FOR EACH RL GALAXY ####
	###########################################################

	#set up a list to which the data for the selected RQ sources will be appended
	data_rq_sub = []

	#make a copy of the RQ data
	data_rq_copy = data_rq.copy()
	#cycle through the RAGERS RL IDs
	for ID in RAGERS_IDs:
		#get the indices in the catalogue for RQ galaxies matched to the current RL galaxy
		idx_matched = np.where(data_rq_copy['RAGERS_ID'] == ID)[0]
		#number of RQ galaxies to select for the current RL galaxy
		gen.n_rq_now = min(len(idx_matched), gen.n_rq)
		#randomly select gen.n_rq_now of these RQ galaxies
		idx_sel = np.random.choice(idx_matched, size=gen.n_rq_now, replace=False)
		#create a table containing this subset of RQ galaxies and append it to the list defined prior to this loop
		data_rq_sel = data_rq_copy[idx_sel]
		data_rq_sub.append(data_rq_sel)
		#create SkyCoord objects from these sources' coordinates
		coords_rq_sel = SkyCoord(data_rq_sel['ALPHA_J2000'], data_rq_sel['DELTA_J2000'], unit='deg')
		#create SkyCoord objects from the coordinates of all sources in the RQ catalogue
		coords_rq_all = SkyCoord(data_rq_copy['ALPHA_J2000'], data_rq_copy['DELTA_J2000'], unit='deg')
		#cross-match between the two within a tiny tolereance to identify duplicates
		idx_repeats, *_ = coords_rq_sel.search_around_sky(coords_rq_all, 0.0001*u.arcsec)
		#remove these sources from the RQ catalogue to avoid selecting them for a subsequent RL galaxy
		data_rq_copy.remove_rows(np.unique(idx_repeats))

	#stack the tables in the list to get the data for all seleted RQ galaxies in one table
	data_rq_sub = vstack(data_rq_sub)
	#create SkyCoord objects from the coordinates of all of these RQ galaxies
	coords_rq_sub = SkyCoord(data_rq_sub['ALPHA_J2000'], data_rq_sub['DELTA_J2000'], unit='deg')

	#create an empty list to which indices of matched submm sources will be appended for all RQ galaxies
	idx_matched_ALL = []

	##########################
	#### REDSHIFT BINNING ####
	##########################
	#cycle through the redshift bins
	for i in range(len(zbin_centres)):
		#get the current redshift bin and print its bounds
		z = zbin_centres[i]
		dz = zbin_widths[i]
	
		#get the IDs of all RAGERS sources in the current bin
		zmask = (RAGERS_zs >= (z - dz / 2.)) * (RAGERS_zs < (z + dz / 2.))
		rl_zbin = RAGERS_IDs[zmask]

		#get the corresponding radio-quiet source data
		zmask_rq = (data_rq_sub['RAGERS_z'] >= (z - dz / 2.)) * (data_rq_sub['RAGERS_z'] < (z + dz / 2.))
		data_rq_zbin = data_rq_sub[zmask_rq]
		#get the corresponding SkyCoords
		coords_rq_zbin = coords_rq_sub[zmask_rq]

		#create an empty list to which indices of matched submm sources will be appended for each RQ galaxy in this zbin
		idx_matched_zbin = []

		###################################
		#### RQ GALAXIES PER RL GALAXY ####
		###################################


		#cycle through the RL galaxies in this redshift bin
		for j in range(len(rl_zbin)):
			#get the RL ID
			ID = rl_zbin[j]

			#select the RQ galaxies corresponding to this RL source
			mask_rq_rl = data_rq_zbin['RAGERS_ID'] == ID
			data_rq_rl = data_rq_zbin[mask_rq_rl]
			#get the SkyCoords for these objects
			coords_rq_rl = coords_rq_zbin[mask_rq_rl]

			#create an empty list to which indices of matched submm sources will be appended for each RQ galaxy corresponding to the current RL galaxy
			idx_matched_rl = []

			################################
			#### INDIVIDUAL RQ GALAXIES ####
			################################

			#cycle through the RQ galaxies matched to this RL galaxy
			for k in range(len(data_rq_rl)):
				#get the coordinates for the current RQ galaxy
				coord_central = coords_rq_rl[k:k+1]

				#search for submm sources within r_search of the galaxy
				idx_coords_submm_matched, *_ = coord_central.search_around_sky(coords_submm, r_search * u.arcmin)
				#append these indices to lists for (a) each RL galaxy, (b) each z bin, (c) all RL galaxies
				idx_matched_rl.append(idx_coords_submm_matched)
				idx_matched_zbin.append(idx_coords_submm_matched)
				idx_matched_ALL.append(idx_coords_submm_matched)


			#concatenate the arrays of indices of matched submm sources for this RL galaxy
			idx_matched_rl = np.concatenate(idx_matched_rl)

			#if each aperture is independent, the area is simply the sum of their areas
			if independent_rq:
				#calculate the total area surveyed for this RL galaxy
				A_rl = A_sqdeg * len(data_rq_rl)
			#if not treating each aperture as independent, remove duplicate matches from the list for this RL galaxy
			if not independent_rq:
				idx_matched_rl = np.unique(idx_matched_rl)
				#calculate the area covered, accounting for overlap between apertures
				A_rl = ast.apertures_area(coords_rq_rl, r=R_deg)

			#retrieve the flux densities and completenesses for the matched sources
			if ndim == 2:
				S850_matched_rl = S850_rand[:,idx_matched_rl]
				comp_matched_rl = comp_rand[:,idx_matched_rl]
			else:
				S850_matched_rl = S850_rand[idx_matched_rl]
				comp_matched_rl = comp_rand[idx_matched_rl]

			#construct the differential number counts
			N_rl_rand, counts_rl_rand = nc.differential_numcounts(
				S850_matched_rl,
				S850_bin_edges,
				A_rl,
				comp=comp_matched_rl,
				return_dist=True)
			#add results to dictionary
			nc_dict[ID] = N_rl_rand

			#construct the cumulative number counts
			cumN_rl_rand, _ = nc.cumulative_numcounts(
				counts=counts_rl_rand,
				A=A_rl,
				return_dist=True
				)
			#add results to dictionary
			cc_dict[ID] = cumN_rl_rand

		#concatenate the arrays of indices of matched submm sources for this z bin
		idx_matched_zbin = np.concatenate(idx_matched_zbin)
		
		#if each aperture is independent, the area is simply the sum of their areas
		if independent_rq:
			#calculate the total area surveyed for this RL galaxy
			A_zbin = A_sqdeg * len(data_rq_zbin)
		#if not treating each aperture as independent, remove duplicate matches from the list for this RL galaxy
		if not independent_rq:
			idx_matched_zbin = np.unique(idx_matched_zbin)
			#calculate the area covered, accounting for overlap between apertures
			A_zbin = ast.apertures_area(coords_rq_zbin, r=R_deg, save_fig=False)	

		#retrieve the flux densities and completenesses for the matched sources
		if ndim == 2:
			S850_matched_zbin = S850_rand[:,idx_matched_zbin]
			comp_matched_zbin = comp_rand[:,idx_matched_zbin]
		else:
			S850_matched_zbin = S850_rand[idx_matched_zbin]
			comp_matched_zbin = comp_rand[idx_matched_zbin]

		#construct the differential number counts
		N_zbin_rand, counts_zbin_rand = nc.differential_numcounts(
			S850_matched_zbin,
			S850_bin_edges,
			A_zbin,
			comp=comp_matched_zbin,
			return_dist=True)
		#add results to dictionary
		nc_dict[f'zbin{i+1}'] = N_zbin_rand

		#construct the cumulative number counts
		cumN_zbin_rand, _ = nc.cumulative_numcounts(
			counts=counts_zbin_rand,
			A=A_zbin,
			return_dist=True
			)
		#add results to dictionary
		cc_dict[f'zbin{i+1}'] = cumN_zbin_rand


	##############################
	#### STELLAR MASS BINNING ####
	##############################
	#cycle through the redshift bins
	for i in range(len(Mbin_centres)):
		#get the current redshift bin and print its bounds
		logM = Mbin_centres[i]
		dlogM = Mbin_widths[i]

		#get the IDs of all RAGERS sources in the current bin
		mass_mask = (RAGERS_Ms >= (logM - dlogM / 2.)) * (RAGERS_Ms < (logM + dlogM / 2.))
		rl_Mbin = RAGERS_IDs[mass_mask]

		#get the corresponding radio-quiet source data
		mass_mask_rq = (data_rq_sub['RAGERS_logMstar'] >= (logM - dlogM / 2.)) * (data_rq_sub['RAGERS_logMstar'] < (logM + dlogM / 2.))
		data_rq_Mbin = data_rq_sub[mass_mask_rq]
		#get the corresponding SkyCoords
		coords_rq_Mbin = coords_rq_sub[mass_mask_rq]

		#concatenate the arrays of indices of matched submm sources for this z bin
		idx_matched_Mbin, *_ = coords_rq_Mbin.search_around_sky(coords_submm, r_search * u.arcmin)
		
		#if each aperture is independent, the area is simply the sum of their areas
		if independent_rq:
			#calculate the total area surveyed for this RL galaxy
			A_Mbin = A_sqdeg * len(data_rq_Mbin)
		#if not treating each aperture as independent, remove duplicate matches from the list for this RL galaxy
		if not independent_rq:
			idx_matched_Mbin = np.unique(idx_matched_Mbin)
			#calculate the area covered, accounting for overlap between apertures
			A_Mbin = ast.apertures_area(coords_rq_Mbin, r=R_deg, save_fig=False)

		#retrieve the flux densities and completenesses for the matched sources
		if ndim == 2:
			S850_matched_Mbin = S850_rand[:,idx_matched_Mbin]
			comp_matched_Mbin = comp_rand[:,idx_matched_Mbin]
		else:
			S850_matched_Mbin = S850_rand[idx_matched_Mbin]
			comp_matched_Mbin = comp_rand[idx_matched_Mbin]

		#construct the differential number counts
		N_Mbin_rand, counts_Mbin_rand = nc.differential_numcounts(
			S850_matched_Mbin,
			S850_bin_edges,
			A_Mbin,
			comp=comp_matched_Mbin,
			return_dist=True)
		#add results to dictionary
		nc_dict[f'Mbin{i+1}'] = N_Mbin_rand

		#construct the cumulative number counts
		cumN_Mbin_rand, _ = nc.cumulative_numcounts(
			counts=counts_Mbin_rand,
			A=A_Mbin,
			return_dist=True
			)
		#add results to dictionary
		cc_dict[f'Mbin{i+1}'] = cumN_Mbin_rand

	#########################
	#### COMBINED COUNTS ####
	#########################

	#concatenate the arrays of indices of matched submm sources for this z bin
	idx_matched_ALL = np.concatenate(idx_matched_ALL)
	
	#if each aperture is independent, the area is simply the sum of their areas
	if independent_rq:
		#calculate the total area surveyed for this RL galaxy
		A_ALL = A_sqdeg * len(data_rq_sub)
	#if not treating each aperture as independent, remove duplicate matches from the list for this RL galaxy
	if not independent_rq:
		idx_matched_ALL = np.unique(idx_matched_ALL)
		#calculate the area covered, accounting for overlap between apertures
		A_ALL = ast.apertures_area(coords_rq_sub, r=R_deg, save_fig=False)

	#retrieve the flux densities and completenesses for the matched sources
	if ndim == 2:
		S850_matched_ALL = S850_rand[:,idx_matched_ALL]
		comp_matched_ALL = comp_rand[:,idx_matched_ALL]
	else:
		S850_matched_ALL = S850_rand[idx_matched_ALL]
		comp_matched_ALL = comp_rand[idx_matched_ALL]

	#construct the differential number counts
	N_ALL_rand, counts_ALL_rand = nc.differential_numcounts(
		S850_matched_ALL,
		S850_bin_edges,
		A_ALL,
		comp=comp_matched_ALL,
		return_dist=True)
	#add results to dictionary
	nc_dict['ALL'] = N_ALL_rand

	#construct the cumulative number counts
	cumN_ALL_rand, _ = nc.cumulative_numcounts(
		counts=counts_ALL_rand,
		A=A_ALL,
		return_dist=True
		)
	#add results to dictionary
	cc_dict['ALL'] = cumN_ALL_rand

	#write the dictionaries to a file
	nc_name = PATH_NC_DISTS + f'sample{idx}_{r_search:.1f}am.npz'
	cc_name = PATH_CC_DISTS + f'sample{idx}_{r_search:.1f}am.npz'
	np.savez_compressed(nc_name, **nc_dict)
	np.savez_compressed(cc_name, **cc_dict)

	print(f'Finished task {idx+1} ({mp.current_process().name}).', flush=True)

	return nc_name, cc_name




if __name__ == '__main__':
	#mp.set_start_method('spawn')
	freeze_support()

	##################
	#### SETTINGS ####
	##################

	#toggle `switches' for additional functionality
	use_S19_bins = False				#use the flux density bins from Simpson+19
	independent_rq = True			#treat the RQ galaxies as independent (i.e. do not account for overlap between search areas)
	comp_correct = True				#apply completeness corrections
	main_only = gen.main_only		#use only sources from the MAIN region of S2COSMOS for blank-field results
	repeat_sel = True				#perform the random RQ selection several times
	many_radii = True
	settings = [
		use_S19_bins,
		independent_rq, 
		comp_correct,
		main_only,
		repeat_sel,
		many_radii]

	#print the chosen settings to the Terminal
	print(gen.colour_string('CHOSEN SETTINGS:', 'white'))
	settings_print = [
		'Use flux density bins from Simpson+19: ', 
		'Treat RQ galaxies independently: ',
		'Apply completeness corrections: ',
		'Use MAIN region only (for blank-field results): ',
		'Repeat the random selection of RQ galaxies several times: ',
		'Calculate number counts using multiple search radii: '
		]
	for i in range(len(settings_print)):
		if settings[i]:
			settings_print[i] += 'y'
		else:
			settings_print[i] += 'n'

	#number of matched galaxies to use per RL galaxy when constructing the number counts
	settings_print.append(f'Number of RQ galaxies per RL galaxy: {gen.n_rq}')

	nsamples = 100		#number of times to reselect RQ subsamples
	if repeat_sel:
		settings_print.append(f'Number of times to select RQ samples: {nsamples}')
	#only use one sample if told to not repeat the selection
	else:
		nsamples = 1

	#radii (arcmin) of search areas to use in the submm data
	if many_radii:
		radii = [1., 2., 4., 6.]
	else:
		radii = [gen.r_search]
	settings_print.append(f'Radius used to search for RQ companions (arcmin): {", ".join(map(str, radii))}')

	print(gen.colour_string('\n'.join(settings_print), 'white'))



	##################################
	#### SETTINGS (NUMBER COUNTS) ####
	##################################

	if use_S19_bins:
		#load the table summarising the nmber counts results
		S19_results = Table.read(gen.S19_results_file, format='ascii')
		#bin edges and centres for the differential number counts
		S850_bin_edges = np.concatenate([np.array(S19_results['S850']), [22.]])
		#delete the Table to conserve memory
		del S19_results
	else:
		S850_bin_edges = np.array([2., 3., 5., 7., 9., 12., 15., 18., 22.])

	#redshift bin edges
	zbin_edges = np.arange(1., 3.5, 0.5)
	#stellar mass bin edges
	Mbin_edges = np.array([11., 11.2, 11.4, 11.7])

	#######################################################
	###############    START OF SCRIPT    #################
	#######################################################

	#catalogue containing data for (radio-quiet) galaxies from COSMOS2020 matched in M* and z with the radio-loud sample
	RQ_CAT = gen.PATH_CATS + 'RAGERS_COSMOS2020_matches_Mstar_z.fits'
	data_rq = Table.read(RQ_CAT, format='fits')
	#get the RAs and DECs
	RA_rq = data_rq['ALPHA_J2000']
	DEC_rq = data_rq['DELTA_J2000']
	#convert these into SkyCoord objects
	coords_rq = SkyCoord(RA_rq, DEC_rq, unit='deg')

	#retrieve relevant data from catalogue containing submm data for S2COSMOS sources
	data_submm = Table.read(gen.S19_cat, format='fits')
	S850, eS850_lo, eS850_hi, RMS, RA_submm, DEC_submm, comp_cat = gen.get_relevant_cols_S19(data_submm, main_only=main_only)
	#create SkyCoord objects from RA and Dec.
	coords_submm = SkyCoord(RA_submm, DEC_submm, unit='deg')

	#attempt to retrieve randomised flux densities and completenesses from an existing file
	S850_npz_file = gen.PATH_SIMS + 'S2COSMOS_randomised_S850.npz'
	if os.path.exists(S850_npz_file):
		rand_data = np.load(S850_npz_file)
		if 'S850_rand' in rand_data.files:
			S850_rand = rand_data['S850_rand']
		else:
			S850_rand = S850[:]
		if comp_correct:
			if 'comp_rand' in rand_data.files:
				comp_rand = rand_data['comp_rand']
			else:
				comp_rand = comp_cat[:]
		else:
			comp_rand = np.ones(S850_rand.shape)
	#if the file doesn't exist, just use the catalogue values
	else:
		S850_rand = S850[:]
		if comp_correct:
			comp_rand = comp_cat[:]
		else:
			comp_rand = np.ones(comp_cat)


	#retrieve a list of the unique RAGERS IDs from the catalogue and the number of matches
	RAGERS_IDs, idx_unique, n_rq_per_rl = np.unique(data_rq['RAGERS_ID'], return_index=True, return_counts=True)
	#sort the IDs in order of increasing number of RQ matches
	idx_ordered = np.argsort(n_rq_per_rl)
	RAGERS_IDs = RAGERS_IDs[idx_ordered]
	#also get lists of the RAGERS RL redshifts and stellar masses and order them
	RAGERS_zs = data_rq['RAGERS_z'][idx_unique]
	RAGERS_zs = RAGERS_zs[idx_ordered]
	RAGERS_Ms = data_rq['RAGERS_logMstar'][idx_unique]
	RAGERS_Ms = RAGERS_Ms[idx_ordered]
	RAGERS_info = [RAGERS_IDs, RAGERS_zs, RAGERS_Ms]

	#see if files exist containing the differential/cumulative number count distributions in each bin
	PATH_NC_DISTS = gen.PATH_SIMS + 'Differential_numcount_dists/'
	if not os.path.exists(PATH_NC_DISTS):
		os.system(f'mkdir -p {PATH_NC_DISTS}')
	PATH_CC_DISTS = gen.PATH_SIMS + 'Cumulative_numcount_dists/'
	if not os.path.exists(PATH_CC_DISTS):
		os.system(f'mkdir -p {PATH_CC_DISTS}')

	#get the number of CPUs available
	ncpu_avail = cpu_count()

	for r in radii:
		print(gen.colour_string(f'Using {r:.1f} arcminute search radius.', 'orange'))
		nc_npz_file = PATH_NC_DISTS + f'All_samples_{r:.1f}am.npz'
		if os.path.exists(nc_npz_file):
			nc_dict_dists = np.load(nc_npz_file)
			nc_dict_dists = dict(zip(nc_dict_dists.files, [nc_dict_dists[f] for f in nc_dict_dists.files]))
		else:
			nc_dict_dists = {'bin_edges' : S850_bin_edges}
		cc_npz_file = PATH_CC_DISTS + f'All_samples_{r:.1f}am.npz'
		if os.path.exists(cc_npz_file):
			cc_dict_dists = np.load(cc_npz_file)
			cc_dict_dists = dict(zip(cc_dict_dists.files, [cc_dict_dists[f] for f in cc_dict_dists.files]))
		else:
			cc_dict_dists = {'bin_edges' : S850_bin_edges}

		#find out how many times the above method has been run in the past
		if 'ALL' in nc_dict_dists:
			N_done = len(nc_dict_dists['ALL'])
		else:
			N_done = 0
		#calculate how many times it needs to be run in order to meet the required number
		N_todo = nsamples - N_done

		#since only the final argument of number_counts changes with each iteration, make a partial function
		number_counts_p = partial(
			number_counts,
			S850_rand,
			S850_bin_edges,
			comp_rand,
			data_rq,
			RAGERS_info,
			coords_submm,
			r,
			zbin_edges,
			Mbin_edges,
			independent_rq,
			PATH_NC_DISTS,
			PATH_CC_DISTS
			)

		#if using the Linux machine, need to conserve memory to avoid the Pool hanging
		if gen.pf == 'Linux':
			if N_todo > ncpu_avail:
				#calculate max memory usage when running number_counts to decide how many processes to run
				print(gen.colour_string('Testing memory usage of number_counts function...', 'purple'))
				mem_usage = memory_usage((number_counts_p, (-1,)))
				mem_usage = max(mem_usage) / 1000
				print(gen.colour_string(f'Memory used by number_counts (GB): {mem_usage}', 'blue'))
				mem_avail = (psutil.virtual_memory().available + psutil.swap_memory().free) / (1000**3)
				print(gen.colour_string(f'Available memory (GB): {mem_avail}', 'blue'))
				ncpu = min(int(mem_avail / mem_usage), ncpu_avail)

			#if not many iterations to run, just assign a maximum of 5 CPUs
			else:
				ncpu = min(5, N_todo)

		else:
			#since macOS dynamically assigns swap memory, just use number of CPUs bar 1
			ncpu = ncpu_avail - 1
		
		print(gen.colour_string(f'Assigning {ncpu} CPUs to run number_counts.', 'blue'))

		print(gen.colour_string(f'Calculating number counts for {N_todo} samples...', 'purple'))
		
		#create a Pool with the specified number of processes
		with Pool(ncpu) as pool:
			results_files = pool.map(number_counts_p, range(N_todo))
			#results_files = list(pool.imap_unordered(number_counts_star, nc_args))
			pool.close()
			pool.join()
		

		##################################################
		#### CALCULATING FINAL BIN HEIGHTS AND ERRORS ####
		##################################################
		
		print(gen.colour_string(f'Collating results from all samples...', 'purple'))

		nc_keys = list(RAGERS_IDs) + [f'zbin{n}' for n in range(1,len(zbin_edges))] + [f'Mbin{n}' for n in range(1,len(Mbin_edges))] + ['ALL']
		
		for nc_file, cc_file in results_files:
			nc_dist_now = np.load(nc_file)
			cc_dist_now = np.load(cc_file)

			for k in nc_keys:
				try:
					nc_dict_dists[k].append(nc_dist_now[k])
					cc_dict_dists[k].append(cc_dist_now[k])
				except KeyError:
					nc_dict_dists[k] = [nc_dist_now[k]]
					cc_dict_dists[k] = [cc_dist_now[k]]
			os.system(f'rm -f {nc_file} {cc_file}')
			del nc_dist_now, cc_dist_now

		#save the number count dictionaries as compressed numpy archives
		np.savez_compressed(nc_npz_file, **nc_dict_dists)
		np.savez_compressed(cc_npz_file, **cc_dict_dists)


		print(gen.colour_string(f'Calculating final bin heights and errors...', 'purple'))
		

		#names to give the files containing the final bin heights and unertainties
		nc_npz_file_final = gen.PATH_CATS + f'Differential_numcounts_and_errs_{r:.1f}am.npz'
		cc_npz_file_final = gen.PATH_CATS + f'Cumulative_numcounts_and_errs_{r:.1f}am.npz'

		#set up dictionaries for these results
		nc_final = {'bin_edges' : nc_dict_dists['bin_edges']}
		cc_final = {'bin_edges' : cc_dict_dists['bin_edges']}

		for k in nc_keys:
			#retrieve the distribution of differential number counts and covert to a 2D numpy array
			nc_dist = np.concatenate(nc_dict_dists[k], axis=0)
			#calculate the median and uncertainties
			N_final, eN_final_lo, eN_final_hi = stats.vals_and_errs_from_dist(nc_dist)
			#store these in the new dictionary
			nc_final[k] = np.array([N_final, eN_final_lo, eN_final_hi])

			#retrieve the distribution of cumulative number counts and covert to a 2D numpy array
			cc_dist = np.concatenate(cc_dict_dists[k], axis=0)
			#calculate the median and uncertainties
			N_final, eN_final_lo, eN_final_hi = stats.vals_and_errs_from_dist(cc_dist)
			#store these in the new dictionary
			cc_final[k] = np.array([N_final, eN_final_lo, eN_final_hi])


		#save the number count dictionaries as compressed numpy archives
		np.savez_compressed(nc_npz_file_final, **nc_final)
		np.savez_compressed(cc_npz_file_final, **cc_final)
		

	print(gen.colour_string(f'Done!', 'purple'))





