import os, sys
from matplotlib import pyplot as plt
from astropy.table import Table
import numpy as np
import plotstyle as ps
import general as gen
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy import units as u
import stats

#######################################################
###############    START OF SCRIPT    #################
#######################################################

#relevant paths
PATH_CATS = gen.PATH_CATS
PATH_PLOTS = gen.PATH_PLOTS
PATH_SIMS = gen.PATH_SIMS
PATH_PARAMS = PATH_CATS + 'Schechter_params/'
PATH_COUNTS = PATH_CATS + 'Number_counts/'
#directory containing the simulated number counts (relevant only if plot_sims = True)
PATH_SIM_NC = PATH_CATS + 'Significance_tests/'
main_only = True

#get the radii used
radii = gen.r_search_all
#get the area of each aperture in deg^2
A_r = [np.pi * (r / 60.) ** 2. for r in radii]

#list the files containing results to plot
cc_files = [PATH_COUNTS + f'Cumulative_with_errs_{r:.1f}am_{gen.n_gal}{gen.gal_type}.npz' for r in radii]


#load the raw differential and cumulative counts for the blank field data
data_cc_bf = np.load(PATH_COUNTS + 'Cumulative_with_errs_bf.npz')

#get the bin edges used for the blank field results
S19_bin_edges = data_cc_bf['bin_edges']
S19_bin_centres = (S19_bin_edges[:-1] + S19_bin_edges[1:]) / 2.

#retrieve the results for the S2COSMOS dataset
N_S19, eN_S19_lo, eN_S19_hi = data_cc_bf['S2COSMOS']


#catalogues containing the RQ and RL samples
data_rq = Table.read(PATH_CATS + 'RAGERS_COSMOS2020_matches_Mstar_z_rq.fits')
data_rl = Table.read(PATH_CATS + 'RAGERS_COSMOS2020_matches_Mstar_z_rl.fits')
#retrieve relevant data from catalogue containing submm data for S2COSMOS sources
data_submm = Table.read(gen.S19_cat, format='fits')
S850, eS850_lo, eS850_hi, RMS, RA_submm, DEC_submm, comp_cat = gen.get_relevant_cols_S19(data_submm, main_only=main_only)
#create SkyCoord objects from RA and Dec.
coords_submm = SkyCoord(RA_submm, DEC_submm, unit='deg')


#create SkyCoord objects for the RQ and RL samples
coords_rq = SkyCoord(data_rq['ALPHA_J2000'], data_rq['DELTA_J2000'], unit='deg')
coords_rl = SkyCoord(data_rl['ALPHA_J2000'], data_rl['DELTA_J2000'], unit='deg')

#load the randomly generated flux densities and completenesses for S2COSMOS sources
data_rand = np.load(PATH_SIMS + 'S2COSMOS_randomised_S850.npz')
S850_rand = data_rand['S850_rand'].T
comp_rand = data_rand['comp_rand'].T
#calculate the reciprocal of the completeness for each source
icomp_rand = 1. / comp_rand
#mask to remove sources with randomly generated flux densities below 3 mJy
mask = S850_rand < 3.
#replace the reciprocal completeness with zero for these sources
icomp_rand[mask] = 0.


#cycle through the search radii
for r in radii:
	A = np.pi * (r / 60.) ** 2.

	#expected number of sources in blank field
	Ntot_exp = N_S19[1] * A
	eNtot_exp_lo = eN_S19_lo[1] * A
	eNtot_exp_hi = eN_S19_hi[1] * A
	#generate random values of the expected counts
	Ntot_exp_rand = stats.random_asymmetric_gaussian(Ntot_exp, eNtot_exp_lo, eNtot_exp_hi, 10000)
	#symmetrise for simplicity
	eNtot_exp = 0.5 * (eNtot_exp_hi + eNtot_exp_lo)
	#set up lists for number of matches
	N_matches_rq = []
	N_matches_rl = []
	#set up lists for the delta values and uncertainties for each source
	deltag_rq, edeltag_rq_lo, edeltag_rq_hi = [], [], []
	deltag_rl, edeltag_rl_lo, edeltag_rl_hi = [], [], []

	#cycle through the RQ sources
	for i in range(len(data_rq)):
		coord_central = coords_rq[i]
		#find matched sources
		matches = coords_rq[i].separation(coords_submm) < r * u.arcmin
		#retrieve the inverse completenesses for these sources
		icomp = icomp_rand[matches]
		#sum these values 
		N_matches_rand = icomp.sum(axis=0)
		#take the median and relevant percentiles as the 'true' value and uncertainties
		N16, N_matches, N84 = np.percentile(N_matches_rand, q=[stats.p16, 50., stats.p84])
		eN_lo, eN_hi = np.diff([N16, N_matches, N84])
		eN_lo_p, eN_hi_p = np.nan_to_num(np.array(stats.poisson_errs_1sig(N_matches)))
		eN_lo = np.sqrt(eN_lo ** 2. + eN_lo_p ** 2.)
		eN_hi = np.sqrt(eN_hi ** 2. + eN_hi_p ** 2.)
		#now regenerate 10000 random values for N_matches with the Poissonian uncertainties included
		N_matches_rand = stats.random_asymmetric_gaussian(N_matches, eN_lo, eN_hi, 10000)
		#calculate delta
		delta_rand = (N_matches_rand / Ntot_exp_rand) - 1.
		d16, delta, d84 = np.nanpercentile(delta_rand, q=[stats.p16, 50., stats.p84])
		edelta_lo, edelta_hi = np.diff([d16, delta, d84])
		N_matches_rq.append(N_matches)
		deltag_rq.append(delta)
		edeltag_rq_lo.append(edelta_lo)
		edeltag_rq_hi.append(edelta_hi)
	#add columns to RQ table
	data_rq[f'N_submm_{r:g}'] = N_matches_rq
	data_rq[f'delta_{r:g}'] = deltag_rq
	data_rq[f'edelta_lo_{r:g}'] = edeltag_rq_lo
	data_rq[f'edelta_hi_{r:g}'] = edeltag_rq_hi
	data_rq[f'nsig_delta_{r:g}'] = data_rq[f'delta_{r:g}'] / data_rq[f'edelta_lo_{r:g}']
	ud_mask_rq = data_rq[f'delta_{r:g}'] < 0.
	data_rq[f'nsig_delta_{r:g}'][ud_mask_rq] = data_rq[f'delta_{r:g}'][ud_mask_rq] / data_rq[f'edelta_hi_{r:g}'][ud_mask_rq]
	#data_rq.write(PATH_CATS + 'RAGERS_COSMOS2020_matches_Mstar_z_rq_with_deltas.fits', overwrite=True)

	#cycle through the RL sources
	for i in range(len(data_rl)):
		coord_central = coords_rl[i]
		#find matched sources
		matches = coords_rq[i].separation(coords_submm) < r * u.arcmin
		#retrieve the inverse completenesses for these sources
		icomp = icomp_rand[matches]
		#sum these values 
		N_matches_rand = icomp.sum(axis=0)
		#take the median and relevant percentiles as the 'true' value and uncertainties
		N16, N_matches, N84 = np.percentile(N_matches_rand, q=[stats.p16, 50., stats.p84])
		eN_lo, eN_hi = np.diff([N16, N_matches, N84])
		eN_lo_p, eN_hi_p = np.array(stats.poisson_errs_1sig(N_matches))
		eN_lo = np.sqrt(eN_lo ** 2. + eN_lo_p ** 2.)
		eN_hi = np.sqrt(eN_hi ** 2. + eN_hi_p ** 2.)
		#now regenerate 10000 random values for N_matches with the Poissonian uncertainties included
		N_matches_rand = stats.random_asymmetric_gaussian(N_matches, eN_lo, eN_hi, 10000)
		#calculate delta
		delta_rand = (N_matches_rand / Ntot_exp_rand) - 1.
		d16, delta, d84 = np.nanpercentile(delta_rand, q=[stats.p16, 50., stats.p84])
		edelta_lo, edelta_hi = np.diff([d16, delta, d84])
		N_matches_rl.append(N_matches)
		deltag_rl.append(delta)
		edeltag_rl_lo.append(edelta_lo)
		edeltag_rl_hi.append(edelta_hi)
	#add columns to RL table
	data_rl[f'N_submm_{r:g}'] = N_matches_rl
	data_rl[f'delta_{r:g}'] = deltag_rl
	data_rl[f'edelta_lo_{r:g}'] = edeltag_rl_lo
	data_rl[f'edelta_hi_{r:g}'] = edeltag_rl_hi
	data_rl[f'nsig_delta_{r:g}'] = data_rl[f'delta_{r:g}'] / data_rl[f'edelta_lo_{r:g}']
	ud_mask_rl = data_rl[f'delta_{r:g}'] < 0.
	data_rl[f'nsig_delta_{r:g}'][ud_mask_rl] = data_rl[f'delta_{r:g}'][ud_mask_rl] / data_rl[f'edelta_hi_{r:g}'][ud_mask_rl]
	#data_rl.write(PATH_CATS + 'RAGERS_COSMOS2020_matches_Mstar_z_rl_with_deltas.fits', overwrite=True)

data_rq.write(PATH_CATS + 'RAGERS_COSMOS2020_matches_Mstar_z_rq_with_deltas.fits', overwrite=True)
data_rl.write(PATH_CATS + 'RAGERS_COSMOS2020_matches_Mstar_z_rl_with_deltas.fits', overwrite=True)