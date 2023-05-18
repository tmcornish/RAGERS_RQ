############################################################################################################
# A script for calculating and plotting the sub-mm number counts in the environments of 10 radio-quiet
# galaxies for each radio-loud galaxy. 
############################################################################################################

#import modules/packages
import os, sys
from astropy.table import Table, vstack
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import colorsys
import glob
from scipy.interpolate import LinearNDInterpolator
import scipy.optimize as opt
import general as gen
import plotstyle as ps
import numcounts as nc
import stats
import astrometry as ast

#######################################################################################
########### FORMATTING FOR GRAPHS #####################################################

#formatting for graphs
plt.style.use(ps.styledict)


############################
#### SETTINGS (GENERAL) ####
############################

#toggle `switches' for additional functionality
use_cat_from_paper = True		#use the unedited catalogue downloaded from the Simpson+19 paper website
use_S19_bins = True				#use the flux density bins from Simpson+19
plot_positions = True			#plot positions of selected RQ galaxies with their search areas
independent_rq = True			#treat the RQ galaxies as independent (i.e. do not account for overlap between search areas)
comp_correct = True				#apply completeness corrections
plot_cumulative = True			#plot cumulative number counts as well as differential
combined_plot = True			#create plot(s) summarising the results from all targets
bf_results = True				#adds the results obtained from the whole S2COSMOS catalogue
main_only = gen.main_only		#use only sources from the MAIN region of S2COSMOS for blank-field results
randomise_fluxes = True			#randomly draw flux densities from possible values
bin_by_mass = True				#bin galaxies by stellar mass as well as redshift
fit_schechter = True			#fit Schechter finctions to the number counts
repeat_sel = True				#perform the random RQ selection several times 
settings = [
	use_cat_from_paper,
	use_S19_bins,
	plot_positions, 
	independent_rq, 
	comp_correct,
	plot_cumulative,
	combined_plot,
	bf_results,
	main_only,
	randomise_fluxes,
	bin_by_mass,
	fit_schechter,
	repeat_sel]

#print the chosen settings to the Terminal
print(gen.colour_string('CHOSEN SETTINGS:', 'white'))
settings_print = [
	'Use catalogue linked to Simpson+19 paper: ',
	'Use flux density bins from Simpson+19: ', 
	'Plot RQ galaxy positions: ',
	'Treat RQ galaxies independently: ',
	'Apply completeness corrections: ',
	'Plot cumulative number counts: ',
	'Combine all targets into one plot: ',
	'Plot blank-field results: ',
	'Use MAIN region only (for blank-field results): ',
	'Randomise source flux densities for error analysis: ',
	'Bin galaxies by stellar mass: ',
	'Fit Schechter functions: ',
	'Repeat the random selection of RQ galaxies several times: ']
for i in range(len(settings_print)):
	if settings[i]:
		settings_print[i] += 'y'
	else:
		settings_print[i] += 'n'

#radius of search area to use in the submm data
R_arcmin = gen.r_search
settings_print.append(f'Radius used to search for RQ companions (arcmin): {R_arcmin}')

nsim = gen.nsim		#number of iterations to use if randomising the flux densities/completeness
if randomise_fluxes:
	settings_print.append(f'Number of iterations for randomisation: {nsim}')

N_sel = gen.n_rq	#number of matched galaxies to use per RL galaxy when constructing the number counts
settings_print.append(f'Number of RQ galaxies per RL galaxy: {N_sel}')

nsamples = 100		#number of times to reselect RQ subsamples
if repeat_sel:
	settings_print.append(f'Number of times to select RQ samples: {nsamples}')

#number of walkers and iterations to use in MCMC fitting
nwalkers = 100
niter = 1000
if fit_schechter:
	settings_print.append(f'Number of walkers to use for MCMC: {nwalkers}')
	settings_print.append(f'Number of iterations to use for MCMC: {niter}')

print(gen.colour_string('\n'.join(settings_print), 'white'))


#relevant paths
PATH_RAGERS = gen.PATH_RAGERS
PATH_CATS = gen.PATH_CATS
PATH_DATA = gen.PATH_DATA
PATH_PLOTS = gen.PATH_PLOTS


##################################
#### SETTINGS (NUMBER COUNTS) ####
##################################

#convert search radius to degrees
R_deg = R_arcmin / 60.
#area of aperture (sq. deg)
A_sqdeg = np.pi * R_deg ** 2.

if use_S19_bins:
	#load the table summarising the nmber counts results
	S19_results_file = PATH_CATS + 'Simpson+19_number_counts_tab.txt'
	S19_results = Table.read(S19_results_file, format='ascii')
	#bin edges and centres for the differential number counts
	S850_bin_edges = np.concatenate([np.array(S19_results['S850']), [22.]])
	S850_bin_centres = (S850_bin_edges[:-1] + S850_bin_edges[1:]) / 2.
	#delete the Table to conserve memory
	del S19_results, S19_results_file
	#bin widths
	dS = S850_bin_edges[1:] - S850_bin_edges[:-1]
else:
	#850 micron flux density bins to use for submm number counts
	dS = 2.		#bin width (mJy)
	S850_bin_centres = np.arange(2.5, 22.5, dS)
	S850_bin_edges = np.append(S850_bin_centres-(dS/2.), S850_bin_centres[-1]+(dS/2.))



#######################################################
###############    START OF SCRIPT    #################
#######################################################


#catalogue containing data for (radio-quiet) galaxies from COSMOS2020 matched in M* and z with the radio-loud sample
RQ_CAT = PATH_CATS + 'RAGERS_COSMOS2020_matches_Mstar_z.fits'
data_rq = Table.read(RQ_CAT, format='fits')
#get the RAs and DECs
RA_rq = data_rq['ALPHA_J2000']
DEC_rq = data_rq['DELTA_J2000']
#convert these into SkyCoord objects
coords_rq = SkyCoord(RA_rq, DEC_rq, unit='deg')

if use_cat_from_paper:
	#catalogue containing submm data for S2COSMOS sources
	SUBMM_CAT = PATH_CATS + 'Simpson+19_S2COSMOS_source_cat.fits'
	data_submm = Table.read(SUBMM_CAT, format='fits')
	if main_only:
		data_submm = data_submm[data_submm['Sample'] == 'MAIN']
	#relevant column names
	RA_col, DEC_col = 'RA_deg', 'DEC_deg'
	S850_col, eS850_lo_col, eS850_hi_col = 'S850-deb', 'e_S850-deb', 'E_S850-deb'
	sample_col = 'Sample'
	rms_col = 'e_S850-obs'
else:
	#catalogue containing submm data for S2COSMOS sources
	SUBMM_CAT = PATH_CATS + 'S2COSMOS_sourcecat850_Simpson18.fits'
	data_submm = Table.read(SUBMM_CAT, format='fits')
	#relevant column names
	RA_col, DEC_col = 'RA_deg', 'DEC_deg'
	S850_col, eS850_lo_col, eS850_hi_col = 'S_deboost', 'S_deboost_errlo', 'S_deboost_errhi'
	sample_col = 'CATTYPE'
	rms_col = 'RMS'

#cut the catalogue to only include the MAIN sample if told to do so
if main_only:
	data_submm = data_submm[data_submm[sample_col] == 'MAIN']
#get the (deboosted) 850 µm flux densities and the uncertainties
S850 = data_submm[S850_col]
eS850_lo = data_submm[eS850_lo_col]
eS850_hi = data_submm[eS850_hi_col]
#also get the RMS
RMS = data_submm[rms_col]
#get the RA and Dec.
RA_submm = data_submm[RA_col]
DEC_submm = data_submm[DEC_col]
#create SkyCoord objects
coords_submm = SkyCoord(RA_submm, DEC_submm, unit='deg')

if randomise_fluxes:
	#see if file exists containing randomised flux densities already
	npz_filename = PATH_CATS + 'S2COSMOS_randomised_S850.npz'
	if os.path.exists(npz_filename):
		rand_data = np.load(npz_filename)
		S850_rand = rand_data['S850_rand']
	else:
		S850_rand = np.array([stats.random_asymmetric_gaussian(S850[i], eS850_lo[i], eS850_hi[i], nsim) for i in range(len(S850))]).T
		#set up a dictionary containing the randomised data
		dict_rand = {'S850_rand':S850_rand}
else:
	S850_rand = S850[:]
#make a note of the dimensions of S850_rand
ndim = gen.get_ndim(S850_rand)

######################
#### FIGURE SETUP ####
######################

#create the figure (dN/dS vs S)
f1, ax1 = plt.subplots(2, 2, figsize=(2.*ps.x_size, 2.*ps.y_size))
#to label axes with common labels, create a big subplot, make it invisible, and label its axes
ax_big1 = f1.add_subplot(111, frameon=False)
ax_big1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, which='both')
#label x and y axes
ax_big1.set_xlabel(r'$S_{850~\mu{\rm m}}$ (mJy)', labelpad=10.)
ax_big1.set_ylabel(r'$dN/dS$ (deg$^{-2}$ mJy$^{-1}$)', labelpad=20.)

#locations at which minor ticks should be placed on each axis, and where they should be labeled
xtick_min_locs = list(np.arange(2,10,1)) + [20]
xtick_min_labels = [2, 5, 20]
xtick_min_labels = [f'{s:g}' if s in xtick_min_labels else '' for s in xtick_min_locs]

#colours to cycle through when plotting multiple curves on a set of axes
#c_cycle = [dark_red, dark_blue, green, coral, teal, magenta]

#remove a chunk of the rainbow colormap
#interval = np.hstack([np.linspace(0, 0.5, 100), np.linspace(0.8, 1., 100)])
interval = np.hstack([np.arange(0, 0.501, 0.001), np.arange(0.7, 1.001, 0.001)])
colors = plt.cm.rainbow(interval)
cmap = LinearSegmentedColormap.from_list('name', colors)

#markers to cycle through when plotting multiple datasets on a set of axes
markers = ['s', '^', 'v', '*', (8,1,0), 'x', '<', '>', 'p']

#if told to make an extra figure for the positions of the RQ and RL sources, create the figure
if plot_positions:
	#create the figure (Dec. vs RA)
	f2, ax2 = plt.subplots(2, 2, figsize=(2.*ps.x_size, 2.*ps.y_size))
	#to label axes with common labels, create a big subplot, make it invisible, and label its axes
	ax_big2 = f2.add_subplot(111, frameon=False)
	ax_big2.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, which='both')
	#label x and y axes
	ax_big2.set_xlabel('RA (deg)', labelpad=10.)
	ax_big2.set_ylabel('Dec. (deg)', labelpad=20.)

if plot_cumulative:
	#create the figure (N(>S) vs S)
	f3, ax3 = plt.subplots(2, 2, figsize=(2.*ps.x_size, 2.*ps.y_size))
	#to label axes with common labels, create a big subplot, make it invisible, and label its axes
	ax_big3 = f3.add_subplot(111, frameon=False)
	ax_big3.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, which='both')
	#label x and y axes
	ax_big3.set_xlabel(r'$S_{850~\mu{\rm m}}$ (mJy)', labelpad=10.)
	ax_big3.set_ylabel(r'$N(>S)$ (deg$^{-2}$)', labelpad=20.)

if combined_plot:
	#label to use for the combined data
	label_combined = 'All RQ targets'
	#list to which the legend labels will be appended in the desired order
	labels_ord_combined = [label_combined]

	#create the figure (dN/dS vs S)
	f4, ax4 = plt.subplots(1, 1)
	#label x and y axes
	ax4.set_xlabel(r'$S_{850~\mu{\rm m}}$ (mJy)')
	ax4.set_ylabel(r'$dN/dS$ (deg$^{-2}$ mJy$^{-1}$)')

	if plot_cumulative:
		#create the figure (N(>S) vs S)
		f5, ax5 = plt.subplots(1, 1)
		#label x and y axes
		ax5.set_xlabel(r'$S_{850~\mu{\rm m}}$ (mJy)')
		ax5.set_ylabel(r'$N(>S)$ (deg$^{-2}$)')

if bin_by_mass:
	#create the figure (dN/dS vs S)
	f6, ax6 = plt.subplots(1, 3, figsize=(3.*ps.x_size, ps.y_size))
	#to label axes with common labels, create a big subplot, make it invisible, and label its axes
	ax_big6 = f6.add_subplot(111, frameon=False)
	ax_big6.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, which='both')
	#label x and y axes
	ax_big6.set_xlabel(r'$S_{850~\mu{\rm m}}$ (mJy)', labelpad=10.)
	ax_big6.set_ylabel(r'$dN/dS$ (deg$^{-2}$ mJy$^{-1}$)', labelpad=20.)

	if plot_cumulative:
		#create the figure (N(>S) vs S)
		f7, ax7 = plt.subplots(1, 3, figsize=(3.*ps.x_size, ps.y_size))
		#to label axes with common labels, create a big subplot, make it invisible, and label its axes
		ax_big7 = f7.add_subplot(111, frameon=False)
		ax_big7.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, which='both')
		#label x and y axes
		ax_big7.set_xlabel(r'$S_{850~\mu{\rm m}}$ (mJy)', labelpad=10.)
		ax_big7.set_ylabel(r'$N(>S)$ (deg$^{-2}$)', labelpad=20.)

###########################################################
#### RANDOMLY SELECTING RQ GALAXIES FOR EACH RL GALAXY ####
###########################################################

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

#set up a list to which the data for the selected RQ sources will be appended
data_rq_sub = []
#cycle through the RAGERS RL IDs
for ID in RAGERS_IDs:
	#get the indices in the catalogue for RQ galaxies matched to the current RL galaxy
	idx_matched = np.where(data_rq['RAGERS_ID'] == ID)[0]
	#number of RQ galaxies to select for the current RL galaxy
	N_sel_now = min(len(idx_matched), N_sel)
	#randomly select N_sel_now of these RQ galaxies
	np.random.seed(0)
	idx_sel = np.random.choice(idx_matched, size=N_sel_now, replace=False)
	#create a table containing this subset of RQ galaxies and append it to the list defined prior to this loop
	data_rq_sel = data_rq[idx_sel]
	data_rq_sub.append(data_rq_sel)
	#create SkyCoord objects from these sources' coordinates
	coords_rq_sel = SkyCoord(data_rq_sel['ALPHA_J2000'], data_rq_sel['DELTA_J2000'], unit='deg')
	#create SkyCoord objects from the coordinates of all sources in the RQ catalogue
	coords_rq_all = SkyCoord(data_rq['ALPHA_J2000'], data_rq['DELTA_J2000'], unit='deg')
	#cross-match between the two within a tiny tolereance to identify duplicates
	idx_repeats, *_ = coords_rq_sel.search_around_sky(coords_rq_all, 0.0001*u.arcsec)
	#remove these sources from the RQ catalogue to avoid selecting them for a subsequent RL galaxy
	data_rq.remove_rows(np.unique(idx_repeats))

#stack the tables in the list to get the data for all seleted RQ galaxies in one table
data_rq_sub = vstack(data_rq_sub)
#save the subsample of RQ galaxies to a new file
data_rq_sub.write(PATH_CATS + 'subsamp_RAGERS_COSMOS2020_matches_Mstar_z.fits', overwrite=True)

#create SkyCoord objects from the coordinates of all of these RQ galaxies
coords_rq_sub = SkyCoord(data_rq_sub['ALPHA_J2000'], data_rq_sub['DELTA_J2000'], unit='deg')


###########################################
#### DERIVING COMPLETENESS CORRECTIONS ####
###########################################

if comp_correct:
	print(gen.colour_string('Calculating completeness corrections...', 'purple'))

	#see if a file exists with the reconstructed S2COSMOS completeness grid (should have been created
	#in previous step of pipeline)
	compgrid_file = PATH_DATA + 'Completeness_at_S850_and_rms.fits'
	if os.path.exists(compgrid_file):
		CG_HDU = fits.open(compgrid_file)[0]
		zgrid = CG_HDU.data.T
		hdr = CG_HDU.header
		#get the min, max and step along the S850 and RMS axes
		xmin = hdr['CRVAL1']
		xstep = hdr['CDELT1']
		xmax = xmin + (hdr['NAXIS1'] - 1) * xstep
		ymin = hdr['CRVAL2']
		ystep = hdr['CDELT2']
		ymax = ymin + (hdr['NAXIS2'] - 1) * ystep
		#create a grid in flux density-RMS space
		xgrid, ygrid = np.mgrid[xmin:xmax+xstep:xstep, ymin:ymax+ystep:ystep]
		#interpolate the completeness values from the file w.r.t. S850 and RMS
		points = np.array([xgrid.flatten(), ygrid.flatten()])
		values = zgrid.flatten()
		comp_interp = LinearNDInterpolator(points.T, values)
	#otherwise, reconstruct the completeness from scratch
	else:
		#list of files contaning the flux densities and RMS values corresponding to a given completeness
		comp_files = sorted(glob.glob(PATH_CATS + 'Simpson+19_completeness_curves/*'))
		#corresponding completenesses
		defined_comps = [0.1, 0.3, 0.5, 0.7, 0.9]
		#min, max and step of the axes in the completeness grid: x (flux density) and y (RMS)
		xparams = [-15., 30., 0.01]
		yparams = [0.45, 3.05, 0.01]

		#run the function that reconstructs the completeness grid from Simpson+19
		comp_interp, zgrid = nc.recreate_S19_comp_grid(comp_files, defined_comps, xparams, yparams, plot_grid=False)
		#save the completeness grid to a FITS file
		gen.array_to_fits(zgrid.T, compgrid_file, CRPIX=[1,1], CRVAL=[xparams[0],yparams[0]], CDELT=[xparams[2],yparams[2]])

	if randomise_fluxes:
		#see if the completeness has already been calculated for the randomised flux densities
		if 'rand_data' in globals():
			comp_submm = rand_data['comp_rand']
		else:
			#set up a multiprocessing Pool using all but one CPU
			pool = Pool(cpu_count()-1)
			#calculate the completeness for the randomly generated flux densities
			comp_submm = np.array(pool.starmap(comp_interp, [[S850_rand[i], RMS] for i in range(len(S850_rand))]))
			#add these completenesses to the dictionary of randomly generated data
			dict_rand['comp_rand'] = comp_submm
	else:
		#calculate the completeness at the flux density and RMS of each S2COSMOS source
		comp_submm = comp_interp(S850_rand, data_submm[rms_col])

	print(gen.colour_string('Done!', 'purple'))

#if not told to do completeness corrections, just set the completeness to 1 for everything
else:
	comp_submm = np.ones(S850_rand.shape)


#############################
#### BLANK-FIELD RESULTS ####
#############################

#generate number counts from the entire S2COSMOS catalogue
if bf_results:
	#label to use for the blank field data points
	label_bf = 'S2COSMOS'

	#blank-field (S2COSMOS) survey area
	if main_only:
		A_bf = 1.6
	else:
		A_bf = 2.6

	#construct the differential number counts
	N_bf, eN_bf_lo, eN_bf_hi, counts_bf, weights = nc.differential_numcounts(S850_rand, S850_bin_edges, A_bf, comp=comp_submm, incl_poisson=True)
	xbins_bf = S850_bin_centres * 10. ** (0.004)
	if combined_plot:
		labels_ord_combined.append(label_bf)
		ax4.plot(xbins_bf, N_bf, linestyle='none', marker='D', color=ps.crimson, label=label_bf)
		ax4.errorbar(xbins_bf, N_bf, fmt='none', yerr=(eN_bf_lo,eN_bf_hi), ecolor=ps.crimson, elinewidth=2.)

	if plot_cumulative:
		#calculate the cumulative counts
		cumN_bf, ecumN_bf_lo, ecumN_bf_hi, cumcounts_bf = nc.cumulative_numcounts(counts=counts_bf, A=A_bf)
		xbins_cumbf = S850_bin_edges[:-1] * 10. ** (0.004)

		if combined_plot:
			ax5.plot(xbins_cumbf, cumN_bf, linestyle='none', marker='D', color=ps.crimson, label=label_bf)
			ax5.errorbar(xbins_cumbf, cumN_bf, fmt='none', yerr=(ecumN_bf_lo,ecumN_bf_hi), ecolor=ps.crimson, elinewidth=2.)


#########################################
#### BINNING RL GALAXIES BY REDSHIFT ####
#########################################

#create redshift bins for the RL galaxies
dz = 0.5
zbin_edges = np.arange(1., 3.+dz, dz)
zbin_centres = zbin_edges[:-1] + 0.5 * dz


####################################
#### CONSTRUCTING NUMBER COUNTS ####
####################################

print_str = 'Constructing number counts (redshift-binned)'
if plot_positions:
	print_str += ' and plotting RQ galaxy positions'
print_str += '...'
print(gen.colour_string(print_str, 'purple'))

#create an empty list to which indices of matched submm sources will be appended for all RQ galaxies
idx_matched_ALL = []

#cycle through the redshift bins
for i in range(len(zbin_centres)):
	#get the current redshift bin and print its bounds
	z = zbin_centres[i]
	print(f'{z-dz/2:.2f} < z < {z+dz/2.:.2f}')

	#plot data in the left-hand column if i is even; plot in the right-hand column if i is odd
	if (i % 2 == 0):
		row_z = int(i/2)			#the row in which the current subplot lies
		col_z = 0					#the column in which the current subplot lies
	else:
		row_z = int((i-1)/2)		#the row in which the current subplot lies
		col_z = 1					#the column in which the current subplot lies

	#get the IDs of all RAGERS sources in the current bin
	zmask = (RAGERS_zs >= (z - dz / 2.)) * (RAGERS_zs < (z + dz / 2.))
	rl_zbin = RAGERS_IDs[zmask]

	##############################
	#### RQ GALAXIES PER ZBIN ####
	##############################

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

	#list to which the legend labels will be appended in the desired order
	labels_ord = []
	#blank field  results
	if bf_results:
		labels_ord.append(label_bf)
		ax1[row_z,col_z].plot(xbins_bf, N_bf, linestyle='none', marker='D', color=ps.grey, label=label_bf, alpha=0.5)
		ax1[row_z,col_z].errorbar(xbins_bf, N_bf, fmt='none', yerr=(eN_bf_lo,eN_bf_hi), ecolor=ps.grey, elinewidth=2., alpha=0.5)
		if plot_cumulative:
			ax3[row_z,col_z].plot(xbins_cumbf, cumN_bf, linestyle='none', marker='D', color=ps.grey, label=label_bf, alpha=0.5)
			ax3[row_z,col_z].errorbar(xbins_cumbf, cumN_bf, fmt='none', yerr=(ecumN_bf_lo,ecumN_bf_hi), ecolor=ps.grey, elinewidth=2., alpha=0.5)

	#cycle through the RL galaxies in this redshift bin
	for j in range(len(rl_zbin)):
		#get the RL ID
		ID = rl_zbin[j]
		labels_ord.insert(-1, ID)
		#select the colour and marker to use for this dataset
		c_now = gen.scale_RGB_colour(cmap((j+1.)/len(rl_zbin))[:-1], scale_l=0.8)
		mkr_now = markers[j]

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

			#if told to plot the RQ galaxy positions, do so
			if plot_positions:
				#retrieve the RA and Dec. of the galaxy
				RA_rq = coord_central[0].ra.value
				DEC_rq = coord_central[0].dec.value
				#plot this position on the current axes
				ax2[row_z,col_z].plot(RA_rq, DEC_rq, linestyle='none', marker=mkr_now, c=c_now, label=ID, alpha=0.7)
				#add circles with radius 4' and 6' centred on the RQ galaxy
				d_circle1 = 8. / 60.
				d_circle2 = 12. / 60.
				f_cosdec = np.cos(DEC_rq * np.pi / 180.)
				ellipse1 = mpl.patches.Ellipse((RA_rq, DEC_rq), width=d_circle1/f_cosdec, height=d_circle1, color=c_now, fill=False, alpha=0.7)
				ellipse2 = mpl.patches.Ellipse((RA_rq, DEC_rq), width=d_circle2/f_cosdec, height=d_circle2, color=c_now, fill=False, linestyle='--', alpha=0.7)
				ax2[row_z,col_z].add_patch(ellipse1)
				ax2[row_z,col_z].add_patch(ellipse2)

			#search for submm sources within R_arcmin of the galaxy
			idx_coords_submm_matched, *_ = coord_central.search_around_sky(coords_submm, R_arcmin * u.arcmin)
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
			comp_matched_rl = comp_submm[:,idx_matched_rl]
		else:
			S850_matched_rl = S850_rand[idx_matched_rl]
			comp_matched_rl = comp_submm[idx_matched_rl]

		#construct the differential number counts
		N_rl, eN_rl_lo, eN_rl_hi, counts_rl, weights_rl = nc.differential_numcounts(
			S850_matched_rl,
			S850_bin_edges,
			A_rl,
			comp=comp_matched_rl,
			incl_poisson=True)

		#remove any bins with 0 sources
		has_sources = N_rl > 0.
		#has_sources = np.full(len(S850_bin_centres), True)
		x_bins = S850_bin_centres[has_sources]
		eN_rl_lo = eN_rl_lo[has_sources]
		eN_rl_hi = eN_rl_hi[has_sources]
		N_rl = N_rl[has_sources]

		if combined_plot:
			ax4.plot(x_bins, N_rl, color=ps.grey, alpha=0.2)

		#offset at which the points will be plotted relative to the bin centre (to avoid overlapping error bars)
		x_bins = 10. ** (np.log10(x_bins) + 0.004 * ((-1.) ** ((j+1) % 2.)) * np.floor((j + 2.) / 2.))
		#x_bins += offset
		
		#plot the bin heights at the bin centres
		ax1[row_z,col_z].plot(x_bins, N_rl, marker=mkr_now, color=c_now, label=ID, ms=9., alpha=0.7, linestyle='none')
		#add errorbars
		eN_rl_lo[eN_rl_lo == N_rl] *= 0.999		#prevents errors when logging plot
		ax1[row_z,col_z].errorbar(x_bins, N_rl, fmt='none', yerr=(eN_rl_lo,eN_rl_hi), ecolor=c_now, alpha=0.7)

		#construct the cumulative number counts if told to do so
		if plot_cumulative:
			cumN_rl, ecumN_rl_lo, ecumN_rl_hi, cumcounts_rl = nc.cumulative_numcounts(
				counts=counts_rl,
				A=A_rl,
				)
			if combined_plot:
				ax5.plot(S850_bin_edges[:-1], cumN_rl, color=ps.grey, alpha=0.2)

			#plot the bin heights at the left bin edges
			x_bins = 10. ** (np.log10(S850_bin_edges[:-1]) + 0.004 * ((-1.) ** ((j+1) % 2.)) * np.floor((j + 2.) / 2.))
			ax3[row_z,col_z].plot(x_bins, cumN_rl, marker=mkr_now, color=c_now, label=ID, ms=9., alpha=0.7, linestyle='none')
			#add errorbars
			ecumN_rl_lo[ecumN_rl_lo == cumN_rl] *= 0.999		#prevents errors when logging plot
			ax3[row_z,col_z].errorbar(x_bins, cumN_rl, fmt='none', yerr=(ecumN_rl_lo,ecumN_rl_hi), ecolor=c_now, alpha=0.7)

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
		A_zbin = ast.apertures_area(coords_rq_zbin, r=R_deg, save_fig=True, figname=PATH_PLOTS+f'Aperture_positions_zbin{i+1}.png')

	#retrieve the flux densities and completenesses for the matched sources
	if ndim == 2:
		S850_matched_zbin = S850_rand[:,idx_matched_zbin]
		comp_matched_zbin = comp_submm[:,idx_matched_zbin]
	else:
		S850_matched_zbin = S850_rand[idx_matched_zbin]
		comp_matched_zbin = comp_submm[idx_matched_zbin]

	#construct the differential number counts
	N_zbin, eN_zbin_lo, eN_zbin_hi, counts_zbin, weights_zbin = nc.differential_numcounts(
		S850_matched_zbin,
		S850_bin_edges,
		A_zbin,
		comp=comp_matched_zbin,
		incl_poisson=True)

	#remove any bins with 0 sources
	has_sources = N_zbin > 0.
	#has_sources = np.full(len(S850_bin_centres), True)
	x_bins = S850_bin_centres[has_sources]
	eN_zbin_lo = eN_zbin_lo[has_sources]
	eN_zbin_hi = eN_zbin_hi[has_sources]
	N_zbin = N_zbin[has_sources]

	#plot the bin heights at the bin centres
	ax1[row_z,col_z].plot(x_bins, N_zbin, marker='o', color='k', label='All', ms=14., linestyle='none')
	#add errorbars
	eN_zbin_lo[eN_zbin_lo == N_zbin] *= 0.999		#prevents errors when logging plot
	ax1[row_z,col_z].errorbar(x_bins, N_zbin, fmt='none', yerr=(eN_zbin_lo,eN_zbin_hi), ecolor='k', elinewidth=2.4)

	#set the axes to log scale
	ax1[row_z,col_z].set_xscale('log')
	ax1[row_z,col_z].set_yscale('log')

	#add text to the top right corner displaying the redshift bin
	bin_text = r'$%.1f \leq z < %.1f$'%(z-dz/2.,z+dz/2.)
	ax1[row_z,col_z].text(0.95, 0.95, bin_text, transform=ax1[row_z,col_z].transAxes, ha='right', va='top')

	#set the minor tick locations on the x-axis
	ax1[row_z,col_z].set_xticks(xtick_min_locs, labels=xtick_min_labels, minor=True)

	#set the axes limits
	ax1[row_z,col_z].set_xlim(1.5, 20.)
	ax1[row_z,col_z].set_ylim(0.1, 1300.)
	#force matplotlib to label with the actual numbers
	#ax1[row_z,col_z].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	#ax1[row_z,col_z].get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax1[row_z,col_z].get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
	ax1[row_z,col_z].get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:g}'))

	labels_ord.insert(0, 'All')
	#add a legend in the bottom left corner, removing duplicate labels
	handles, labels = ax1[row_z,col_z].get_legend_handles_labels()
	labels_ord = [s for s in labels_ord if s in labels]
	by_label = dict(zip(labels, handles))
	ax1[row_z,col_z].legend([by_label[i] for i in labels_ord], [i for i in labels_ord], loc=3)


	if plot_cumulative:
		#construct the cumulative number counts
		cumN_zbin, ecumN_zbin_lo, ecumN_zbin_hi, cumcounts_zbin = nc.cumulative_numcounts(
			counts=counts_zbin,
			A=A_zbin,
			)
		if combined_plot:
			ax5.plot(S850_bin_edges[:-1], cumN_zbin, color=ps.grey, alpha=0.2)

		#plot the bin heights at the left bin edges
		x_bins = S850_bin_edges[:-1]
		ax3[row_z,col_z].plot(x_bins, cumN_zbin, marker='o', color='k', label='All', ms=14., linestyle='none')
		#add errorbars
		ecumN_zbin_lo[ecumN_zbin_lo == cumN_zbin] *= 0.999		#prevents errors when logging plot
		ax3[row_z,col_z].errorbar(x_bins, cumN_zbin, fmt='none', yerr=(ecumN_zbin_lo,ecumN_zbin_hi), ecolor='k', elinewidth=2.4)

		#set the axes to log scale
		ax3[row_z,col_z].set_xscale('log')
		ax3[row_z,col_z].set_yscale('log')

		#add text to the top right corner displaying the redshift bin
		ax3[row_z,col_z].text(0.95, 0.95, bin_text, transform=ax3[row_z,col_z].transAxes, ha='right', va='top')

		#set the minor tick locations on the x-axis
		ax3[row_z,col_z].set_xticks(xtick_min_locs, labels=xtick_min_labels, minor=True)

		#set the axes limits
		ax3[row_z,col_z].set_xlim(1.5, 20.)
		ax3[row_z,col_z].set_ylim(0.1, 4000.)
		#force matplotlib to label with the actual numbers
		#ax3[row_z,col_z].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
		#ax3[row_z,col_z].get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
		ax3[row_z,col_z].get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
		ax3[row_z,col_z].get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:g}'))

		#add a legend in the bottom left corner, removing duplicate labels
		handles, labels = ax3[row_z,col_z].get_legend_handles_labels()
		labels_ord = [s for s in labels_ord if s in labels]
		by_label = dict(zip(labels, handles))
		ax3[row_z,col_z].legend([by_label[i] for i in labels_ord], [i for i in labels_ord], loc=3)

	#perform whichever steps from above are relevant for the positions plot if created
	if plot_positions:
		#add text to the top right corner displaying the redshift bin
		ax2[row_z,col_z].text(0.95, 0.95, bin_text, transform=ax2[row_z,col_z].transAxes, ha='right', va='top')
		#set the axes limits
		ax2[row_z,col_z].set_xlim(150.9, 149.3)
		ax2[row_z,col_z].set_ylim(1.5, 3.)
		#add a legend in the bottom left corner, removing duplicate labels
		handles, labels = ax2[row_z,col_z].get_legend_handles_labels()
		by_label = dict(zip(labels, handles))
		ax2[row_z,col_z].legend(by_label.values(), by_label.keys(), loc=3)

if combined_plot:
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
		A_ALL = ast.apertures_area(coords_rq_sub, r=R_deg, save_fig=True, figname=PATH_PLOTS+'Aperture_positions_all_RQ_galaxies.png')

	#retrieve the flux densities and completenesses for the matched sources
	if ndim == 2:
		S850_matched_ALL = S850_rand[:,idx_matched_ALL]
		comp_matched_ALL = comp_submm[:,idx_matched_ALL]
	else:
		S850_matched_ALL = S850_rand[idx_matched_ALL]
		comp_matched_ALL = comp_submm[idx_matched_ALL]

	#construct the differential number counts
	N_ALL, eN_ALL_lo, eN_ALL_hi, counts_ALL, weights_ALL = nc.differential_numcounts(
		S850_matched_ALL,
		S850_bin_edges,
		A_ALL,
		comp=comp_matched_ALL,
		incl_poisson=True)

	#remove any bins with 0 sources
	has_sources = N_ALL > 0.
	#has_sources = np.full(len(S850_bin_centres), True)
	x_bins = S850_bin_centres[has_sources]
	eN_ALL_lo = eN_ALL_lo[has_sources]
	eN_ALL_hi = eN_ALL_hi[has_sources]
	N_ALL = N_ALL[has_sources]

	#plot the bin heights at the bin centres
	ax4.plot(x_bins, N_ALL, marker='o', color='k', label=label_combined, ms=11., linestyle='none')
	#add errorbars
	eN_ALL_lo[eN_ALL_lo == N_ALL] *= 0.999		#prevents errors when logging plot
	ax4.errorbar(x_bins, N_ALL, fmt='none', yerr=(eN_ALL_lo,eN_ALL_hi), ecolor='k', elinewidth=2.4)

	#set the axes to log scale
	ax4.set_xscale('log')
	ax4.set_yscale('log')

	#set the minor tick locations on the x-axis
	ax4.set_xticks(xtick_min_locs, labels=xtick_min_labels, minor=True)

	#set the axes limits
	ax4.set_xlim(1.5, 20.)
	ax4.set_ylim(0.1, 1300.)
	#force matplotlib to label with the actual numbers
	#ax4.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	#ax4.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax4.get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
	ax4.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:g}'))

	#add a legend in the bottom left corner, removing duplicate labels
	handles, labels = ax4.get_legend_handles_labels()
	labels_ord = [s for s in labels_ord_combined if s in labels]
	by_label = dict(zip(labels, handles))
	ax4.legend([by_label[i] for i in labels_ord_combined], [i for i in labels_ord_combined], loc=3)

	if plot_cumulative:
		#construct the cumulative number counts
		cumN_ALL, ecumN_ALL_lo, ecumN_ALL_hi, cumcounts_ALL = nc.cumulative_numcounts(
			counts=counts_ALL,
			A=A_ALL,
			)

		#remove any bins with 0 sources
		has_sources = cumN_ALL > 0.
		#has_sources = np.full(len(S850_bin_centres), True)
		x_bins = S850_bin_edges[:-1][has_sources]
		ecumN_ALL_lo = ecumN_ALL_lo[has_sources]
		ecumN_ALL_hi = ecumN_ALL_hi[has_sources]
		cumN_ALL = cumN_ALL[has_sources]

		#plot the bin heights at the bin centres
		ax5.plot(x_bins, cumN_ALL, marker='o', color='k', label=label_combined, ms=11., linestyle='none')
		#add errorbars
		eN_ALL_lo[eN_ALL_lo == N_ALL] *= 0.999		#prevents errors when logging plot
		ax5.errorbar(x_bins, cumN_ALL, fmt='none', yerr=(ecumN_ALL_lo,ecumN_ALL_hi), ecolor='k', elinewidth=2.4)

		#set the axes to log scale
		ax5.set_xscale('log')
		ax5.set_yscale('log')

		#set the minor tick locations on the x-axis
		ax5.set_xticks(xtick_min_locs, labels=xtick_min_labels, minor=True)

		#set the axes limits
		ax5.set_xlim(1.5, 20.)
		ax5.set_ylim(0.1, 4000.)
		#force matplotlib to label with the actual numbers
		#ax5.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
		#ax5.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
		ax5.get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
		ax5.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:g}'))

		#add a legend in the bottom left corner, removing duplicate labels
		handles, labels = ax5.get_legend_handles_labels()
		labels_ord = [s for s in labels_ord_combined if s in labels]
		by_label = dict(zip(labels, handles))
		ax5.legend([by_label[i] for i in labels_ord_combined], [i for i in labels_ord_combined], loc=3)

print(gen.colour_string('Done!', 'purple'))

#####################################
#### BINNING RL GALAXIES BY MASS ####
#####################################

if bin_by_mass:
	#create redshift bins for the RL galaxies
	Mbin_edges = np.array([11., 11.2, 11.4, 11.7])
	Mbin_centres = (Mbin_edges[:-1] + Mbin_edges[1:]) / 2.
	Mbin_widths = np.diff(Mbin_edges)

	print_str = 'Constructing number counts (mass-binned)...'
	print(gen.colour_string(print_str, 'purple'))

	#cycle through the redshift bins
	for i in range(len(Mbin_centres)):
		#get the current redshift bin and print its bounds
		logM = Mbin_centres[i]
		dlogM = Mbin_widths[i]
		print(f'{logM-dlogM/2:.2f} < log(M/Msun) < {logM+dlogM/2.:.2f}')

		#get the IDs of all RAGERS sources in the current bin
		mass_mask = (RAGERS_Ms >= (logM - dlogM / 2.)) * (RAGERS_Ms < (logM + dlogM / 2.))
		rl_Mbin = RAGERS_IDs[mass_mask]

		##############################
		#### RQ GALAXIES PER ZBIN ####
		##############################

		#get the corresponding radio-quiet source data
		mass_mask_rq = (data_rq_sub['RAGERS_logMstar'] >= (logM - dlogM / 2.)) * (data_rq_sub['RAGERS_logMstar'] < (logM + dlogM / 2.))
		data_rq_Mbin = data_rq_sub[mass_mask_rq]
		#get the corresponding SkyCoords
		coords_rq_Mbin = coords_rq_sub[mass_mask_rq]

		#create an empty list to which indices of matched submm sources will be appended for each RQ galaxy in this zbin
		idx_matched_Mbin = []

		###################################
		#### RQ GALAXIES PER RL GALAXY ####
		###################################

		#list to which the legend labels will be appended in the desired order
		labels_ord = []
		#blank field  results
		if bf_results:
			labels_ord.append(label_bf)
			ax6[i].plot(xbins_bf, N_bf, linestyle='none', marker='D', color=ps.grey, label=label_bf, alpha=0.5)
			ax6[i].errorbar(xbins_bf, N_bf, fmt='none', yerr=(eN_bf_lo,eN_bf_hi), ecolor=ps.grey, elinewidth=2., alpha=0.5)
			if plot_cumulative:
				ax7[i].plot(xbins_cumbf, cumN_bf, linestyle='none', marker='D', color=ps.grey, label=label_bf, alpha=0.5)
				ax7[i].errorbar(xbins_cumbf, cumN_bf, fmt='none', yerr=(ecumN_bf_lo,ecumN_bf_hi), ecolor=ps.grey, elinewidth=2., alpha=0.5)

		#cycle through the RL galaxies in this redshift bin
		for j in range(len(rl_Mbin)):
			#get the RL ID
			ID = rl_Mbin[j]
			labels_ord.insert(-1, ID)
			#select the colour and marker to use for this dataset
			c_now = gen.scale_RGB_colour(cmap((j+1.)/len(rl_Mbin))[:-1], scale_l=0.8)
			mkr_now = markers[j]

			#select the RQ galaxies corresponding to this RL source
			mask_rq_rl = data_rq_Mbin['RAGERS_ID'] == ID
			data_rq_rl = data_rq_Mbin[mask_rq_rl]
			#get the SkyCoords for these objects
			coords_rq_rl = coords_rq_Mbin[mask_rq_rl]

			#create an empty list to which indices of matched submm sources will be appended for each RQ galaxy corresponding to the current RL galaxy
			idx_matched_rl = []

			################################
			#### INDIVIDUAL RQ GALAXIES ####
			################################

			#cycle through the RQ galaxies matched to this RL galaxy
			for k in range(len(data_rq_rl)):
				#get the coordinates for the current RQ galaxy
				coord_central = coords_rq_rl[k:k+1]

				#search for submm sources within R_arcmin of the galaxy
				idx_coords_submm_matched, *_ = coord_central.search_around_sky(coords_submm, R_arcmin * u.arcmin)
				#append these indices to lists for (a) each RL galaxy, (b) each mass bin
				idx_matched_rl.append(idx_coords_submm_matched)
				idx_matched_Mbin.append(idx_coords_submm_matched)


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
				comp_matched_rl = comp_submm[:,idx_matched_rl]
			else:
				S850_matched_rl = S850_rand[idx_matched_rl]
				comp_matched_rl = comp_submm[idx_matched_rl]

			#construct the differential number counts
			N_rl, eN_rl_lo, eN_rl_hi, counts_rl, weights_rl = nc.differential_numcounts(
				S850_matched_rl,
				S850_bin_edges,
				A_rl,
				comp=comp_matched_rl,
				incl_poisson=True)

			#remove any bins with 0 sources
			has_sources = N_rl > 0.
			#has_sources = np.full(len(S850_bin_centres), True)
			x_bins = S850_bin_centres[has_sources]
			eN_rl_lo = eN_rl_lo[has_sources]
			eN_rl_hi = eN_rl_hi[has_sources]
			N_rl = N_rl[has_sources]

			#offset at which the points will be plotted relative to the bin centre (to avoid overlapping error bars)
			x_bins = 10. ** (np.log10(x_bins) + 0.004 * ((-1.) ** ((j+1) % 2.)) * np.floor((j + 2.) / 2.))
			#x_bins += offset
			
			#plot the bin heights at the bin centres
			ax6[i].plot(x_bins, N_rl, marker=mkr_now, color=c_now, label=ID, ms=9., alpha=0.7, linestyle='none')
			#add errorbars
			eN_rl_lo[eN_rl_lo == N_rl] *= 0.999		#prevents errors when logging plot
			ax6[i].errorbar(x_bins, N_rl, fmt='none', yerr=(eN_rl_lo,eN_rl_hi), ecolor=c_now, alpha=0.7)

			#construct the cumulative number counts if told to do so
			if plot_cumulative:
				cumN_rl, ecumN_rl_lo, ecumN_rl_hi, cumcounts_rl = nc.cumulative_numcounts(
					counts=counts_rl,
					A=A_rl,
					)

				#plot the bin heights at the left bin edges
				x_bins = 10. ** (np.log10(S850_bin_edges[:-1]) + 0.004 * ((-1.) ** ((j+1) % 2.)) * np.floor((j + 2.) / 2.))
				ax7[i].plot(x_bins, cumN_rl, marker=mkr_now, color=c_now, label=ID, ms=9., alpha=0.7, linestyle='none')
				#add errorbars
				ecumN_rl_lo[ecumN_rl_lo == cumN_rl] *= 0.999		#prevents errors when logging plot
				ax7[i].errorbar(x_bins, cumN_rl, fmt='none', yerr=(ecumN_rl_lo,ecumN_rl_hi), ecolor=c_now, alpha=0.7)

		#concatenate the arrays of indices of matched submm sources for this z bin
		idx_matched_Mbin = np.concatenate(idx_matched_Mbin)
		
		#if each aperture is independent, the area is simply the sum of their areas
		if independent_rq:
			#calculate the total area surveyed for this RL galaxy
			A_Mbin = A_sqdeg * len(data_rq_Mbin)
		#if not treating each aperture as independent, remove duplicate matches from the list for this RL galaxy
		if not independent_rq:
			idx_matched_Mbin = np.unique(idx_matched_Mbin)
			#calculate the area covered, accounting for overlap between apertures
			A_Mbin = ast.apertures_area(coords_rq_Mbin, r=R_deg, save_fig=True, figname=PATH_PLOTS+f'Aperture_positions_Mbin{i+1}.png')

		#retrieve the flux densities and completenesses for the matched sources
		if ndim == 2:
			S850_matched_Mbin = S850_rand[:,idx_matched_Mbin]
			comp_matched_Mbin = comp_submm[:,idx_matched_Mbin]
		else:
			S850_matched_Mbin = S850_rand[idx_matched_Mbin]
			comp_matched_Mbin = comp_submm[idx_matched_Mbin]

		#construct the differential number counts
		N_Mbin, eN_Mbin_lo, eN_Mbin_hi, counts_Mbin, weights_Mbin = nc.differential_numcounts(
			S850_matched_Mbin,
			S850_bin_edges,
			A_Mbin,
			comp=comp_matched_Mbin,
			incl_poisson=True)

		#remove any bins with 0 sources
		has_sources = N_Mbin > 0.
		#has_sources = np.full(len(S850_bin_centres), True)
		x_bins = S850_bin_centres[has_sources]
		eN_Mbin_lo = eN_Mbin_lo[has_sources]
		eN_Mbin_hi = eN_Mbin_hi[has_sources]
		N_Mbin = N_Mbin[has_sources]

		#plot the bin heights at the bin centres
		ax6[i].plot(x_bins, N_Mbin, marker='o', color='k', label='All', ms=14., linestyle='none')
		#add errorbars
		eN_Mbin_lo[eN_Mbin_lo == N_Mbin] *= 0.999		#prevents errors when logging plot
		ax6[i].errorbar(x_bins, N_Mbin, fmt='none', yerr=(eN_Mbin_lo,eN_Mbin_hi), ecolor='k', elinewidth=2.4)

		#set the axes to log scale
		ax6[i].set_xscale('log')
		ax6[i].set_yscale('log')

		#add text to the top right corner displaying the redshift bin
		bin_text = f'{logM-dlogM/2:.1f}' + r' $< \log({\rm M}/M_{\odot}) <$ ' + f'{logM+dlogM/2.:.1f}'
		ax6[i].text(0.95, 0.95, bin_text, transform=ax6[i].transAxes, ha='right', va='top')

		#set the minor tick locations on the x-axis
		ax6[i].set_xticks(xtick_min_locs, labels=xtick_min_labels, minor=True)

		#set the axes limits
		ax6[i].set_xlim(1.5, 20.)
		ax6[i].set_ylim(0.1, 1300.)
		#force matplotlib to label with the actual numbers
		#ax6[i].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
		#ax6[i].get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
		ax6[i].get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
		ax6[i].get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:g}'))

		labels_ord.insert(0, 'All')
		#add a legend in the bottom left corner, removing duplicate labels
		handles, labels = ax6[i].get_legend_handles_labels()
		labels_ord = [s for s in labels_ord if s in labels]
		by_label = dict(zip(labels, handles))
		ax6[i].legend([by_label[i] for i in labels_ord], [i for i in labels_ord], loc=3)


		if plot_cumulative:
			#construct the cumulative number counts
			cumN_Mbin, ecumN_Mbin_lo, ecumN_Mbin_hi, cumcounts_Mbin = nc.cumulative_numcounts(
				counts=counts_Mbin,
				A=A_Mbin,
				)
			#plot the bin heights at the left bin edges
			x_bins = S850_bin_edges[:-1]
			ax7[i].plot(x_bins, cumN_Mbin, marker='o', color='k', label='All', ms=14., linestyle='none')
			#add errorbars
			ecumN_Mbin_lo[ecumN_Mbin_lo == cumN_Mbin] *= 0.999		#prevents errors when logging plot
			ax7[i].errorbar(x_bins, cumN_Mbin, fmt='none', yerr=(ecumN_Mbin_lo,ecumN_Mbin_hi), ecolor='k', elinewidth=2.4)

			#set the axes to log scale
			ax7[i].set_xscale('log')
			ax7[i].set_yscale('log')

			#add text to the top right corner displaying the redshift bin
			ax7[i].text(0.95, 0.95, bin_text, transform=ax7[i].transAxes, ha='right', va='top')

			#set the minor tick locations on the x-axis
			ax7[i].set_xticks(xtick_min_locs, labels=xtick_min_labels, minor=True)

			#set the axes limits
			ax7[i].set_xlim(1.5, 20.)
			ax7[i].set_ylim(0.1, 4000.)
			#force matplotlib to label with the actual numbers
			#ax7[i].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
			#ax7[i].get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
			ax7[i].get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
			ax7[i].get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:g}'))

			#add a legend in the bottom left corner, removing duplicate labels
			handles, labels = ax7[i].get_legend_handles_labels()
			labels_ord = [s for s in labels_ord if s in labels]
			by_label = dict(zip(labels, handles))
			ax7[i].legend([by_label[ii] for ii in labels_ord], [ii for ii in labels_ord], loc=3)

###########################################
#### FINAL FORMATTING & SAVING FIGURES ####
###########################################

#suffix to use for figure based on the operations performed
suffix = ''
if not independent_rq:
	suffix += '_overlap'
if randomise_fluxes:
	suffix += '_randf'
if comp_correct:
	suffix += '_cc'

#add the search radius to the file name
suffix += f'_{R_arcmin:.1f}am'

#minimise unnecesary whitespace
f1.tight_layout()
figname = PATH_PLOTS + f'S850_number_counts_zbinned{suffix}.png'
f1.savefig(figname, bbox_inches='tight', dpi=300)

if plot_positions:
	#minimise unnecesary whitespace
	f2.tight_layout()
	figname = PATH_PLOTS + 'RQ_positions_with_search_areas.png'
	f2.savefig(figname, bbox_inches='tight', dpi=300)

if plot_cumulative:
	#minimise unnecesary whitespace
	f3.tight_layout()
	figname = PATH_PLOTS + f'S850_cumulative_counts_zbinned{suffix}.png'
	f3.savefig(figname, bbox_inches='tight', dpi=300)

if combined_plot:
	#minimise unnecesary whitespace
	f4.tight_layout()
	figname = PATH_PLOTS + f'S850_number_counts_ALL{suffix}.png'
	f4.savefig(figname, bbox_inches='tight', dpi=300)

	if plot_cumulative:
		#minimise unnecesary whitespace
		f5.tight_layout()
		figname = PATH_PLOTS + f'S850_cumulative_counts_ALL{suffix}.png'
		f5.savefig(figname, bbox_inches='tight', dpi=300)

if bin_by_mass:
	#minimise unnecesary whitespace
	f6.tight_layout()
	figname = PATH_PLOTS + f'S850_number_counts_Mbinned{suffix}.png'
	f6.savefig(figname, bbox_inches='tight', dpi=300)

	if plot_cumulative:
		#minimise unnecesary whitespace
		f7.tight_layout()
		figname = PATH_PLOTS + f'S850_cumulative_counts_Mbinned{suffix}.png'
		f7.savefig(figname, bbox_inches='tight', dpi=300)

print(gen.colour_string('Done!', 'purple'))












