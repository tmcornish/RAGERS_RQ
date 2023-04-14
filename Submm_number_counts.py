############################################################################################################
# A script for calculating and plotting the sub-mm number counts in the environments of 10 radio-quiet
# galaxies for each radio-loud galaxy. The radio-quiet sample may have include galaxies that are 
# 'double-counted', i.e. they have been matched to more than one radio-loud galaxy; this script accounts
# for that when selecting the 10 galaxies to use for each set of number counts. Does not apply completeness
# corrections at this stage.
#
# v2: Also plots number counts for all sources in each redshift bin.
# v3: Adds the option to create an additional plot showing the RAs and DECs of each RQ galaxy with its 
# search radius.
# v4: Fixed bug - script now actually ensures the same RQ galaxy is not selected for multiple RL galaxies.
# v5: Adapted to account for overlapping areas when placing apertures around each RQ galaxy.
# v6: Derives completeness corrections by recreating Fig. 6b from Simpson+19 and applies them to the number 
# counts.
# v7: Plots cumulative number counts as well as differential number counts. Also added previously omitted 
# step in the completeness calculation in which the defined completeness range is broadened to 0–100%.
# v8: Creates plot(s) showing the combined results for all RQ targets.
# v9: Adds the blank-field results to each relevant plot, where these are obtained using the whole S2COSMOS
# catalogue.
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
import my_functions as mf
import colorsys
import glob
from scipy.interpolate import LinearNDInterpolator, interp1d
import scipy.optimize as opt

#######################################################################################
########### FORMATTING FOR GRAPHS #####################################################

#formatting for graphs
plt.style.use('plotstyle.mplstyle')

#colours
red = '#eb0505'
dark_red = '#ab0000'
ruby = '#C90058'
crimson = '#AF0404'
coral = '#FF4848'
magenta = '#C3027D'
orange = '#ED5A01'
green = '#0A8600'
light_green = '#11C503'
teal = '#00A091'
cyan = '#00d0f0'
blue = '#0066ff'
light_blue = '#00C2F2'
dark_blue = '#004ab8'
purple = '#6D04C4'
lilac = '#EB89FF'
plum = '#862388'
pink = '#E40ACA'
baby_pink = '#FF89FD'
fuchsia = '#E102B5'
grey = '#969696'

#obtain the figure size in inches
x_size, y_size = 8., 8.
#formatting for any arrows to be added to the plot for representing upper/lower limits
ax_frac = 1/40.				#the fraction of the y axis that the total length of a vertical arrow should occupy
al = ax_frac * y_size		#the length of each arrow in inches (ew, but sadly metric isn't allowed)
scale = 1./al				#'scale' parameter used for defining the length of each arrow in a quiver
aw = 0.0175 * al				#the width of each arrow shaft in inches
hw = 4.						#width of the arrowheads in units of shaft width
hl = 3.						#length of the arrowheads in units of shaft width
hal = 2.5					#length of the arrowheads at the point where they intersect the shaft 
							#(e.g. hal = hl gives a triangular head, hal < hl gives a more pointed head)


#######################################################
###################    FUNCTIONS  #####################
#######################################################

def area_of_intersection(r_a, r_b, d):
	'''
	Calculates the intersecting area between two overlappig circles.
		r_a: Radius of the first circle.
		r_b: Radius of the second circle.
		d: Distance between the centres of the two circles.
	'''
	#see which circle has the smaller radius (r1 = bigger circle, r2 = smaller circle)
	if r_a >= r_b:
		r1 = r_a
		r2 = r_b
	else:
		r1 = r_b
		r2 = r_a
	#check if the distance is greater than the sum of the radii - if so then there is no intersection
	if d > (r1 + r2):
		return 0.
	#if the distance is less than the difference between the two radii, the area is equal to that of the smaller circle
	if d <= (r1 - r2):
		return np.pi * (r2 ** 2.)

	#otherwise, the area of intersection is non-trivial and must be calculated differently; see
	#https://diego.assencio.com/?index=8d6ca3d82151bad815f78addf9b5c1c6 for derivation and notation
	d1 = ((r1 ** 2.) - (r2 ** 2.) + (d ** 2.)) / (2. * d)
	d2 = d - d1
	A1 = (r1 ** 2.) * np.arccos(d1 / r1) - d1 * np.sqrt((r1 ** 2.) - (d1 ** 2.))
	A2 = (r2 ** 2.) * np.arccos(d2 / r2) - d2 * np.sqrt((r2 ** 2.) - (d2 ** 2.))
	return A1 + A2


def scale_colour(rgb, scale_l=1., scale_s=1.):
	'''
	Takes an rgb colour and scales its 'lightness' (according to the hls colour model) by a factor
	of scale_l.
		rgb: The (R, G, B) colour specifications.
		scale_l: The factor by which to scale the lightness.
	'''
	#convert the rgb to hls (hue, lightness, saturation)
	h, l, s = colorsys.rgb_to_hls(*rgb)
	#scale the lightness and ensure the result is between 0 and 1
	l_new = max(0, min(1, l * scale_l))
	s_new = max(0, min(1, s * scale_s))
	#convert back to rgb and return the result
	return colorsys.hls_to_rgb(h, l_new, s_new)

def model(theta, x):
	'''
	Fermi-Dirac-like distribution to model a completeness curve.
		theta: fit parameters defining the slope and the x-position at which the completeness = 0.5.
		x: values (of flux density) at which the distribution is analysed.
	'''
	A, B = theta
	return 1. / (np.exp(A * (-x + B)) + 1.)

def chisq(theta, x, y, yerr):
	'''
	Calculates the chi-squared value by comparing a model to the observed values.
		theta: model fit parameters.
		x: x-values at which the observed y-values are taken.
		y: observed y values.
		yerr: uncertainties in the observed y-values.
	'''
	ymodel = model(theta, x)
	return np.sum(((ymodel - y)/yerr) ** 2.)

def nll(*args):
	'''
	Chi-squared function reformatted to be compatible with scipy.optimize.minimize.
	'''
	return chisq(*args)


##################
#### SETTINGS ####
##################

#toggle `switches' for additional functionality
plot_positions = True			#plot positions of selected RQ galaxies with their search areas
independent_rq = True			#treat the RQ galaxies as independent (i.e. do not account for overlap between search areas)
comp_correct = True				#apply completeness corrections
plot_cumulative = True			#plot cumulative number counts as well as differential
combined_plot = True			#create plot(s) summarising the results from all targets
bf_results = True				#adds the results obtained from the whole S2COSMOS catalogue
settings = [plot_positions, independent_rq, comp_correct, plot_cumulative, combined_plot, bf_results]

#print the chosen settings to the Terminal
print(mf.colour_string('CHOSEN SETTINGS:', 'white'))
settings_print = [
	'Plot RQ galaxy positions: ',
	'Treat RQ galaxies independently: ',
	'Apply completeness corrections: ',
	'Plot cumulative number counts: ',
	'Combine all targets into one plot: ',
	'Plot blank-field results: ']
for i in range(len(settings_print)):
	if settings[i]:
		settings_print[i] += 'y'
	else:
		settings_print[i] += 'n'
print(mf.colour_string('\n'.join(settings_print), 'white'))

#850 micron flux density bins to use for submm number counts
dS = 2.		#bin width (mJy)
S850_bin_centres = np.arange(2.5, 22.5, dS)
S850_bin_edges = np.append(S850_bin_centres-(dS/2.), S850_bin_centres[-1]+(dS/2.))

#radius of search area to use in the submm data
R_arcmin = 6.
R_deg = R_arcmin / 60.
#area of aperture (sq. deg)
A_sqdeg = np.pi * R_deg ** 2.

#number of matched galaxies to use per RL galaxy when constructing the number counts
N_sel = 10

#######################################################
###############    START OF SCRIPT    #################
#######################################################

#relevant paths
PATH_RAGERS = sys.argv[1]
PATH_CATS = sys.argv[2]
PATH_PLOTS = sys.argv[3]

#catalogue containing data for (radio-quiet) galaxies from COSMOS2020 matched in M* and z with the radio-loud sample
RQ_CAT = PATH_CATS + 'RAGERS_COSMOS2020_matches_Mstar_z.fits'
data_rq = Table.read(RQ_CAT, format='fits')
#get the RAs and DECs
RA_rq_all = data_rq['ALPHA_J2000']
DEC_rq_all = data_rq['DELTA_J2000']
#convert these into SkyCoord objects
coords_rq_all = SkyCoord(RA_rq_all, DEC_rq_all, unit='deg')

#catalogue containing submm data for S2COSMOS sources
SUBMM_CAT = PATH_CATS + 'S2COSMOS_sourcecat850_Simpson18.fits'
data_submm = Table.read(SUBMM_CAT, format='fits')
#get the RAs and DECs
RA_submm = data_submm['RA_deg']
DEC_submm = data_submm['DEC_deg']
#convert these into SkyCoord objects
coords_submm = SkyCoord(RA_submm, DEC_submm, unit='deg')

######################
#### FIGURE SETUP ####
######################

#create the figure (dN/dS vs S)
f1, ax1 = plt.subplots(2, 2, figsize=(2.*x_size, 2.*y_size))
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
	f2, ax2 = plt.subplots(2, 2, figsize=(2.*x_size, 2.*y_size))
	#to label axes with common labels, create a big subplot, make it invisible, and label its axes
	ax_big2 = f2.add_subplot(111, frameon=False)
	ax_big2.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, which='both')
	#label x and y axes
	ax_big2.set_xlabel('RA (deg)', labelpad=10.)
	ax_big2.set_ylabel('Dec. (deg)', labelpad=20.)

if plot_cumulative:
	#create the figure (N(>S) vs S)
	f3, ax3 = plt.subplots(2, 2, figsize=(2.*x_size, 2.*y_size))
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

###########################################################
#### RANDOMLY SELECTING RQ GALAXIES FOR EACH RL GALAXY ####
###########################################################

#retrieve a list of the unique RAGERS IDs from the catalogue and the number of matches
RAGERS_IDs, idx_unique, n_rq_per_rl = np.unique(data_rq['RAGERS_ID'], return_index=True, return_counts=True)
#sort the IDs in order of increasing number of RQ matches
idx_ordered = np.argsort(n_rq_per_rl)
RAGERS_IDs = RAGERS_IDs[idx_ordered]
#also get a list of RAGERS RL redshifts and order them
RAGERS_zs = data_rq['RAGERS_z'][idx_unique]
RAGERS_zs = RAGERS_zs[idx_ordered]

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
	print(mf.colour_string('Calculating completeness corrections...', 'purple'))
	#create a grid in flux density-RMS space
	xmin, xmax = -15., 30.
	ymin, ymax = 0.45, 3.05
	xstep, ystep = 0.01, 0.01
	xgrid, ygrid = np.mgrid[xmin:xmax+xstep:xstep, ymin:ymax+ystep:ystep]
	#flat arrays of intervals in flux density and RMS
	xspace = np.arange(xmin, xmax+xstep, xstep)
	yspace = np.arange(ymin, ymax+ystep, ystep)

	#list to which the flux density, RMS and completeness values will be appended
	values_list = []
	#list to which the interpolated RMS as functions of flux density will be appended
	x_interp_list = []
	y_interp_list = []
	#lists to which interpolated values of x and y will be appended for each completeness curve
	x_interp_vals = []
	y_interp_vals = []
	#completenesses at which the curves are well-defined
	comp_list = []
	#cycle through the files containing data for the completeness curves in Simpson+19
	files = sorted(glob.glob(PATH_CATS+'Simpson+19_completeness_curves/*'))
	for file in files:
		t_now = Table.read(file, format='ascii')
		#interpolate the RMS as a function of flux density
		y_interp = interp1d(t_now['col1'], t_now['col2'], fill_value='extrapolate')
		#reverse the interpolation to get flux density as a function of RMS as well
		x_interp = interp1d(t_now['col2'], t_now['col1'], fill_value='extrapolate')
		#get the completeness from the filename
		comp = float(file[-8:-6]) / 100.
		comp_list.append(comp)
		#use the interpolated functions to calculate the flux density at regular intervals of rms and vice versa
		Y = y_interp(xspace)
		X = x_interp(yspace)
		#append the relevant values/functions to the lists
		values_list.append([xspace, Y, np.full(len(xspace), comp)])
		x_interp_list.append(x_interp)
		y_interp_list.append(y_interp)
		x_interp_vals.append(X)
		y_interp_vals.append(Y)

	#stack the results for each completeness curve
	t_all = np.hstack(values_list)
	#split the array into x-y coordinate pairs and the corresponding z values
	points = t_all[:2]
	values = t_all[-1]
	#interpolate completeness as a function of flux density and RMS
	comp_interp = LinearNDInterpolator(points.T, values)
	#create a grid of completeness values
	zgrid = comp_interp(xgrid, ygrid)

	#uncertainties to give the observed completeness values, arbitrarily chosen such that the 10% and 90% 
	#values have the highest weighting
	comp_err = np.array([0.001, 0.01, 0.01, 0.01, 0.001])
	#completeness values at which new curves are to be estimated
	comp_to_find = np.concatenate(([0.0001], np.arange(0.01, 0.1, 0.01), np.arange(0.91, 1., 0.01), [0.99999]))

	S_comp_to_find = []
	#cycle through values of RMS at which to collapse the 2D completeness function to 1D functions of flux density
	for i in range(len(yspace)):
		rms_now = yspace[i]
		#flux densities at which the completeness curves are well-defined by Simpson et al.
		S_list = np.array([x_interp_list[j](rms_now) for j in range(len(x_interp_list))])
		#fit a curve to the data, weighting the 10% and 90% points more heavily
		initial = [1., S_list[2]]
		popt = opt.minimize(nll, x0=initial, args=(S_list, comp_list, comp_err))['x']
		#create x-ranges at which to plot things
		xrange_lo = xspace[xspace < S_list[0]]
		xrange_mid = xspace[(xspace >= S_list[0]) * (xspace <= S_list[-1])]
		xrange_hi = xspace[xspace > S_list[-1]]
		#evaluate the fitted curve over the low and high x-ranges, and the original completeness curve in the mid range
		x_range = np.concatenate((xrange_lo, xrange_mid, xrange_hi))
		z_range = np.concatenate((model(popt, xrange_lo), comp_interp(xrange_mid, rms_now), model(popt, xrange_hi)))
		#interpolate x w.r.t. z
		S_interp = interp1d(z_range, x_range, fill_value='extrapolate')
		#get the flux densities at which the completenesses 
		S_at_comp = S_interp(comp_to_find)
		S_comp_to_find.append(S_at_comp)

	S_comp_to_find = np.array(S_comp_to_find).T

	#interpolate the flux density w.r.t. rms at each value of completeness
	for i in range(len(comp_to_find)):
		ctf = comp_to_find[i]
		y_interp_new = interp1d(S_comp_to_find[i], yspace, fill_value=(ymin-ystep, ymax+ystep), bounds_error=False)
		Y_new = y_interp_new(xspace)
		if ctf < comp_list[0]:
			y_interp_list.insert(i, y_interp_new)
			values_list.insert(i, [xspace, Y_new, np.full(len(xspace), ctf)])
		else:
			y_interp_list.append(y_interp_new)
			values_list.append([xspace, Y_new, np.full(len(xspace), ctf)])

	#stack the results for each completeness curve
	t_all = np.hstack(values_list)
	#split the array into x-y coordinate pairs and the corresponding z values
	points = t_all[:2]
	values = t_all[-1]
	#interpolate completeness as a function of flux density and RMS
	comp_interp = LinearNDInterpolator(points.T, values)
	#create a grid of completeness values
	zgrid = comp_interp(xgrid, ygrid)

	print(t_all.T)

	#get the interpolated ~100% completeness curve
	interp_100 = y_interp_list[-1]
	#identify all elements in the grid that lie below the 100% completeness curve
	lower_mask = ygrid < interp_100(xgrid)
	#fill these values with 100%
	zgrid[lower_mask] = 1.
	#identify all elements in the grid that lie below a flux density of 0 mJy
	zero_mask = xgrid <= 0.
	#fill these values with 0
	zgrid[zero_mask] = 0.

	#re-interpolate the completeness now that the NaNs have been replaced
	points = np.array([xgrid.flatten(), ygrid.flatten()])
	values = zgrid.flatten()
	comp_interp = LinearNDInterpolator(points.T, values)

	#calculate the completeness at the flux density and RMS of each S2COSMOS source
	comp_submm = comp_interp(data_submm['S_deboost'], data_submm['RMS'])

	print(mf.colour_string('Done!', 'purple'))

#if not told to do completeness corrections, just set the completeness to 1 for everything
else:
	comp_submm = np.ones(len(data_submm))


#############################
#### BLANK-FIELD RESULTS ####
#############################

#generate number counts from the entire S2COSMOS catalogue
if bf_results:
	#label to use for the blank field data points
	label_bf = 'S2COSMOS'

	#bin the sources in the catalogue by their flux densities using the bins defined above
	counts_bf, _ = np.histogram(data_submm['S_deboost'], bins=S850_bin_edges, weights=1./comp_submm)
	#Poissonian uncertainties
	ecounts_bf = np.sqrt(counts_bf)
	#survey area in square degrees
	A_bf = 2.6
	#divide the counts by the area times the width of the bin
	weights = 1. / (A_bf * dS)
	N_bf = counts_bf * weights
	eN_bf = ecounts_bf * weights
	eN_bf[eN_bf == N_bf] *= 0.999		#prevents errors when logging plot
	xbins_bf = S850_bin_centres * 10. ** (0.004)
	if combined_plot:
		labels_ord_combined.append(label_bf)
		ax4.plot(xbins_bf, N_bf, linestyle='none', marker='D', color=crimson, label=label_bf)
		ax4.errorbar(xbins_bf, N_bf, fmt='none', yerr=eN_bf, ecolor=crimson, elinewidth=2.)

	if plot_cumulative:
		#calculate the cumulative counts
		cumcounts_bf = np.cumsum(counts_bf[::-1])[::-1]
		#calculate the Poissonian uncertainties
		ecumcounts_bf = np.sqrt(cumcounts_bf)

		#divide the counts (and uncertainties) by the area
		cumN_bf = cumcounts_bf / A_bf
		ecumN_bf = ecumcounts_bf / A_bf
		ecumN_bf[ecumN_bf == cumN_bf] *= 0.999		#prevents errors when logging plot
		xbins_cumbf = S850_bin_edges[:-1] * 10. ** (0.004)

		if combined_plot:
			ax5.plot(xbins_cumbf, cumN_bf, linestyle='none', marker='D', color=crimson, label=label_bf)
			ax5.errorbar(xbins_cumbf, cumN_bf, fmt='none', yerr=ecumN_bf, ecolor=crimson, elinewidth=2.)


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

print_str = 'Constructing number counts'
if plot_positions:
	print_str += ' and plotting RQ galaxy positions'
print_str += '...'
print(mf.colour_string(print_str, 'purple'))

#set up a float to which the total overlapping area of annuli placed around ALL RQ galaxies will be added
A_overlap_ALL = 0.
#identify all submm sources within the defined search radius from the RQ galaxy
idx_submm_ALL, *_ = coords_rq_sub.search_around_sky(coords_submm, R_arcmin * u.arcmin)
#remove duplicates
idx_submm_ALL = np.unique(idx_submm_ALL)
#create a Table containing this subset of submm sources
data_submm_ALL = data_submm[idx_submm_ALL]
comp_submm_ALL = comp_submm[idx_submm_ALL]
if independent_rq:
	#create an array to which the total counts will be added from all RQ galaxies in the current z bin
	counts_ALL = np.zeros(len(S850_bin_centres))
	#set up an array for recording if a source is multiply-counted in the same redshift bin
	repeated_ALL = np.zeros(len(data_submm), dtype=bool)
else:
	#bin these sources by their 850 micron flux density
	counts_ALL, _ = np.histogram(data_submm_ALL['S_deboost'], bins=S850_bin_edges, weights=1./comp_submm_ALL)

#cycle through the redshift bins
for i in range(len(zbin_centres)):
	#get the current redshift bin and print its bounds
	z = zbin_centres[i]
	print(f'{z-dz/2:.2f} < z < {z+dz/2.:.2f}')

	#plot data in the left-hand column if i is even; plot in the right-hand column if i is odd
	if (i % 2 == 0):
		row_c = int(i/2)			#the row in which the current subplot lies
		col_c = 0					#the column in which the current subplot lies
	else:
		row_c = int((i-1)/2)		#the row in which the current subplot lies
		col_c = 1					#the column in which the current subplot lies

	#set up a float to which the total overlapping area of annuli placed around RQ galaxies in this redshift bin will be added
	A_overlap_zbin = 0.

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

	#identify all submm sources within the defined search radius from the RQ galaxy
	idx_submm_zbin, *_ = coords_rq_zbin.search_around_sky(coords_submm, R_arcmin * u.arcmin)
	#remove duplicates
	idx_submm_zbin = np.unique(idx_submm_zbin)
	#create a Table containing this subset of submm sources
	data_submm_zbin = data_submm[idx_submm_zbin]
	comp_submm_zbin = comp_submm[idx_submm_zbin]
	if independent_rq:
		#create an array to which the total counts will be added from all RQ galaxies in the current z bin
		counts_zbin = np.zeros(len(S850_bin_centres))
		#set up an array for recording if a source is multiply-counted in the same redshift bin
		repeated_zbin = np.zeros(len(data_submm), dtype=bool)
	else:
		#bin these sources by their 850 micron flux density
		counts_zbin, _ = np.histogram(data_submm_zbin['S_deboost'], bins=S850_bin_edges, weights=1./comp_submm_zbin)
	

	###################################
	#### RQ GALAXIES PER RL GALAXY ####
	###################################

	#list to which the legend labels will be appended in the desired order
	labels_ord = []
	#blank field  results
	if bf_results:
		labels_ord.append(label_bf)
		ax1[row_c,col_c].plot(xbins_bf, N_bf, linestyle='none', marker='D', color=grey, label=label_bf, alpha=0.5)
		ax1[row_c,col_c].errorbar(xbins_bf, N_bf, fmt='none', yerr=eN_bf, ecolor=grey, elinewidth=2., alpha=0.5)
		if plot_cumulative:
			ax3[row_c,col_c].plot(xbins_cumbf, cumN_bf, linestyle='none', marker='D', color=grey, label=label_bf, alpha=0.5)
			ax3[row_c,col_c].errorbar(xbins_cumbf, cumN_bf, fmt='none', yerr=ecumN_bf, ecolor=grey, elinewidth=2., alpha=0.5)

	#cycle through the RL galaxies in this redshift bin
	for j in range(len(rl_zbin)):
		#get the RL ID
		ID = rl_zbin[j]
		labels_ord.insert(-1, ID)
		#select the colour and marker to use for this dataset
		c_now = scale_colour(cmap((j+1.)/len(rl_zbin))[:-1], scale_l=0.8)
		mkr_now = markers[j]

		#set up a float to which the total overlapping area of annuli placed around RQ galaxies matched to this RL galaxy will be added
		A_overlap_rl = 0.

		#select the RQ galaxies corresponding to this RL source
		mask_rq_rl = data_rq_zbin['RAGERS_ID'] == ID
		data_rq_rl = data_rq_zbin[mask_rq_rl]
		#get the SkyCoords for these objects
		coords_rq_rl = coords_rq_zbin[mask_rq_rl]

		#identify all submm sources within the defined search radius from the RQ galaxies
		idx_submm_rl, *_ = coords_rq_rl.search_around_sky(coords_submm, R_arcmin * u.arcmin)
		#remove duplicates
		idx_submm_rl = np.unique(idx_submm_rl)
		#create a Table containing this subset of submm sources
		data_submm_rl = data_submm[idx_submm_rl]
		comp_submm_rl = comp_submm[idx_submm_rl]
		if independent_rq:
			#set up array for the total counts in each bin for all N_sel RQ galaxies
			counts_rl = np.zeros(len(S850_bin_centres))
			#set up an array for recording if a source is multiply-counted for the current RL galaxy
			repeated_rl = np.zeros(len(data_submm), dtype=bool)
		else:
			#bin these sources by their 850 micron flux density
			counts_rl, _ = np.histogram(data_submm_rl['S_deboost'], bins=S850_bin_edges, weights=1./comp_submm_rl)
			
		
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
				ax2[row_c,col_c].plot(RA_rq, DEC_rq, linestyle='none', marker=mkr_now, c=c_now, label=ID, alpha=0.7)
				#add circles with radius 4' and 6' centred on the RQ galaxy
				r_circle1 = 4. / 60.
				r_circle2 = 6. / 60.
				f_cosdec = np.cos(DEC_rq * np.pi / 180.)
				ellipse1 = mpl.patches.Ellipse((RA_rq, DEC_rq), width=r_circle1/f_cosdec, height=r_circle1, color=c_now, fill=False, alpha=0.7)
				ellipse2 = mpl.patches.Ellipse((RA_rq, DEC_rq), width=r_circle2/f_cosdec, height=r_circle2, color=c_now, fill=False, linestyle='--', alpha=0.7)
				ax2[row_c,col_c].add_patch(ellipse1)
				ax2[row_c,col_c].add_patch(ellipse2)

			if independent_rq:
				#search for submm sources within 4' of the galaxy
				idx_coords_submm_matched, *_ = coord_central.search_around_sky(coords_submm, R_arcmin * u.arcmin)

				#get the (deboosted) flux densities of the matched sources, ensuring no repeated sources
				matched_mask = np.zeros(len(data_submm), dtype=bool)
				matched_mask[idx_coords_submm_matched] = True
				S850_matched_rl = data_submm['S_deboost'][matched_mask]
				comp_matched_rl = comp_submm[matched_mask]
				#bin the sources by their flux densities
				counts_now, _ = np.histogram(S850_matched_rl, bins=S850_bin_edges, weights=1./comp_matched_rl)
				#add these to the array of counts for all matched RQ galaxies
				counts_rl += counts_now

				#add these to the array of counts for all matched RQ galaxies
				counts_zbin += counts_now

				#add these to the array of counts for ALL RQ galaxies
				counts_ALL += counts_now
			else:
				#get the coordinates of the other RQ galaxies in this subsample
				coords_other_rl = np.delete(coords_rq_rl, k)
				#calculate the separation from the current R1 galaxy in deg
				sep = coord_central.separation(coords_other_rl).value
				#calculate the area of intersection (if any) between this galaxy's search radius and others
				A_inter = [area_of_intersection(R_deg, R_deg, s) for s in sep]
				#sum the intersection areas and divide by two to account for fact that the overlapping area is calculated
				#twice for each pair of intersecting circles
				A_inter = 0.5 * np.sum(A_inter)
				#add this to the total overlapping area for all RQ galaxies matched to this RL galaxy
				A_overlap_rl += A_inter

				#get the coordinates of the other RQ galaxies in this redshift bin
				idx_now = np.where(coords_rq_zbin == coord_central[0])
				coords_other_zbin = np.delete(coords_rq_zbin, idx_now)
				#calculate the separation from the current R1 galaxy in deg
				sep = coord_central.separation(coords_other_zbin).value
				#calculate the area of intersection (if any) between this galaxy's search radius and others
				A_inter = [area_of_intersection(R_deg, R_deg, s) for s in sep]
				#sum the intersection areas and divide by two to account for fact that the overlapping area is calculated
				#twice for each pair of intersecting circles
				A_inter = 0.5 * np.sum(A_inter)
				#add this to the total overlapping area for all RQ galaxies in this z bin
				A_overlap_zbin += A_inter

				#get the coordinates of the other RQ galaxies in this redshift bin
				idx_now = np.where(coords_rq_ALL == coord_central[0])
				coords_other_ALL = np.delete(coords_rq_ALL, idx_now)
				#calculate the separation from the current R1 galaxy in deg
				sep = coord_central.separation(coords_other_ALL).value
				#calculate the area of intersection (if any) between this galaxy's search radius and others
				A_inter = [area_of_intersection(R_deg, R_deg, s) for s in sep]
				#sum the intersection areas and divide by two to account for fact that the overlapping area is calculated
				#twice for each pair of intersecting circles
				A_inter = 0.5 * np.sum(A_inter)
				#add this to the total overlapping area for all RQ galaxies in this z bin
				A_overlap_ALL += A_inter

		#calculate the total area surveyed for this RL galaxy
		A_rl = A_sqdeg * len(data_rq_rl) - A_overlap_rl

		#calculate the Poissonian uncertainties
		ecounts_rl = np.sqrt(counts_rl)

		#divide the counts (and uncertainties) by the area times the width of the bin
		weights = 1. / (A_rl * dS)
		N_rl = counts_rl * weights
		eN_rl = ecounts_rl * weights

		#remove any bins with 0 sources
		has_sources = N_rl > 0.
		x_bins = S850_bin_centres[has_sources]
		eN_rl = eN_rl[has_sources]
		N_rl = N_rl[has_sources]

		if combined_plot:
			ax4.plot(x_bins, N_rl, color=grey, alpha=0.2)

		#offset at which the points will be plotted relative to the bin centre (to avoid overlapping error bars)
		x_bins = 10. ** (np.log10(x_bins) + 0.004 * ((-1.) ** ((j+1) % 2.)) * np.floor((j + 2.) / 2.))
		#x_bins += offset
		
		#plot the bin heights at the bin centres
		ax1[row_c,col_c].plot(x_bins, N_rl, marker=mkr_now, color=c_now, label=ID, ms=9., alpha=0.7, linestyle='none')
		#add errorbars
		eN_rl[eN_rl == N_rl] *= 0.999		#prevents errors when logging plot
		ax1[row_c,col_c].errorbar(x_bins, N_rl, fmt='none', yerr=eN_rl, ecolor=c_now, alpha=0.7)


		if plot_cumulative:
			#calculate the cumulative counts
			cumcounts_rl = np.nancumsum(counts_rl[::-1])[::-1]
			#calculate the Poissonian uncertainties
			ecumcounts_rl = np.sqrt(cumcounts_rl)

			#divide the counts (and uncertainties) by the area
			cumN_rl = cumcounts_rl / A_rl
			ecumN_rl = ecumcounts_rl / A_rl

			if combined_plot:
				ax5.plot(S850_bin_edges[:-1], cumN_rl, color=grey, alpha=0.2)

			#plot the bin heights at the left bin edges
			x_bins = 10. ** (np.log10(S850_bin_edges[:-1]) + 0.004 * ((-1.) ** ((j+1) % 2.)) * np.floor((j + 2.) / 2.))
			ax3[row_c,col_c].plot(x_bins, cumN_rl, marker=mkr_now, color=c_now, label=ID, ms=9., alpha=0.7, linestyle='none')
			#add errorbars
			ecumN_rl[ecumN_rl == cumN_rl] *= 0.999		#prevents errors when logging plot
			ax3[row_c,col_c].errorbar(x_bins, cumN_rl, fmt='none', yerr=ecumN_rl, ecolor=c_now, alpha=0.7)

	#calculate the total area surveyed for this RL galaxy
	A_zbin = A_sqdeg * len(data_rq_zbin) - A_overlap_zbin


	### differential counts ###
	#calculate the Poissonian uncertainties in the bins
	ecounts_zbin = np.sqrt(counts_zbin)

	#divide the counts (and uncertainties) by the area times the width of the bin
	weights = 1. / (A_zbin * dS)
	N_zbin = counts_zbin * weights
	eN_zbin = ecounts_zbin * weights
	#remove any bins with 0 sources
	has_sources = N_zbin > 0.
	x_bins = S850_bin_centres[has_sources]
	eN_zbin = eN_zbin[has_sources]
	N_zbin = N_zbin[has_sources]

	#plot the bin heights at the bin centres
	ax1[row_c,col_c].plot(x_bins, N_zbin, marker='o', color='k', label='All', ms=14., linestyle='none')
	#add errorbars
	eN_zbin[eN_zbin == N_zbin] *= 0.999		#prevents errors when logging plot
	ax1[row_c,col_c].errorbar(x_bins, N_zbin, fmt='none', yerr=eN_zbin, ecolor='k', elinewidth=2.4)

	#set the axes to log scale
	ax1[row_c,col_c].set_xscale('log')
	ax1[row_c,col_c].set_yscale('log')

	#add text to the top right corner displaying the redshift bin
	ax1[row_c,col_c].text(0.95, 0.95, r'$%.1f \leq z < %.1f$'%(z-dz/2.,z+dz/2.), transform=ax1[row_c,col_c].transAxes, ha='right', va='top')

	#set the minor tick locations on the x-axis
	ax1[row_c,col_c].set_xticks(xtick_min_locs, labels=xtick_min_labels, minor=True)

	#set the axes limits
	ax1[row_c,col_c].set_xlim(1.5, 20.)
	ax1[row_c,col_c].set_ylim(0.1, 1000.)
	#force matplotlib to label with the actual numbers
	#ax1[row_c,col_c].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	#ax1[row_c,col_c].get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax1[row_c,col_c].get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
	ax1[row_c,col_c].get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:g}'))

	labels_ord.insert(0, 'All')
	#add a legend in the bottom left corner, removing duplicate labels
	handles, labels = ax1[row_c,col_c].get_legend_handles_labels()
	labels_ord = [s for s in labels_ord if s in labels]
	by_label = dict(zip(labels, handles))
	ax1[row_c,col_c].legend([by_label[i] for i in labels_ord], [i for i in labels_ord], loc=3)


	if plot_cumulative:
		#calculate the cumulative counts in each bin
		cumcounts_zbin = np.nancumsum(counts_zbin[::-1])[::-1]
		#calculate the Poissonian uncertainties
		ecumcounts_zbin = np.sqrt(cumcounts_zbin)

		#divide the counts (and uncertainties) by the area
		cumN_zbin = cumcounts_zbin / A_zbin
		ecumN_zbin = ecumcounts_zbin / A_zbin

		#plot the bin heights at the left bin edges
		x_bins = S850_bin_edges[:-1]
		ax3[row_c,col_c].plot(x_bins, cumN_zbin, marker='o', color='k', label='All', ms=14., linestyle='none')
		#add errorbars
		ecumN_zbin[ecumN_zbin == cumN_zbin] *= 0.999		#prevents errors when logging plot
		ax3[row_c,col_c].errorbar(x_bins, cumN_zbin, fmt='none', yerr=ecumN_zbin, ecolor='k', elinewidth=2.4)

		#set the axes to log scale
		ax3[row_c,col_c].set_xscale('log')
		ax3[row_c,col_c].set_yscale('log')

		#add text to the top right corner displaying the redshift bin
		ax3[row_c,col_c].text(0.95, 0.95, r'$%.1f \leq z < %.1f$'%(z-dz/2.,z+dz/2.), transform=ax3[row_c,col_c].transAxes, ha='right', va='top')

		#set the minor tick locations on the x-axis
		ax3[row_c,col_c].set_xticks(xtick_min_locs, labels=xtick_min_labels, minor=True)

		#set the axes limits
		ax3[row_c,col_c].set_xlim(0.8, 20.)
		ax3[row_c,col_c].set_ylim(0.1, 4000.)
		#force matplotlib to label with the actual numbers
		#ax3[row_c,col_c].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
		#ax3[row_c,col_c].get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
		ax3[row_c,col_c].get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
		ax3[row_c,col_c].get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:g}'))

		#add a legend in the bottom left corner, removing duplicate labels
		handles, labels = ax3[row_c,col_c].get_legend_handles_labels()
		labels_ord = [s for s in labels_ord if s in labels]
		by_label = dict(zip(labels, handles))
		ax3[row_c,col_c].legend([by_label[i] for i in labels_ord], [i for i in labels_ord], loc=3)

	#perform whichever steps from above are relevant for the positions plot if created
	if plot_positions:
		#add text to the top right corner displaying the redshift bin
		ax2[row_c,col_c].text(0.95, 0.95, r'$%.1f \leq z < %.1f$'%(z-dz/2.,z+dz/2.), transform=ax2[row_c,col_c].transAxes, ha='right', va='top')
		#set the axes limits
		ax2[row_c,col_c].set_xlim(150.9, 149.3)
		ax2[row_c,col_c].set_ylim(1.5, 3.)
		#add a legend in the bottom left corner, removing duplicate labels
		handles, labels = ax2[row_c,col_c].get_legend_handles_labels()
		by_label = dict(zip(labels, handles))
		ax2[row_c,col_c].legend(by_label.values(), by_label.keys(), loc=3)

if combined_plot:
	### differential number counts
	#calculate the Poissonian uncertainties in the bins
	ecounts_ALL = np.sqrt(counts_ALL)

	#calculate the total area surveyed for all targets
	A_ALL = A_sqdeg * len(data_rq_sub) - A_overlap_ALL

	#divide the counts (and uncertainties) by the area times the width of the bin
	weights = 1. / (A_ALL * dS)
	N_ALL = counts_ALL * weights
	eN_ALL = ecounts_ALL * weights
	#remove any bins with 0 sources
	has_sources = N_ALL > 0.
	x_bins = S850_bin_centres[has_sources]
	eN_ALL = eN_ALL[has_sources]
	N_ALL = N_ALL[has_sources]

	#plot the bin heights at the bin centres
	ax4.plot(x_bins, N_ALL, marker='o', color='k', label=label_combined, ms=11., linestyle='none')
	#add errorbars
	eN_ALL[eN_ALL == N_ALL] *= 0.999		#prevents errors when logging plot
	ax4.errorbar(x_bins, N_ALL, fmt='none', yerr=eN_ALL, ecolor='k', elinewidth=2.4)

	#set the axes to log scale
	ax4.set_xscale('log')
	ax4.set_yscale('log')

	#set the minor tick locations on the x-axis
	ax4.set_xticks(xtick_min_locs, labels=xtick_min_labels, minor=True)

	#set the axes limits
	ax4.set_xlim(1.5, 20.)
	ax4.set_ylim(0.1, 1000.)
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
		#calculate the cumulative counts in each bin
		cumcounts_ALL = np.nancumsum(counts_ALL[::-1])[::-1]
		#calculate the Poissonian uncertainties
		ecumcounts_ALL = np.sqrt(cumcounts_ALL)

		#divide the counts (and uncertainties) by the area
		cumN_ALL = cumcounts_ALL / A_ALL
		ecumN_ALL = ecumcounts_ALL / A_ALL

		#remove any bins with 0 sources
		has_sources = cumN_ALL > 0.
		x_bins = S850_bin_edges[:-1][has_sources]
		ecumN_ALL = ecumN_ALL[has_sources]
		cumN_ALL = cumN_ALL[has_sources]

		#plot the bin heights at the bin centres
		ax5.plot(x_bins, cumN_ALL, marker='o', color='k', label=label_combined, ms=11., linestyle='none')
		#add errorbars
		eN_ALL[eN_ALL == N_ALL] *= 0.999		#prevents errors when logging plot
		ax5.errorbar(x_bins, cumN_ALL, fmt='none', yerr=ecumN_ALL, ecolor='k', elinewidth=2.4)

		#set the axes to log scale
		ax5.set_xscale('log')
		ax5.set_yscale('log')

		#set the minor tick locations on the x-axis
		ax5.set_xticks(xtick_min_locs, labels=xtick_min_labels, minor=True)

		#set the axes limits
		ax5.set_xlim(0.8, 20.)
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

########################
#### SAVING FIGURES ####
########################

#suffix to use for figure based on the operations performed
suffix = ''
if not independent_rq:
	suffix += '_no_overlap'
if comp_correct:
	suffix += '_comp_corr'

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

print(mf.colour_string('Done!', 'purple'))












