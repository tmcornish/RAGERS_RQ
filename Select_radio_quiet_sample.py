############################################################################################################
# A script for selecting a sample of galaxies from a catalogue with similar stellar masses and redshifts to
# the RAGERS radio-loud sample.
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
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from photutils.aperture import CircularAperture, SkyCircularAperture, ApertureStats, aperture_photometry
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import general as gen
import plotstyle as ps
import astrometry as ast
import stats



##################
#### SETTINGS ####
##################

include_lims = False	#include radio-loud RAGERS galaxies with limiting stellar masses
plot_selection = True	#visualise the sample selection in z-M* space
plot_cosmos = True		#plot the whole COSMOS ditribution (requires plot_selection is also True)
sep_figure = True		#makes a separate figure for the COSMOS distribution (requires plot_selection and plot_cosmos are True)
plot_N_per_RL = True	#plot the number of RQ counterparts per RL source
plot_positions = True	#plot the positions of the RQ counterparts
settings = [include_lims, plot_selection, plot_cosmos, plot_N_per_RL, plot_positions]
#print the chosen settings to the Terminal
print(gen.colour_string('CHOSEN SETTINGS:', 'white'))
settings_print = [
	'Include stellar mass limits: ',
	'Plot sample selection: ',
	'Show full COSMOS catalogue on selection plot: ',
	'Plot number of RQ counterparts per RL galaxy: ',
	'Plot RQ counterpart positions: ',
]
for i in range(len(settings_print)):
	if settings[i]:
		settings_print[i] += 'y'
	else:
		settings_print[i] += 'n'

#define z and Mstar intervals within which sources will be considered 'matched' to a RAGERS galaxy
dz_sel = 0.1
dm_sel = 0.05
sel_str = u'Selection criteria:\n  dz = \u00B1'+f'{dz_sel}\n  '+u'dlog(M/Msun) = \u00B1'+f'{dm_sel}'
#search radius (arcmin) to be used later when looking for submm companions 
r_search = gen.r_search
search_str = f'SMG search radius: {r_search} arcmin'
settings_print.extend([sel_str, search_str])

print(gen.colour_string('\n'.join(settings_print), 'white'))

#formatting for graphs
plt.style.use(ps.styledict)

#######################################################
###############    START OF SCRIPT    #################
#######################################################

#relevant paths
PATH_RAGERS = gen.PATH_RAGERS
PATH_CATS = gen.PATH_CATS
PATH_DATA = gen.PATH_DATA
PATH_PLOTS = gen.PATH_PLOTS


######################
#### FIGURE SETUP ####
######################

if plot_selection:
	#create the figure (stellar mass vs redshift)
	f1, ax1 = plt.subplots(1, 1)
	#label the axes
	ax1.set_xlabel(r'$z$')
	ax1.set_ylabel(r'log$_{10}$($M_{\star}$/M$_{\odot}$)')

	#set the minimum and maximum redshifts and stellar masses for the plot
	if plot_cosmos:
		xmin1, xmax1 = ax1.set_xlim(0.88, 3.2)
		ymin1, ymax1 = ax1.set_ylim(11.04, 12.2)
	else:
		xmin1, xmax1 = ax1.set_xlim(0.88, 3.2)
		ymin1, ymax1 = ax1.set_ylim(10.95, 11.9)
	

if plot_N_per_RL:
	#create the figure (number of matches vs redshift and vs stellar mass)
	f2 = plt.figure(figsize=(2*ps.x_size, ps.y_size))
	gs = f2.add_gridspec(ncols=2, nrows=1, width_ratios=[1,1])
	#add axes for each panel
	ax2a = f2.add_subplot(gs[0,0])
	ax2b = f2.add_subplot(gs[0,1], sharey=ax2a)
	#label the axes
	ax2a.set_xlabel(r'$z$')
	ax2a.set_ylabel(r'Number of matches')
	ax2b.set_xlabel(r'log$_{10}$($M_{\star}$/M$_{\odot}$)')

if plot_positions:
	#create the figure (RA and Dec. of the matched galaxies)
	f3, ax3 = plt.subplots(1, 1, figsize=(1.6*ps.x_size, ps.y_size))
	#label the axes
	ax3.set_xlabel(r'RA (deg)')
	ax3.set_ylabel(r'Dec. (deg)')

#colourmap to use for plots
cmap = mpl.cm.rainbow


#########################################
#### LOADING DATA AND CROSS-MATCHING ####
#########################################

print(gen.colour_string('Loading RAGERS catalogue...', 'purple'))

#path to the RAGERS radio-loud catalogue
RL_CAT = PATH_CATS + 'RAGERS_radio_loud_sample.fits'
#load the radio-loud catalogue
data_rl = Table.read(RL_CAT, format='fits')
#the stellar masses are all strings because some are limits and thus contain '<'; need to convert
Mstar_rl_str = data_rl['log(Mstar/Mo)']
Mstar_lims = np.array([True if '<' in m else False for m in Mstar_rl_str])
if not include_lims:
	#remove the mass limit galaxies from the sample
	data_rl = data_rl[~Mstar_lims]
	Mstar_rl_str = data_rl['log(Mstar/Mo)']
#convert to floats
Mstar_rl = np.array(list(map(float,[m[1:] if '<' in m else m for m in Mstar_rl_str])))
#retrieve the IDs and redshifts
id_rl = data_rl['ID']
z_rl = data_rl['z']
#number of RAGERS sources
N_rl = len(data_rl)

#get N colours from the colourmap where N is the number of RAGERS sources
colours = [cmap(i/N_rl) for i in range(N_rl)]


print(gen.colour_string('Loading COSMOS2020 catalogue...', 'purple'))

#path to the reference catalogue (COSMOS2020)
REF_CAT = PATH_CATS + 'COSMOS2020_CLASSIC_R1_v2.0.fits'
#load the catalogue
data_ref = Table.read(REF_CAT, format='fits')
#keep only relevant columns from COSMOS2020 catalogue
cols_keep = ['ALPHA_J2000', 'DELTA_J2000', 'ez_z_phot', 'ez_z160', 'ez_z840', 'ez_mass', 'ez_mass_p160', 'ez_mass_p840']
data_ref = data_ref[cols_keep]
#create mask to remove all sources with no redshift or stellar mass
mask = ~data_ref['ez_z_phot'].mask * ~data_ref['ez_mass'].mask
#mask *= (data_ref['ez_z_phot'] < 7.) * (data_ref['ez_mass'] < 13.)
data_ref = data_ref[mask]
#create a mask to select COSMOS2020 sources within the RA and Dec boundaries
#coord_mask = (data_ref['ALPHA_J2000'] > RA_min) * (data_ref['ALPHA_J2000'] < RA_max) * (data_ref['DELTA_J2000'] > DEC_min) * (data_ref['DELTA_J2000'] < DEC_max)
#data_ref = data_ref[coord_mask]


if plot_selection and plot_cosmos:

	print(gen.colour_string('Plotting COSMOS2020 Mstar vs z...', 'purple'))

	#contour levels to mark intervals of sigma
	sig_levels = [1., 2., 3.]
	#clevels = np.array([np.diff(stats.percentiles_nsig(i))[0] for i in sig_levels])[::-1] / 100.
	clevels = np.arange(0.1, 1.1, 0.2)[::-1]

	#create an inset panel
	axins = inset_axes(ax1,
		width="100%",
		height="100%",
		bbox_to_anchor=(.65, .65, .3, .3),
		bbox_transform=ax1.transAxes)

	#label the axes
	axins.set_xlabel(r'$z$', fontsize=14., labelpad=-5.)
	axins.set_ylabel(r'log$_{10}$($M_{\star}$/M$_{\odot}$)', fontsize=14.)

	#create a 2D histogram of all COSMO2020 sources in M*-z space
	P, zbins, Mbins = np.histogram2d(data_ref['ez_z_phot'], data_ref['ez_mass'], 50)
	#get the bin centres in x and y
	zc = (zbins[1:] + zbins[:-1]) / 2.
	Mc = (Mbins[1:] + Mbins[:-1]) / 2.
	#get the extent for plotting the contours
	ext = [min(zc), max(zc), min(Mc), max(Mc)]

	#normalise the histogram so that it represents the 2D PDF
	P = gaussian_filter(P, (1., 1.))
	P /= P.sum()
	#create an array of steps at which the PDF will be integrated
	t = np.linspace(0, P.max(), 1000)
	#integrate the PDF
	integral = ((P >= t[:, None, None]) * P).sum(axis=(1,2))
	#interpolate t with respect to the integral
	f_int = interp1d(integral, t)
	#find t at the value of each contour
	t_contours = f_int(clevels)

	#plot the contours
	axins.contour(zc, Mc, P.T, levels=t_contours, colors=ps.grey, extent=None, linewidths=1.)
	#plot the RAGERS galaxies
	axins.plot(z_rl, Mstar_rl, marker='o', linestyle='none', c='k', ms=2.)

	axins.set_xlim(0.4, 4.)
	axins.set_ylim(7., 12.6)

	#make the fontsize of the tick labels smaller
	axins.tick_params(labelsize=14.)

	#plot the full distribution of M* vs z for the COSMOS catalogue
	#to_plot = (data_ref['ez_z_phot'] >= xmin1) *  (data_ref['ez_z_phot'] <= xmax1) * (data_ref['ez_mass'] >= ymin1) *  (data_ref['ez_mass'] <= ymax1)
	#ax1.plot(data_ref['ez_z_phot'][to_plot], data_ref['ez_mass'][to_plot], marker='.', c=ps.grey, linestyle='none', ms=2., label='Full COSMOS2020 catalogue', alpha=0.05)

	#if told to make a separate figure, also plot the RAGERS positions and COSMOS2020 distribution on that
	if sep_figure:
		f_sep, ax_sep = plt.subplots(1, 1)
		ax_sep.set_xlabel(r'$z$')
		ax_sep.set_ylabel(r'log$_{10}$($M_{\star}$/M$_{\odot}$)')

		ax_sep.contour(zc, Mc, P.T, levels=t_contours, colors=ps.grey, extent=None)

		#plot the RAGERS galaxies
		ax_sep.plot(z_rl, Mstar_rl, marker='o', linestyle='none', c='k', ms=5., label='RAGERS sample')

		#retrieve the handles and labels for the legend so that one can be added for the contours
		handles, labels = ax_sep.get_legend_handles_labels()
		by_label = dict(zip(labels, handles))
		by_label['COSMOS2020 catalogue'] = Line2D([0], [0], color=ps.grey)

		ax_sep.legend([by_label[l] for l in by_label], [l for l in by_label])

		ax_sep.set_xlim(0.4, 4.)
		ax_sep.set_ylim(7., 12.6)

		f_sep.tight_layout()
		f_sep.savefig(PATH_PLOTS + 'RAGERS_RL_with_COSMOS_Mstar_z.png', dpi=300)


print(gen.colour_string('Loading VLA-COSMOS catalogue...', 'purple'))

#path to the catalogue containing data for VLA sources with COSMOS2015 counterparts
VLA_CAT = PATH_CATS + 'VLA_3GHz_counterpart_array_20170210_paper_smolcic_et_al.fits'
data_vla = Table.read(VLA_CAT, format='fits')
#remove multi-component radio sources and sources with high probability of being falsely matched
vla_keep = (data_vla['MULTI'] == 0) * (data_vla['P_FALSE'] < 0.2)
data_vla = data_vla[vla_keep]
#only keep relevant columns
#cols_keep = ['ID_VLA', 'RA_VLA_J2000', 'DEC_VLA_J2000', 'ID_CPT', 'RA_CPT_J2000', 'DEC_CPT_J2000', 'Z_BEST', 'FLUX_INT_3GHz', 'Lradio_10cm', 'Lradio_21cm']
#data_vla = data_vla[cols_keep]

print(gen.colour_string('Cross-matching VLA-COSMOS with COSMOS2020...', 'purple'))
#cross-match with the COSMOS2020 catalogue
data_ref = ast.cross_match_catalogues(data_ref, data_vla, 'ALPHA_J2000', 'DELTA_J2000', 'RA_CPT_J2000', 'DEC_CPT_J2000', tol=1., join='all1')
#deal with a bug in which the 'true' and 'false' strings in the various VLA flag columns have added whitespace
for col in ['Xray_AGN', 'MIR_AGN', 'SED_AGN', 'Quiescent_MLAGN', 'SFG', 'Clean_SFG', 'HLAGN', 'MLAGN', 'Radio_excess']:
	for s in np.unique(data_ref[col]):
		data_ref[col][data_ref[col] == s] = s.strip()
data_ref.write(PATH_CATS+'COSMOS2020_zphot_Mstar_VLA_data.fits', format='fits', overwrite=True)

#retrieve relevant parameters and uncertainties
z_ref = data_ref['ez_z_phot']
Mstar_ref = data_ref['ez_mass']


#name to be given to catalogue of radio-quiet sources matched in M* and z to the radio-loud sample
if include_lims:
	MATCHED_CAT_RQ = PATH_CATS + 'RAGERS_COSMOS2020_matches_Mstar_z_rq_incl_RAGERS_lims.fits'
	MATCHED_CAT_RL = PATH_CATS + 'RAGERS_COSMOS2020_matches_Mstar_z_rl_incl_RAGERS_lims.fits'
else:
	MATCHED_CAT_RQ = PATH_CATS + 'RAGERS_COSMOS2020_matches_Mstar_z_rq.fits'
	MATCHED_CAT_RL = PATH_CATS + 'RAGERS_COSMOS2020_matches_Mstar_z_rl.fits'
#create lists which will later be converted to columns in the table for the RAGERS ID
ragers_id_col_rq = []
ragers_z_col_rq = []
ragers_mstar_col_rq = []
ragers_id_col_rl = []
ragers_z_col_rl = []
ragers_mstar_col_rl = []
#create lists to which the COSMOS data will be appended for each matched RQ or RL galaxy
COSMOS_data_matched_rq = []
COSMOS_data_matched_rl = []
#create one more list to which the number of matched sources will be appended for each RAGERS galaxy
N_matches_all_rq = []
N_matches_all_rl = []


#################################
#### SCUBA-2 SENSITIVITY MAP ####
#################################

print(gen.colour_string('Creating mask using SCUBA-2 sensitivity mask...', 'purple'))

#set a sensitivity limit (mJy/beam)
sensitivity_limit = gen.sens_lim
#load the SCUBA-2 sensitivity map
SMAP_file = PATH_DATA + 'S2COSMOS_20180927_850_err_mf_crop.fits'
smap = fits.open(SMAP_file)
#retrieve the data and header (NOTE: the main HDU has 3 axes but the third has one value labelled as 'wavelength' - not actually useful)
smap_data = smap[0].data
smap_hdr = smap[0].header
#create a wcs object for just the first 2 axes
w = wcs.WCS(smap_hdr, naxis=2)

#number of pixels in each dimension
NAXIS1 = smap_hdr['NAXIS1']
NAXIS2 = smap_hdr['NAXIS2']
#get the coordinates at the lower left and upper right corners of the S2COSMOS field
RA_max, DEC_min = w.wcs_pix2world(0.5, 0.5, 1)
RA_min, DEC_max = w.wcs_pix2world(NAXIS1+0.5, NAXIS2+0.5, 1)
#define the `extent' within which the data from the sensitivity map can be plotted with imshow
extent = (RA_max, RA_min, DEC_min, DEC_max)

#create a copy of the smap_data where everything below the sensitivity limit = 1 and everything above = 0
smap_sel = np.zeros(smap_data[0].shape)
smap_sel[smap_data[0] <= sensitivity_limit] = 1.



###########################################
#### RADIO QUIET GALAXY IDENTIFICATION ####
###########################################

print(gen.colour_string('Identifying RQ counterparts...', 'purple'))

#count all galaxies detected in Smolcic as 'radio loud'
#RL_cut = ~data_ref['RA_VLA_J2000'].mask

#data_ref = Table.read(PATH_CATS+'COSMOS2020_zphot_Mstar_VLA_data.fits', format='fits')
#assume the HLAGN and MLAGN flags as flags for RL galaxies
RL_cut = (data_ref['HLAGN'] == 'true') | (data_ref['MLAGN'] == 'true')


#########################
#### SAMPLE MATCHING ####
#########################

#filename for a file containing the number of COSMOS2020 counterparts for each RAGERS galaxy
N_matched_file = PATH_CATS + 'N_RQ_per_RL.txt'
#open the file and write to it
with open(N_matched_file, 'w') as file:
	#write some information to the file header
	file.write(f'#Galaxies selected within redshift and stellar mass intervals of dz = {dz_sel} and dM = {dm_sel} of each RAGERS RL galaxy.\n#\n')
	#write the column names
	file.write(f'#RAGERS_ID\tN_RQ\n')

	######################

	if plot_selection:
		#begin by plotting the RAGERS RL sample
		ax1.plot(z_rl, Mstar_rl, linestyle='none', marker='o', c='k', ms=5., label='RAGERS sample', zorder=10.)
		if include_lims:
			ax1.quiver(z_rl[Mstar_lims], Mstar_rl[Mstar_lims], 0., -1., color='k', scale=scale, scale_units='inches', width=aw, headwidth=hw, headlength=hl, headaxislength=hal, zorder=9.)

	'''
	#create a mask that's initially all False (one for RQ and one for RL galaxies)
	mask_matched_all_RQ = np.full(len(data_ref), False)
	mask_matched_all_RL = np.full(len(data_ref), False)
	'''
	masked_cat_all_RQ = []
	masked_cat_all_RL = []

	print('Number of counterparts:')
	print(f'\t{gen.colour_string("RQ", "purple")}\t{gen.colour_string("RL", "cyan")}')

	#now cycle through each RAGERS galaxy and identify galaxies with similar redshifts
	for j in range(N_rl):
		ID,z,m,c = id_rl[j], z_rl[j], Mstar_rl[j], colours[j]
		dz = z_ref - z
		dm = Mstar_ref - m
		zmask = (np.abs(dz) <= dz_sel)
		Mstar_mask = (np.abs(dm) <= dm_sel)
		selection_mask = zmask & Mstar_mask
		masked_cat = data_ref[selection_mask]
		RL_cut_masked = RL_cut[selection_mask]

		#get the RAs and Decs of the selected sources
		RAs = masked_cat['ALPHA_J2000']
		DECs = masked_cat['DELTA_J2000']
		#create SkyCoord objects from these coordinates
		coords = SkyCoord(RAs, DECs, unit='deg')
		#place apertures of radius 4' around each source and take the median sensitivity
		apers = SkyCircularAperture(coords, r_search*u.arcmin)
		aperstats = ApertureStats(smap_data[0], apers, wcs=w)
		rms_medians = aperstats.median
		rms_means = aperstats.mean
		rms_stds = aperstats.std
		rms_mads = aperstats.mad_std
		#rms_maxes = aperstats.max
		rms_maxes = rms_means + rms_stds
		#keep only the sources for which the median sensitivity in its aperture is less than the limit
		#sens_mask = rms_medians <= sensitivity_limit
		#sens_mask = (rms_means + rms_stds) <= sensitivity_limit
		sens_mask = rms_maxes <= sensitivity_limit
		#add this condition to the mask
		masked_cat = masked_cat[sens_mask]
		RL_cut_masked = RL_cut_masked[sens_mask]

		'''
		#create another mask for removing sources closer to another RL galaxy in M-z space
		closer_to_other_RL = np.full(len(masked_cat), False)
		#calculate the 'distance' in M-z space from each source to the current RL galaxy
		mz_dist = np.sqrt((masked_cat['ez_z_phot'] - z) ** 2. + (masked_cat['ez_mass'] - m) ** 2.)
		#cycle through the other RL sources
		for m_other, z_other in zip(np.delete(Mstar_rl,j), np.delete(z_rl,j)):
			#calculate the distance in M-z space
			dz_other = masked_cat['ez_z_phot'] - z_other
			dm_other = masked_cat['ez_mass'] - m_other
			mz_dist_other = np.sqrt((dz_other ** 2.) + (dm_other ** 2.))
			#mask any sources for which the distance is less than the value for the target RL galaxy;
			#also apply the condition that it has to be in the selection box for the other galaxy
			closer_mask = (mz_dist_other < mz_dist) & (np.abs(dz_other) <= dz_sel) & (np.abs(dm_other <= dm_sel))
			#closer_to_other_RL[closer_mask] = True
			closer_to_other_RL |= closer_mask
		#mask the catalogue and the RL mask
		masked_cat = masked_cat[~closer_to_other_RL]
		RL_cut_masked = RL_cut_masked[~closer_to_other_RL]
		'''

		#mask the catalogue to select RL or RQ sources
		masked_cat_RQ = masked_cat[~RL_cut_masked]
		masked_cat_RL = masked_cat[RL_cut_masked]
		#append these masked catalogues to the relevant lists
		masked_cat_all_RQ.append(masked_cat_RQ)
		masked_cat_all_RL.append(masked_cat_RL)

		#count the number of matched sources for this RAGERS galaxy
		N_matches_rq = len(masked_cat_RQ)
		N_matches_all_rq.append(N_matches_rq)
		N_matches_rl = len(masked_cat_RL)
		N_matches_all_rl.append(N_matches_rl)
		print(f'{ID}\t'+gen.colour_string(f'{N_matches_rq}', 'purple')+'\t'+gen.colour_string(f'{N_matches_rl}', 'cyan'))

		#extend the 'ragers_id_col', 'ragers_z_col' and 'ragers_mstar_col' lists by repeating the relevant value x N_matches
		ragers_id_col_rq.extend([ID]*N_matches_rq)
		ragers_z_col_rq.extend([z]*N_matches_rq)
		ragers_mstar_col_rq.extend([m]*N_matches_rq)
		ragers_id_col_rl.extend([ID]*N_matches_rl)
		ragers_z_col_rl.extend([z]*N_matches_rl)
		ragers_mstar_col_rl.extend([m]*N_matches_rl)
		#append the COSMOS2020 data for the matched galaxies to the relevant list
		COSMOS_data_matched_rq.append(masked_cat_RQ)
		COSMOS_data_matched_rl.append(masked_cat_RL)

		#write the RAGERS ID and number of companions to the file
		file.write(f'{ID}\t{N_matches_rq}\n')

		if plot_selection:
			#draw a box on Figure 1 representing this selection window
			rect = patches.Rectangle((z-dz_sel, m-dm_sel), 2.*dz_sel, 2.*dm_sel, edgecolor=ps.green, facecolor='none', alpha=0.3, zorder=1)
			ax1.add_patch(rect)

		if plot_positions:
			#plot the RAs and Decs of the matched sources on Figure 3
			ax3.plot(masked_cat_RQ['ALPHA_J2000'], masked_cat_RQ['DELTA_J2000'], color=c, label=ID, linestyle='None', marker='o', alpha=0.5)

#mask the catalogue with the RQ and RL masks
masked_cat_all_RQ = vstack(masked_cat_all_RQ)
masked_cat_all_RL = vstack(masked_cat_all_RL)



#retrieve relevant parameters and uncertainties
z_matched_RQ = masked_cat_all_RQ['ez_z_phot']
Mstar_matched_RQ = masked_cat_all_RQ['ez_mass']
z_matched_RL = masked_cat_all_RL['ez_z_phot']
Mstar_matched_RL = masked_cat_all_RL['ez_mass']

if plot_selection:
	print(gen.colour_string('Plotting selection...', 'purple'))

	#plot the values with their errorbars
	ax1.plot(z_matched_RQ, Mstar_matched_RQ, marker='.', c=ps.magenta, linestyle='None', alpha=0.3, ms=3., label='COSMOS2020 sample (RQ)')
	ax1.plot(z_matched_RL, Mstar_matched_RL, marker='.', c=ps.dark_blue, linestyle='None', alpha=0.3, ms=3., label='COSMOS2020 sample (RL)')
	#ax1.errorbar(z_matched, Mstar_matched, xerr=(ezlo_matched,ezhi_matched), yerr=(eMstarlo_matched,eMstarhi_matched), ecolor=lilac, fmt='None', alpha=0.3)


	#remove duplicates from legend
	handles, labels = ax1.get_legend_handles_labels()
	by_label = dict(zip(labels, handles))
	ax1.legend(by_label.values(), by_label.keys(), loc=2)

	#format the figure
	f1.tight_layout()
	#save the figure
	figname = 'RQ_sample_selection.png'
	if include_lims:
		figname = figname[:-4] + '_incl_lims.png'
	if plot_cosmos:
		figname = figname[:-4] + '_with_cosmos.png'

	#set the axis limits
	#xmin, xmax = ax1.set_xlim(0.88, 3.2)
	#ymin, ymax = ax1.set_ylim(11.04, 11.9)
	#ymin, ymax = ax1.set_ylim(10., 11.9)
	f1.savefig(PATH_PLOTS+figname, bbox_inches='tight', dpi=300)



#stack the data for the matched sources into one table
COSMOS_data_matched_rq = vstack(COSMOS_data_matched_rq)
#add the columns for the RAGERS galaxy ID, z and Mstar
COSMOS_data_matched_rq.add_columns([ragers_id_col_rq, ragers_z_col_rq, ragers_mstar_col_rq], names=['RAGERS_ID', 'RAGERS_z', 'RAGERS_logMstar'], indexes=[0,0,0])
#write the catalogue to a file
COSMOS_data_matched_rq.write(MATCHED_CAT_RQ, overwrite=True)

#stack the data for the matched sources into one table
COSMOS_data_matched_rl = vstack(COSMOS_data_matched_rl)
#add the columns for the RAGERS galaxy ID, z and Mstar
COSMOS_data_matched_rl.add_columns([ragers_id_col_rl, ragers_z_col_rl, ragers_mstar_col_rl], names=['RAGERS_ID', 'RAGERS_z', 'RAGERS_logMstar'], indexes=[0,0,0])
#write the catalogue to a file
COSMOS_data_matched_rl.write(MATCHED_CAT_RL, overwrite=True)


######################################
#### NUMBER OF MATCHES PER SOURCE ####
######################################

if plot_N_per_RL:
	print(gen.colour_string('Plotting number of RQ counterparts...', 'purple'))

	#plot the number of matches vs redshift in panel 1 of Figure 2
	sc2a = ax2a.scatter(z_rl, N_matches_all_rq, c=Mstar_rl, cmap=cmap)
	#add a colourbar
	cbar2a = f2.colorbar(sc2a, ax=ax2a)
	cbar2a.ax.set_ylabel(r'log$_{10}$($M_{\star}$/M$_{\odot}$)')


	#plot the number of matches vs redshift in panel 1 of Figure 2
	sc2b = ax2b.scatter(Mstar_rl, N_matches_all_rq, c=z_rl, cmap=cmap)
	#add a colourbar
	cbar2b = f2.colorbar(sc2b, ax=ax2b)
	cbar2b.ax.set_ylabel(r'$z$')

	#format the figure
	f2.tight_layout()
	#save the figure 
	if include_lims:
		figname = 'N_RQ_vs_Mstar_and_z_incl_RL_limits.png'
	else:
		figname = 'N_RQ_vs_Mstar_and_z.png'
	f2.savefig(PATH_PLOTS+figname, bbox_inches='tight', dpi=300)



#######################################
#### RA AND DEC OF MATCHED SOURCES ####
#######################################

if plot_positions:
	print(gen.colour_string('Plotting RQ positions...', 'purple'))

	#invert x-axis
	ax3.invert_xaxis()
	#get the axis limits
	xmin, xmax = ax3.get_xlim()
	ymin, ymax = ax3.get_ylim()

	#add selection area to the background of the plot
	ax3.imshow(smap_sel, extent=extent, origin='lower', cmap='Greys', alpha=0.1, zorder=0)
	#reset the axis limits
	ax3.set_xlim(xmin, xmax)
	ax3.set_ylim(ymin, ymax)

	#add legend to right of plot
	ax3.legend(loc='center right', title='RQ analogues of...', title_fontsize='x-small', bbox_to_anchor=(1.34, 0.5))

	#format the figure
	f3.tight_layout()
	#save the figure
	if include_lims:
		figname = 'RQ_positions_sky_incl_RL_limits.png'
	else:
		figname = 'RQ_positions_sky.png'
	f3.savefig(PATH_PLOTS+figname, bbox_inches='tight', dpi=300)




print(gen.colour_string('Done!', 'purple'))



