############################################################################################################
# A script for selecting a sample of galaxies from a catalogue with similar stellar masses and redshifts to
# the RAGERS radio-loud sample. NOTE: Before running must be in the 'photom' environment - type  
# 'conda activate photom' in the Terminal prior to running this script.
#
# v2: Rather than use estimated rest-frame 500 MHz radio luminosities to categorise galaxies as RL or RQ,
# counts all galaxies detected in VLA-COSMOS (3 GHz) as RL.
# v3: Uses RA and Dec cuts defined using the SCUBA-2 sensitivity map.
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
import my_functions as mf
from photutils.aperture import CircularAperture, SkyCircularAperture, ApertureStats, aperture_photometry

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
###############    START OF SCRIPT    #################
#######################################################

#relevant paths
PATH_RAGERS = sys.argv[1]
PATH_CATS = sys.argv[2]
PATH_DATA = sys.argv[3]
PATH_PLOTS = sys.argv[4]

#whether or not to include radio-loud RAGERS galaxies with limiting stellar masses
include_lims = False


#########################################
#### LOADING DATA AND CROSS-MATCHING ####
#########################################

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


'''
#path to the S2COSMOS catalogue (Simpson+19)
S2C_CAT = PATH_CATS + 'S2COSMOS_sourcecat850_Simpson18.fits'
data_s2c = Table.read(S2C_CAT, format='fits')
#get the minimum and maximum RA and Dec. (in deg) from the catalogue
RA_min, RA_max = min(data_s2c['RA_deg']), max(data_s2c['RA_deg'])
DEC_min, DEC_max = min(data_s2c['DEC_deg']), max(data_s2c['DEC_deg'])
#ultimately only want COSMOS2020 galaxies with >4' SCUBA-2 coverage in all directions; add appropriate padding to min and max values
pad = 5. 	#arcmin
pad /= 60.	#deg
RA_min += pad
RA_max -= pad
DEC_min += pad
DEC_max -= pad
'''

#path to the reference catalogue (COSMOS2020)
REF_CAT = PATH_CATS + 'COSMOS2020_CLASSIC_R1_v2.0.fits'
#load the catalogue
data_ref = Table.read(REF_CAT, format='fits')
#keep only relevant columns from COSMOS2020 catalogue
cols_keep = ['ALPHA_J2000', 'DELTA_J2000', 'ez_z_phot', 'ez_z160', 'ez_z840', 'ez_mass', 'ez_mass_p160', 'ez_mass_p840']
data_ref = data_ref[cols_keep]
#create mask to remove all sources with no redshift or stellar mass
mask = ~data_ref['ez_z_phot'].mask * ~data_ref['ez_mass'].mask
data_ref = data_ref[mask]
#create a mask to select COSMOS2020 sources within the RA and Dec boundaries
#coord_mask = (data_ref['ALPHA_J2000'] > RA_min) * (data_ref['ALPHA_J2000'] < RA_max) * (data_ref['DELTA_J2000'] > DEC_min) * (data_ref['DELTA_J2000'] < DEC_max)
#data_ref = data_ref[coord_mask]

#path to the catalogue containing data for VLA sources with COSMOS2015 counterparts
VLA_CAT = PATH_CATS + 'VLA_3GHz_counterpart_array_20170210_paper_smolcic_et_al.fits'
data_vla = Table.read(VLA_CAT, format='fits')
#remove multi-component radio sources and sources with high probability of being falsely matched
vla_keep = (data_vla['MULTI'] == 0) * (data_vla['P_FALSE'] < 0.2)
data_vla = data_vla[vla_keep]
#only keep relevant columns
#cols_keep = ['ID_VLA', 'RA_VLA_J2000', 'DEC_VLA_J2000', 'ID_CPT', 'RA_CPT_J2000', 'DEC_CPT_J2000', 'Z_BEST', 'FLUX_INT_3GHz', 'Lradio_10cm', 'Lradio_21cm']
#data_vla = data_vla[cols_keep]
#cross-match with the COSMOS2020 catalogue
data_ref = mf.cross_match_catalogues(data_ref, data_vla, 'ALPHA_J2000', 'DELTA_J2000', 'RA_CPT_J2000', 'DEC_CPT_J2000', tol=1., join='all1')
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
	MATCHED_CAT = PATH_CATS + 'RAGERS_COSMOS2020_matches_Mstar_z_incl_RL_limits.fits'
else:
	MATCHED_CAT = PATH_CATS + 'RAGERS_COSMOS2020_matches_Mstar_z.fits'
#create lists which will later be converted to columns in the table for the RAGERS ID
ragers_id_col = []
ragers_z_col = []
ragers_mstar_col = []
#create another list to which the COSMOS data will be appended for each matched galaxy
COSMOS_data_matched = []
#create one more list to which the number of matched sources will be appended for each RAGERS galaxy
N_matches_all = []


#define z and Mstar intervals within which sources will be considered 'matched' to a RAGERS galaxy
dz_sel = 0.1
dm_sel = 0.05


#################################
#### SCUBA-2 SENSITIVITY MAP ####
#################################

#set a sensitivity limit (mJy/beam)
sensitivity_limit = 1.3
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


######################
#### FIGURE SETUP ####
######################

#create the figure (stellar mass vs redshift)
f1, ax1 = plt.subplots(1, 1)
#label the axes
ax1.set_xlabel(r'$z$')
ax1.set_ylabel(r'log$_{10}$($M_{\star}$/M$_{\odot}$)')

#create the figure (number of matches vs redshift and vs stellar mass)
f2 = plt.figure(figsize=(2*x_size, y_size))
gs = f2.add_gridspec(ncols=2, nrows=1, width_ratios=[1,1])
#add axes for each panel
ax2a = f2.add_subplot(gs[0,0])
ax2b = f2.add_subplot(gs[0,1], sharey=ax2a)
#label the axes
ax2a.set_xlabel(r'$z$')
ax2a.set_ylabel(r'Number of matches')
ax2b.set_xlabel(r'log$_{10}$($M_{\star}$/M$_{\odot}$)')

#create the figure (RA and Dec. of the matched galaxies)
f3, ax3 = plt.subplots(1, 1, figsize=(1.6*x_size, y_size))
#label the axes
ax3.set_xlabel(r'RA (deg)')
ax3.set_ylabel(r'Dec. (deg)')

#colourmap to use for plots
cmap = mpl.cm.rainbow
#get N colours from the colourmap where N is the number of RAGERS sources
colours = [cmap(i/N_rl) for i in range(N_rl)]


###########################################
#### RADIO QUIET GALAXY IDENTIFICATION ####
###########################################

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
		apers = SkyCircularAperture(coords, 6.*u.arcmin)
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
		N_matches = len(masked_cat_RQ)
		N_matches_all.append(N_matches)
		print(N_matches, np.sum(N_matches_all))

		#extend the 'ragers_id_col', 'ragers_z_col' and 'ragers_mstar_col' lists by repeating the relevant value x N_matches
		ragers_id_col.extend([ID]*N_matches)
		ragers_z_col.extend([z]*N_matches)
		ragers_mstar_col.extend([m]*N_matches)
		#append the COSMOS2020 data for the matched galaxies to the relevant list
		COSMOS_data_matched.append(masked_cat_RQ)

		#write the RAGERS ID and number of companions to the file
		file.write(f'{ID}\t{N_matches}\n')

		#draw a box on Figure 1 representing this selection window
		rect = patches.Rectangle((z-dz_sel, m-dm_sel), 2.*dz_sel, 2.*dm_sel, edgecolor=green, facecolor='none', alpha=0.3, zorder=1)
		ax1.add_patch(rect)

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


#plot the values with their errorbars
ax1.plot(z_matched_RQ, Mstar_matched_RQ, marker='.', c=magenta, linestyle='None', alpha=0.3, ms=3., label='COSMOS2020 sample (RQ)')
ax1.plot(z_matched_RL, Mstar_matched_RL, marker='.', c=dark_blue, linestyle='None', alpha=0.3, ms=3., label='COSMOS2020 sample (RL)')
#ax1.errorbar(z_matched, Mstar_matched, xerr=(ezlo_matched,ezhi_matched), yerr=(eMstarlo_matched,eMstarhi_matched), ecolor=lilac, fmt='None', alpha=0.3)


#remove duplicates from legend
handles, labels = ax1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax1.legend(by_label.values(), by_label.keys(), loc=2)

#format the figure
f1.tight_layout()
#save the figure
if include_lims:
	figname = 'RQ_sample_selection_incl_RL_limits.png'
else:
	figname = 'RQ_sample_selection.png'
	#set the axis limits
	#xmin, xmax = ax1.set_xlim(0.5, 4.)
	ymin, ymax = ax1.set_ylim(11.04, 11.9)
f1.savefig(PATH_PLOTS+figname, bbox_inches='tight', dpi=300)


#stack the data for the matched sources into one table
COSMOS_data_matched = vstack(COSMOS_data_matched)
#add the columns for the RAGERS galaxy ID, z and Mstar
COSMOS_data_matched.add_columns([ragers_id_col, ragers_z_col, ragers_mstar_col], names=['RAGERS_ID', 'RAGERS_z', 'RAGERS_logMstar'], indexes=[0,0,0])
#write the catalogue to a file
COSMOS_data_matched.write(MATCHED_CAT, overwrite=True)



######################################
#### NUMBER OF MATCHES PER SOURCE ####
######################################

#plot the number of matches vs redshift in panel 1 of Figure 2
sc2a = ax2a.scatter(z_rl, N_matches_all, c=Mstar_rl, cmap=cmap)
#add a colourbar
cbar2a = f2.colorbar(sc2a, ax=ax2a)
cbar2a.ax.set_ylabel(r'log$_{10}$($M_{\star}$/M$_{\odot}$)')


#plot the number of matches vs redshift in panel 1 of Figure 2
sc2b = ax2b.scatter(Mstar_rl, N_matches_all, c=z_rl, cmap=cmap)
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
ax3.legend(loc='center right', bbox_to_anchor=(1.34, 0.5))

#format the figure
f3.tight_layout()
#save the figure
if include_lims:
	figname = 'RQ_positions_sky_incl_RL_limits.png'
else:
	figname = 'RQ_positions_sky.png'
f3.savefig(PATH_PLOTS+figname, bbox_inches='tight', dpi=300)
