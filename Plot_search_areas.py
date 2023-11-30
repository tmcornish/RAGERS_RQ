############################################################################################################
# A script for plotting the search areas used for a given RQ sample.
############################################################################################################

#import modules/packages
import os, sys
from astropy.table import Table, vstack
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import general as gen
import plotstyle as ps
import astrometry as ast
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


##################
#### SETTINGS ####
##################

#formatting of plots
plt.style.use(ps.styledict)

#toggle `switches' for additional functionality
plot_per_r = True			#makes a plot for each search radius used
save_samp = True			#saves the details of the RQ galaxies in this sample to a file
settings = [plot_per_r, save_samp]

#print the chosen settings to the Terminal
print(gen.colour_string('CHOSEN SETTINGS:', 'white'))
settings_print = [
	'Make a plot for each search radius used: ',
	'Save the details of the RQ sample to a FITS file: '
	]
for i in range(len(settings_print)):
	if settings[i]:
		settings_print[i] += 'y'
	else:
		settings_print[i] += 'n'

#markers to use for the plot; retrieve all possible markers, excluding the first two and final 4
markers = [m for m in mpl.lines.Line2D.markers][2:-4]

#remove a chunk of the rainbow colormap
interval = np.hstack([np.arange(0, 0.501, 0.001), np.arange(0.7, 1.001, 0.001)])
colors = plt.cm.rainbow(interval)
cmap = LinearSegmentedColormap.from_list('name', colors)

#number of RQ galaxies to use per RL galaxy
n_rq = 4

print(gen.colour_string('\n'.join(settings_print), 'white'))


#######################################################
###############    START OF SCRIPT    #################
#######################################################

#relevant paths
PATH_RAGERS = gen.PATH_RAGERS
PATH_CATS = gen.PATH_CATS
PATH_DATA = gen.PATH_DATA
PATH_PLOTS = gen.PATH_PLOTS


#set the radii (in arcmin) for which plots will be made
if plot_per_r:
	r_to_plot = gen.r_search_all
else:
	r_to_plot = [gen.r_search]


#############################################
#### SURVEY AREA BELOW SENSITIVITY LIMIT ####
#############################################

#load the SCUBA-2 sensitivity map
SMAP_file = PATH_DATA + 'S2COSMOS_20180927_850_err_mf_crop.fits'
#calculate the area below the desired sensitivity limit and retrieve the coordinates of the map corners
A_survey, smap_sel, (RAmax_smap, RAmin_smap, DECmin_smap, DECmax_smap) = ast.area_within_sensitivity_limit(SMAP_file, gen.sens_lim, to_plot=True)



##########################
#### SAMPLE SELECTION ####
##########################

#catalogue containing data for (radio-quiet) galaxies from COSMOS2020 matched in M* and z with the radio-loud sample
RQ_CAT = PATH_CATS + f'RAGERS_COSMOS2020_matches_Mstar_z_{gen.gal_type}.fits'
data_rq = Table.read(RQ_CAT, format='fits')
#get the RAs and DECs
RA_rq = data_rq['ALPHA_J2000']
DEC_rq = data_rq['DELTA_J2000']
#convert these into SkyCoord objects
coords_rq = SkyCoord(RA_rq, DEC_rq, unit='deg')


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
	N_sel_now = min(len(idx_matched), gen.n_gal)
	#randomly select N_sel_now of these RQ galaxies
	#np.random.seed(0)
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

if save_samp:
	#save the subsample of RQ galaxies to a new file
	data_rq_sub.write(PATH_CATS + f'subsamp_RAGERS_COSMOS2020_matches_Mstar_z_{gen.gal_type}.fits', overwrite=True)

#create SkyCoord objects from the coordinates of all of these RQ galaxies
coords_rq_sub = SkyCoord(data_rq_sub['ALPHA_J2000'], data_rq_sub['DELTA_J2000'], unit='deg')


##########################
#### MAKING THE PLOTS ####
##########################

for r in r_to_plot:

	#convert search radius to degrees
	R_deg = r / 60.
	#area of aperture (sq. deg)
	A_sqdeg = np.pi * R_deg ** 2.

	#create the axes
	f, ax = plt.subplots(1, 1)
	#label the axes
	ax.set_xlabel('RA (deg.)')
	ax.set_ylabel('Dec. (deg.)')

	#show the region of S2COSMOS below the sensitivity limit
	ax.imshow(smap_sel, extent=(RAmax_smap, RAmin_smap, DECmin_smap, DECmax_smap), origin='lower', cmap='Greys', alpha=0.1, zorder=0)

	labels_ord = []

	#calculate the area covered by all apertures
	A_all = ast.apertures_area(coords_rq_sub, r=R_deg)
	#add text displaying the total fractional coverage
	y_header = 0.02 + (len(RAGERS_IDs)+1) * 0.04
	ax.text(0.97, y_header, 'Fraction covered:', color='k', transform=ax.transAxes, ha='right', va='bottom', fontsize=14.)
	A_frac = A_all / A_survey
	ax.text(0.97, 0.02, f'{A_frac:.4f}', color='k', transform=ax.transAxes, ha='right', va='bottom', fontsize=14.)

	#cycle through the RL galaxies
	for i in range(len(RAGERS_IDs)):
		#get the current ID
		ID = RAGERS_IDs[i]
		labels_ord.append(ID)

		#select the colour and marker to use for this dataset
		c_now = gen.scale_RGB_colour(cmap((i+1.)/len(RAGERS_IDs))[:-1], scale_l=0.8)
		mkr_now = markers[i]

		#select the RQ galaxies corresponding to this RL source
		mask_rq_rl = data_rq_sub['RAGERS_ID'] == ID
		data_rq_rl = data_rq_sub[mask_rq_rl]
		#get the SkyCoords for these objects
		coords_rq_rl = coords_rq_sub[mask_rq_rl]


		#cycle through the RQ galaxies matched to this RL galaxy
		for k in range(len(data_rq_rl)):
			#get the coordinates for the current RQ galaxy
			coord_central = coords_rq_rl[k:k+1]

			#retrieve the RA and Dec. of the galaxy
			RA_rq = coord_central[0].ra.value
			DEC_rq = coord_central[0].dec.value
			#plot this position on the current axes
			ax.plot(RA_rq, DEC_rq, linestyle='none', marker=mkr_now, c=c_now, label=ID, alpha=0.6)
			#add circles with radius 4' and 6' centred on the RQ galaxy
			d_circle = 2. * R_deg
			f_cosdec = np.cos(DEC_rq * np.pi / 180.)
			ellipse = mpl.patches.Ellipse((RA_rq, DEC_rq), width=d_circle/f_cosdec, height=d_circle, color=c_now, fill=False, alpha=0.6)
			ax.add_patch(ellipse)

		#calculate the area covered, accounting for overlap between apertures
		A_rl = ast.apertures_area(coords_rq_rl, r=R_deg)
		
		y_area = 0.02 + (len(RAGERS_IDs) - i) * 0.04
		A_frac = A_rl / A_survey
		ax.text(0.97, y_area, f'{A_frac:.4f}', color=c_now, transform=ax.transAxes, ha='right', va='bottom', fontsize=14.)


	#add legend to right of plot
	handles, labels = ax.get_legend_handles_labels()
	labels_ord = [s for s in labels_ord if s in labels]
	by_label = dict(zip(labels, handles))
	ax.legend([by_label[l] for l in labels_ord], [l for l in labels_ord], loc='center right', title='RQ analogues of...', title_fontsize='xx-small', fontsize=11., bbox_to_anchor=(1.34, 0.5))


	#set the axes limits
	ax.set_xlim(150.9, 149.15)
	ax.set_ylim(1.5, 3.)
	#minimise unnecesary whitespace
	f.tight_layout()
	figname = PATH_PLOTS + f'RQ_positions_with_search_areas{r:.1f}am_{gen.n_gal}{gen.gal_type}.png'
	f.savefig(figname, bbox_inches='tight', dpi=300)
