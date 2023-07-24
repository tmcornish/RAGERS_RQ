############################################################################################################
# A script for measuring the SMG density in different environments using the S2COSMOS SNR map,
# constructing histograms, and comparing those with blank-field results.
############################################################################################################

'''
Plan for the script:
	- create catalogue/Table of all SNR>2 detections in the S2COSMOS map (within region above sensitivity limit)
	- count sources within R of each RQ analogue
	- count sources in ~10000 randomly placed apertures across whole blank field
	- make normalised histograms of both
	- perform KS test (p-value > 0.05 means can't reject possibility that both samples drawn from same dist)
	- (optional) do same for RL analogues and plot on same figure
'''

import os
from astropy.io import fits
from astropy import wcs
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure, binary_erosion
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
import astrometry as ast
from astropy.coordinates import SkyCoord
from astropy import units as u
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.detection import find_peaks
from scipy.stats import ks_2samp
import general as gen
import plotstyle as ps
from matplotlib.colors import LinearSegmentedColormap


##################
#### SETTINGS ####
##################

#toggle `switches' for additional functionality
incl_rl = True			#make histograms for the environments of RL analogues as well
all_fig = False			#make a figure showing all histograms on one set of axes
settings = [
	incl_rl
]
#print the chosen settings to the Terminal
print(gen.colour_string('CHOSEN SETTINGS:', 'white'))
settings_print = [
	'Make histograms for the environments of RL analogues as well as RQ: ',
]
for i in range(len(settings_print)):
	if settings[i]:
		settings_print[i] += 'y'
	else:
		settings_print[i] += 'n'

SNR_thresh = 2.		#SNR threshold to use for peak finding
N_aper_bf = 10000	#number of apertures to use for measuring the blank field
settings_print.extend([
	f'SNR threshold: {SNR_thresh}',
	f'Number of blank field apertures: {N_aper_bf}'
])

print(gen.colour_string('\n'.join(settings_print), 'white'))


#number of radii used to make number counts
N_radii = len(gen.r_search_all)
#select a radius for which the results will be emphasised on the plot
r_emph = 6.

#formatting of plots
plt.style.use(ps.styledict)
#use colours from middle section of the GnBu colourmap
interval = np.arange(0.1, 1.001, 0.001)
colors = plt.cm.GnBu(interval)
cmap = LinearSegmentedColormap.from_list('name', colors)
#choose N colours from this map, where N is the number of radii used minus any that are to be emphasised
N_clr = len([r for r in gen.r_search_all if r != r_emph])
cycle_clr = [gen.scale_RGB_colour(cmap((j+1.)/N_radii)[:-1], scale_l=0.7) for j in range(N_clr)]

#######################################################
###############    START OF SCRIPT    #################
#######################################################

#############################
#### DETECTING SNR PEAKS ####
#############################

print(gen.colour_string(f'Detecting SNR peaks...', 'purple'))

#load the S2COSMOS SNR map
hdu = fits.open(gen.PATH_DATA + 'S2COSMOS_20180927_850_snr_mf_crop.fits')[0]
data = hdu.data[0]
hdr = hdu.header

#create a WCS object
w = wcs.WCS(hdr, naxis=2)
#convert nans to 0
data[np.isnan(data)] = 0.

#run the photutils peak finder
peaks = find_peaks(data, threshold=SNR_thresh, box_size=10)

#make a grid with the same dimensions as the SNR map
peak_grid = np.zeros(data.shape)
#set pixels to 1 at the location of a peak
for x,y in zip(peaks['x_peak'], peaks['y_peak']):
	peak_grid[x,y] = 1.


######################################
#### MASKING WITH SENSITIVITY MAP ####
######################################

#load the S2COSMOS sensitivity map
SMAP_file = gen.PATH_DATA + 'S2COSMOS_20180927_850_err_mf_crop.fits'
smap = fits.open(SMAP_file)
smap_data = smap[0].data
smap_hdr = smap[0].header
#create a mask to select all pixels below the sensitivity limit
smap_sel = np.zeros(smap_data[0].shape)
smap_sel[smap_data[0] <= gen.sens_lim] = 1.
smap_sel_bool = smap_sel.astype(bool)

#reset to 0 any pixels outside the desired sensitivity range
peak_grid[~smap_sel_bool] = 0.


################################
#### CATALOGUE OF SNR PEAKS ####
################################

#get the coordinates 
peaks_masked = np.argwhere(peak_grid == 1.)
#convert to WCS coordinates
coords = w.wcs_pix2world(peaks_masked[:,0], peaks_masked[:,1], 0)
#make SkyCoord objects from these coordinates
coords_submm = SkyCoord(*coords, unit='deg')

#create a Table with these coordinates
t = Table([*coords], names=['RA', 'DEC'])
#make a regions file
ast.table_to_DS9_regions(t, 'RA', 'DEC', output_name=gen.PATH_CATS+'S2COSMOS_SNR_peaks.reg', radius='10"')



##########################################
#### BLANK-FIELD APERTURE COORDINATES ####
##########################################

print(gen.colour_string(f'Defining {N_aper_bf} blank-field apertures...', 'purple'))

#file for containing the blank field aperture information
bf_apers_file = gen.PATH_SIMS + 'Blank_field_aper_coords.npz'
#see if the file exists
if os.path.exists(bf_apers_file):
	bf_apers_data = np.load(bf_apers_file)
	apers_bf_dict = dict(zip(bf_apers_data.files, [bf_apers_data[f] for f in bf_apers_data.files]))
	#get the RAs and DECs
	RA_bf = bf_apers_data['RA']
	DEC_bf = bf_apers_data['DEC']
	del bf_apers_data
else:
	#get the pixel scale (''/pix) of the SNR map
	pixscale = (wcs.utils.proj_plane_pixel_scales(w) * 3600.).mean()
	#calculate the radius in pixels of the largest aperture used for number counts
	r_pix_max = max(gen.r_search_all) * 60. / pixscale
	#area of apertures with this radius (in pix^2)
	A_pix = np.pi * r_pix_max ** 2.
	#set up arrays for containing the coordinates of pixels within the sensitivity limit
	#X_bf, Y_bf = [np.empty((0,), dtype=int) for _ in range(2)]
	#set up a list for the tables of the blank field aperture information
	t_apers_bf_all = []
	#number of apertures defined
	N_done = 0
	while N_done < N_aper_bf:
		#calculate number of apertures still required
		N_todo = N_aper_bf - N_done
		#randomly choose X and Y coordinates
		X = np.random.rand(N_todo) * smap_hdr['NAXIS1']
		Y = np.random.rand(N_todo) * smap_hdr['NAXIS2']
		#place maximum-sized apertures on the smap_sel array and sum the pixel values
		apers = CircularAperture(np.array([X,Y]).T, r_pix_max)
		t_apers = aperture_photometry(smap_sel, apers, method='center')
		#find all apertures for which the sum is equal to the area
		within_area = t_apers['aperture_sum'] > 0.99 * A_pix
		#append these sources to the list of Tables
		t_apers_bf_all.append(t_apers[within_area])
		N_done += within_area.sum()
		print(N_done)
	#join the aperture info from each iteration into one Table
	t_apers_bf_all = vstack(t_apers_bf_all)
	#convert the X and Y coordinates into RAs and DECs
	coords_bf = w.wcs_pix2world(t_apers_bf_all['xcenter'], t_apers_bf_all['ycenter'], 0)
	RA_bf, DEC_bf = coords_bf

	#save the aperture information to a file
	apers_bf_dict = {
		'X' : np.array(t_apers_bf_all['xcenter']),
		'Y' : np.array(t_apers_bf_all['ycenter']),
		'RA' : RA_bf,
		'DEC' : DEC_bf
	}
	#np.savez_compressed(bf_apers_file, **apers_bf_dict)

	del t_apers_bf_all, coords_bf

#make SkyCoord objects from the aperture coordinates
coords_bf = SkyCoord(RA_bf, DEC_bf, unit='deg')



######################
#### FIGURE SETUP ####
######################

#axes labels
xlabel = r'SMG density (deg$^{-2}$)'
ylabel = r'Normalised counts'

#set up a figure for the current radius
ncols = 2
nrows = int(np.ceil(len(gen.r_search_all)/2))
f, ax = plt.subplots(nrows, ncols, figsize=(ncols*ps.x_size, 0.7*nrows*ps.y_size))
figname = gen.PATH_PLOTS + f'Density_histograms.png'

#to label axes with common labels, create a big subplot, make it invisible, and label its axes
ax_big = f.add_subplot(111, frameon=False)
ax_big.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, which='both')
#label x and y axes
ax_big.set_xlabel(xlabel, labelpad=20.)
ax_big.set_ylabel(ylabel, labelpad=20.)

#global labels to use for the blank field, RQ and RL analogues
bf_label = 'Blank field (S2COSMOS)'
rq_label = 'RQ analogues'
rl_label = 'RL analogues'
labels_ord = [rq_label, rl_label, bf_label]

#set up a combined figure if told to make one
if all_fig:
	#set up the figure anf label the axes
	f_rq, ax_rq = plt.subplots(1,1)
	ax_rq.set_xlabel(xlabel)
	ax_rq.set_ylabel(ylabel)
	#filename for the figure
	figname_rq = gen.PATH_PLOTS + 'Density_histograms_all_rq.png'

	if incl_rl:
		#set up the figure anf label the axes
		f_rl, ax_rl = plt.subplots(1,1)
		ax_rl.set_xlabel(xlabel)
		ax_rl.set_ylabel(ylabel)
		#filename for the figure
		figname_rl = gen.PATH_PLOTS + 'Density_histograms_all_rl.png'

	#set up lists for the legend labels
	labels_ord_com = []



############################
#### DENSITY HISTOGRAMS ####
############################

print(gen.colour_string(f'Creating density histograms...', 'purple'))

#load the catalogues containing the RQ and RL counterpart info
t_rq = Table.read(gen.PATH_CATS + 'RAGERS_COSMOS2020_matches_Mstar_z_rq.fits')
t_rl = Table.read(gen.PATH_CATS + 'RAGERS_COSMOS2020_matches_Mstar_z_rl.fits')

#files containing the SMG density information for all RQ and RL counterparts
rq_file = gen.PATH_CATS + 'RQ_SMG_densities.npz'
rl_file = gen.PATH_CATS + 'RL_SMG_densities.npz'
dicts_rq_rl = []
for file,t in zip([rq_file, rl_file], [t_rq, t_rl]):
	if os.path.exists(file):
		D = np.load(file)
		dicts_rq_rl.append(dict(zip(D.files, [D[k] for k in D.files])))
	else:
		dicts_rq_rl.append({'RA' : np.array(t['ALPHA_J2000']), 'DEC' : np.array(t['DELTA_J2000'])})
rq_dict, rl_dict = dicts_rq_rl
#make SkyCoord objects from the coordinates
coords_rq = SkyCoord(rq_dict['RA'], rq_dict['DEC'], unit='deg')
coords_rl = SkyCoord(rl_dict['RA'], rl_dict['DEC'], unit='deg')


#set up lists for containing the p-values from KS tests
p_rq, p_rl = [], []

#cycle through the search radii
nclr = 0
for i in range(len(gen.r_search_all)):

	r = gen.r_search_all[i]
	print(gen.colour_string(f'R = {r:.1f} arcmin', 'cyan'))

	#choose axes in main figure on which to plot
	nrow = i // ncols
	ncol = i % ncols

	#select the colour and marker to use for this dataset in the combined plot
	if r == r_emph:
		c_now = 'k'
		alpha = 1.
		label = r'$\mathbf{R = %.1f^{\prime}}$'%r
	else:
		c_now = cycle_clr[nclr]
		alpha = 0.4
		label = r'$R = %.1f^{\prime}$'%r
		nclr += 1
	#labels_ord_com.append(label)

	#search area in deg^2
	A_sqdeg = np.pi * (r / 60.) ** 2.

	print(gen.colour_string(f'Creating blank field histograms...', 'white'))

	#find the density of SMGs in each of the blank field apertures
	if f'density_{r:.1f}am' not in apers_bf_dict:
		N_matched_bf = []
		for k in range(len(coords_bf)):
			seps = coords_bf[k].separation(coords_submm)
			N_matched_bf.append((seps <= r * u.arcmin).sum())
		N_matched_bf = np.asarray(N_matched_bf)
		density_bf = N_matched_bf / A_sqdeg
		#add these to the dictionary for blank-field apertures
		apers_bf_dict[f'N_{r:.1f}'] = N_matched_bf
		apers_bf_dict[f'density_{r:.1f}am'] = density_bf
	else:
		N_matched_bf = apers_bf_dict[f'N_{r:.1f}']
		density_bf = apers_bf_dict[f'density_{r:.1f}am']

	#plot the histogram of densities
	weights_bf = np.ones_like(density_bf)/float(len(density_bf))
	counts_bf, bins_bf, _ = ax[nrow,ncol].hist(density_bf, weights=weights_bf, color=ps.crimson, alpha=0.5, linestyle='--', label=bf_label, histtype='step')
	if all_fig and r == r_emph:
		ax_rq.hist(density_bf, weights=weights_bf, color=ps.crimson, alpha=0.5, linestyle='--', label=bf_label, histtype='step')
		if incl_rl:
			ax_rl.hist(density_bf, weights=weights_bf, color=ps.crimson, alpha=0.5, linestyle='--', label=bf_label, histtype='step')


	print(gen.colour_string(f'Creating RQ histograms...', 'white'))

	#find the density of SMGs around each RQ source
	if f'density_{r:.1f}am' not in rq_dict:
		N_matched_rq = []
		for k in range(len(coords_rq)):
			seps = coords_rq[k].separation(coords_submm)
			N_matched_rq.append((seps <= r * u.arcmin).sum())
		N_matched_rq = np.asarray(N_matched_rq)
		density_rq = N_matched_rq / A_sqdeg
		#add these to the dictionary for blank-field apertures
		rq_dict[f'N_{r:.1f}'] = N_matched_rq
		rq_dict[f'density_{r:.1f}am'] = density_rq
	else:
		N_matched_rq = rq_dict[f'N_{r:.1f}']
		density_rq = rq_dict[f'density_{r:.1f}am']

	#plot the histogram of densities
	weights_rq = np.ones_like(density_rq)/float(len(density_rq))
	counts_rq, bins_rq, _ = ax[nrow,ncol].hist(density_rq, weights=weights_rq, color='k', label=rq_label, histtype='step')
	if all_fig:
		ax_rq.hist(density_rq, weights=weights_rq, color=c_now, alpha=alpha, label=label, histtype='step')

	#add a label to show the current radius
	ax[nrow,ncol].text(0.05, 0.95, r'$R = %.0f^{\prime}$'%r, transform=ax[nrow,ncol].transAxes, ha='left', va='top')

	#perform a 2-sample KS test and retrieve the p-value
	ks_rq = ks_2samp(density_bf, density_rq)
	p_rq.append(ks_rq.pvalue)


	if incl_rl:
		print(gen.colour_string(f'Creating RL histograms...', 'white'))
		#find the density of SMGs around each RQ source
		if f'density_{r:.1f}am' not in rl_dict:
			N_matched_rl = []
			for k in range(len(coords_rl)):
				seps = coords_rl[k].separation(coords_submm)
				N_matched_rl.append((seps <= r * u.arcmin).sum())
			N_matched_rl = np.asarray(N_matched_rl)
			density_rl = N_matched_rl / A_sqdeg
			#add these to the dictionary for blank-field apertures
			rl_dict[f'N_{r:.1f}'] = N_matched_rl
			rl_dict[f'density_{r:.1f}am'] = density_rl
		else:
			N_matched_rl = rl_dict[f'N_{r:.1f}']
			density_rl = rl_dict[f'density_{r:.1f}am']

		#plot the histogram of densities
		weights_rl = np.ones_like(density_rl)/float(len(density_rl))
		counts_rl, bins_rl, _ = ax[nrow,ncol].hist(density_rl, weights=weights_rl, color=ps.dark_blue, linestyle=':', label=rl_label, histtype='step')
		if all_fig:
			ax_rl.hist(density_rl, weights=weights_rl, color=c_now, alpha=alpha, label=label, histtype='step')

		#perform a 2-sample KS test and retrieve the p-value
		ks_rl = ks_2samp(density_bf, density_rl)
		p_rl.append(ks_rl.pvalue)

#legend formatting
handles, labels = ax[0,0].get_legend_handles_labels()
labels_ord = [s for s in labels_ord if s in labels]
by_label = dict(zip(labels, handles))
ax[0,1].legend([by_label[l] for l in labels_ord], [l for l in labels_ord], loc=1)



######################
#### SAVING FILES ####
######################

#save the dictionaries with the aperture information
np.savez_compressed(bf_apers_file, **apers_bf_dict)
np.savez_compressed(rq_file, **rq_dict)
np.savez_compressed(rl_file, **rl_dict)

#make a Table containing the p-values
if incl_rl:
	t_ks = Table([gen.r_search_all, p_rq, p_rl], names=['r', 'p_rq', 'p_rl'])
else:
	t_ks = Table([gen.r_search_all, p_rq], names=['r', 'p_rq'])
ks_filename = gen.PATH_CATS + 'KS_test_pvalues.txt'
t_ks.write(ks_filename, format='ascii', overwrite=True)


#format and save histograms
f.tight_layout()
f.savefig(figname, dpi=300, bbox_inches='tight')


if all_fig:
	f_rq.tight_layout()
	f_rq.savefig(figname_rq, dpi=300, bbox_inches='tight')

	if incl_rl:
		f_rl.tight_layout()
		f_rl.savefig(figname_rl, dpi=300, bbox_inches='tight')


