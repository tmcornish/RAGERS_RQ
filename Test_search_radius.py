############################################################################################################
# A script for testing different search radii when looking for submm counterparts.
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
from matplotlib.lines import Line2D
import colorsys
import glob
from scipy.interpolate import LinearNDInterpolator
import scipy.optimize as opt
import general as gen
import plotstyle as ps
import numcounts as nc
import stats
import astrometry as ast
import emcee


#######################################################################################
########### FORMATTING FOR GRAPHS #####################################################

#formatting for graphs
plt.style.use(ps.styledict)

#formatting for arrows a la quiver
arrow_settings = ps.arrow_settings
#increase the arrow size from the default
arrow_settings['scale'] /= 1.5
arrow_settings['width'] *= 1.2

markers = ['o', 's', '^', '*', 'v']
colours = [ps.crimson, ps.teal, ps.magenta, ps.plum, ps.blue]

radii = [1., 2., 3., 4., 6.]

#create redshift bins for the RL galaxies
dz = 0.5
zbin_edges = np.arange(1., 3.+dz, dz)
zbin_centres = zbin_edges[:-1] + 0.5 * dz

nwalkers = 100
niter = 1000
#initial guesses for the fit parameters N0, S0, gamma
p0 = [5000., 3., 1.6]
#offsets for the initial walker positions from the initial guess values
offsets_init = [10., 0.01, 0.01]

flux_lim = 3.

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



#######################################################
###############    START OF SCRIPT    #################
#######################################################

#relevant paths
PATH_RAGERS = gen.PATH_RAGERS
PATH_CATS = gen.PATH_CATS
PATH_DATA = gen.PATH_DATA
PATH_PLOTS = gen.PATH_PLOTS


#load the table summarising the nmber counts results
S19_results_file = PATH_CATS + 'Simpson+19_number_counts_tab.txt'
S19_results = Table.read(S19_results_file, format='ascii')
#bin edges and centres for the differential number counts
S19_bin_edges = np.concatenate([np.array(S19_results['S850']), [22.]])
S19_bin_centres = (S19_bin_edges[:-1] + S19_bin_edges[1:]) / 2.
dS_S19 = S19_bin_edges[1:] - S19_bin_edges[:-1]

x_range_fit = np.linspace(S19_bin_edges.min(), S19_bin_edges.max(), 100)

#get randomised flux densities and completenesses
S850_npz_filename = PATH_CATS + 'S2COSMOS_randomised_S850.npz'
rand_data = np.load(S850_npz_filename)
S850_rand = rand_data['S850_rand']
comp_submm = rand_data['comp_rand']

#blank-field number counts
A_bf = 1.6
#construct the differential number counts
N_bf, eN_bf_lo, eN_bf_hi, counts_bf, weights_bf = nc.differential_numcounts(
	S850_rand,
	S19_bin_edges,
	A_bf,
	comp=comp_submm,
	incl_poisson=True)

#create appropriate masks for plotting the data
plot_masks_bf = nc.mask_numcounts(S19_bin_centres, N_bf, limits=True, exclude_all_zero=True, Smin=flux_lim)


labels_ord = [f'r = {r:.1f} arcmin' for r in radii] + ['S2COSMOS']

for i in range(len(radii)):
	offset_plot = 0.008 * ((-1.) ** ((i+1) % 2.)) * np.floor((i + 2.) / 2.)
	print(radii[i])
	#load the number counts data
	nc_data = np.load(PATH_CATS + f'Differential_numcounts_{radii[i]:.1f}.npz')

	#cycle through the redshift bins
	for j in range(len(zbin_centres)):
		#get the current redshift bin and print its bounds
		z = zbin_centres[j]
		print(f'{z-dz/2:.2f} < z < {z+dz/2.:.2f}')

		#plot data in the left-hand column if i is even; plot in the right-hand column if i is odd
		if (j % 2 == 0):
			row_z = int(j/2)			#the row in which the current subplot lies
			col_z = 0					#the column in which the current subplot lies
		else:
			row_z = int((j-1)/2)		#the row in which the current subplot lies
			col_z = 1					#the column in which the current subplot lies


		#if first time iterating to this redshift bin, plot blank-field results
		if i == 0:
			#plot the number counts
			nc.plot_numcounts(
				S19_bin_centres,
				N_bf,
				yerr=(eN_bf_lo,eN_bf_hi),
				ax=ax1[row_z,col_z],
				offset=0.,
				masks=plot_masks_bf,
				weights=weights_bf,
				data_kwargs=dict(
					color=ps.grey,
					label='S2COSMOS',
					linestyle='none',
					marker='D',
					zorder=100),
				ebar_kwargs=dict(
					ecolor=ps.grey,
					zorder=99
					),
				limit_kwargs=arrow_settings
				)

			#use MCMC to fit a Schechter function to the combined data for this redshift bin
			popt_bf, epopt_lo_bf, epopt_hi_bf = nc.fit_schechter_mcmc(
				S19_bin_centres[plot_masks_bf[0]],
				N_bf[plot_masks_bf[0]],
				(eN_bf_lo+eN_bf_hi)[plot_masks_bf[0]]/2.,
				nwalkers,
				niter,
				p0,
				offsets=offsets_init,
				plot_on_axes=True,
				ax=ax1[row_z,col_z],
				x_range=x_range_fit,
				color=ps.grey,
				add_text=False
				)
			bin_text = r'$%.1f \leq z < %.1f$'%(z-dz/2.,z+dz/2.)
			ax1[row_z,col_z].text(0.95, 0.95, bin_text, transform=ax1[row_z,col_z].transAxes, ha='right', va='top')

		#retrieve the number counts data for this search radius
		bin_edges = nc_data['bin_edges']
		bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.
		N_zbin, eN_zbin_lo, eN_zbin_hi = nc_data[f'zbin{j+1}']

		#create appropriate masks for plotting the data
		plot_masks_zbin = nc.mask_numcounts(bin_centres, N_zbin, limits=True, exclude_all_zero=True, Smin=flux_lim)

		#plot the number counts
		nc.plot_numcounts(
			bin_centres,
			N_zbin,
			yerr=(eN_zbin_lo,eN_zbin_hi),
			ax=ax1[row_z,col_z],
			offset=offset_plot,
			masks=plot_masks_zbin,
			data_kwargs=dict(
				color=colours[i],
				label=labels_ord[i],
				linestyle='none',
				marker=markers[i]
			),
			ebar_kwargs=dict(
				ecolor=colours[i]
				),
			limit_kwargs=arrow_settings
			)

		#use MCMC to fit a Schechter function to the combined data for this redshift bin
		popt_zbin, epopt_lo_zbin, epopt_hi_zbin = nc.fit_schechter_mcmc(
			bin_centres[plot_masks_zbin[0]],
			N_zbin[plot_masks_zbin[0]],
			(eN_zbin_lo+eN_zbin_hi)[plot_masks_zbin[0]]/2.,
			nwalkers,
			niter,
			p0,
			offsets=offsets_init,
			plot_on_axes=True,
			ax=ax1[row_z,col_z],
			x_range=x_range_fit,
			x_offset=offset_plot,
			color=colours[i],
			add_text=False
			)

		if i == len(zbin_centres):
			#add a legend in the bottom left corner, removing duplicate labels
			handles, labels = ax1[row_z,col_z].get_legend_handles_labels()
			labels_ord = [s for s in labels_ord if s in labels]
			by_label = dict(zip(labels, handles))
			ax1[row_z,col_z].legend([by_label[l] for l in labels_ord], [l for l in labels_ord], loc=3)

for i in range(len(ax1)):
	for j in range(len(ax1[0])):
		#set the axes to log scale
		ax1[i,j].set_xscale('log')
		ax1[i,j].set_yscale('log')
		#set the minor tick locations on the x-axis
		ax1[i,j].set_xticks(xtick_min_locs, labels=xtick_min_labels, minor=True)
		#set the axes limits
		ax1[i,j].set_xlim(1.5, 25.)
		ax1[i,j].set_ylim(0.05, 2500.)
		#force matplotlib to label with the actual numbers
		ax1[i,j].get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
		ax1[i,j].get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:g}'))
#minimise unnecesary whitespace
f1.tight_layout()
figname = PATH_PLOTS + f'Radius_test_diff_numcounts_zbinned.png'
f1.savefig(figname, bbox_inches='tight', dpi=300)
















