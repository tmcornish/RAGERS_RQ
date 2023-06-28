############################################################################################################
# A script for plotting the sub-mm number counts calculated in the previous step of the pipeline.
############################################################################################################

#import modules/packages
import os, sys
import general as gen
import plotstyle as ps
import numcounts as nc
import stats
from matplotlib import pyplot as plt
from astropy.table import Table
import numpy as np
import glob
import matplotlib as mpl

##################
#### SETTINGS ####
##################

#formatting of plots
plt.style.use(ps.styledict)

#minimum flux density allowed when fitting (mJy)
Smin = gen.Smin

#######################################################
###############    START OF SCRIPT    #################
#######################################################

#relevant paths
PATH_CATS = gen.PATH_CATS
PATH_PLOTS = gen.PATH_PLOTS
PATH_SIMS = gen.PATH_SIMS
PATH_PARAMS = PATH_CATS + 'Schechter_params/'
PATH_COUNTS = PATH_CATS + 'Number_counts/'

#get the radii used
radii = gen.r_search_all

#list the files containing results to plot
nc_files = [PATH_COUNTS + f'Differential_with_errs_{r:.1f}am_{gen.n_rq}rq.npz' for r in radii]
cc_files = [PATH_COUNTS + f'Cumulative_with_errs_{r:.1f}am_{gen.n_rq}rq.npz' for r in radii]


######################
#### FIGURE SETUP ####
######################

#number of rows of panels
n_rows = len(nc_files)
#create the figure
f = plt.figure(figsize=(2.*ps.x_size, 0.6*n_rows*ps.y_size))
#set up a dictionary for the various axes
ax = {}

#locations at which minor ticks should be placed on each axis, and where they should be labeled
xtick_min_locs = list(np.arange(2,10,1)) + [20]
xtick_min_labels = [2, 5, 20]
xtick_min_labels = [f'{s:g}' if s in xtick_min_labels else '' for s in xtick_min_locs]

#to label axes with common labels, create a big subplot, make it invisible, and label its axes
ax_big1 = f.add_subplot(121, frameon=False)
ax_big1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, which='both')
#label x and y axes
ax_big1.set_xlabel(r'$S_{850}$ (mJy)', labelpad=10.)
ax_big1.set_ylabel(r'$dN/dS$ (deg$^{-2}$ mJy$^{-1}$)', labelpad=20.)

#create a second big subplot covering the second column make it invisible, and label its axes
ax_big2 = f.add_subplot(122, frameon=False)
ax_big2.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, which='both')
#label x and y axes
ax_big2.set_xlabel(r'$S_{850}$ (mJy)', labelpad=10.)
ax_big2.set_ylabel(r'$N(>S)$ (deg$^{-2}$)', labelpad=20.)

#############################
#### BLANK_FIELD RESULTS ####
#############################

#load the differential and cumulative counts for the blank field data
data_nc_bf = np.load(PATH_COUNTS + 'Differential_with_errs_bf.npz')
data_cc_bf = np.load(PATH_COUNTS + 'Cumulative_with_errs_bf.npz')

#get the bin edges used for the blank field results
S19_bin_edges = data_nc_bf['bin_edges']
S19_bin_centres = (S19_bin_edges[:-1] + S19_bin_edges[1:]) / 2.

#retrieve the results for the S2COSMOS dataset
N_S19, eN_S19_lo, eN_S19_hi = data_nc_bf['S2COSMOS']
c_S19, ec_S19_lo, ec_S19_hi = data_cc_bf['S2COSMOS']
#weights for each bin
weights_S19 = data_nc_bf['w_S2COSMOS']
#masks for visualisation
plot_masks_nc_S19 = nc.mask_numcounts(S19_bin_centres, N_S19, limits=False, Smin=gen.Smin)
plot_masks_cc_S19 = nc.mask_numcounts(S19_bin_edges[:-1], c_S19, limits=False, Smin=gen.Smin)


#load the best-fit parameters for the blank field
nc_params_bf = np.load(PATH_PARAMS + f'Differential_bf.npz')
cc_params_bf = np.load(PATH_PARAMS + f'Cumulative_bf.npz')

#retrieve the best-fit parameters
nc_popt_S19, enc_popt_lo_S19, enc_popt_hi_S19 = nc_params_bf['S2COSMOS']
cc_popt_S19, ecc_popt_lo_S19, ecc_popt_hi_S19 = cc_params_bf['S2COSMOS']

plot_offset_S19 = 0.01


#########################
#### PLOTTING COUNTS ####
#########################

#cycle through the search radii used
for i in range(n_rows):

	#get the current radius
	r = radii[i]

	#calculate the index to assign the subplots for this set of results
	nc_idx = (i * 2) + 1
	cc_idx = (i + 1) * 2

	#add a subplot with this index
	if i == 0:
		ax_nc = f.add_subplot(n_rows, 2, nc_idx)
		ax_cc = f.add_subplot(n_rows, 2, cc_idx)
	else:
		ax_nc = f.add_subplot(n_rows, 2, nc_idx, sharex=ax['ax1'], sharey=ax['ax1'])
		ax_cc = f.add_subplot(n_rows, 2, cc_idx, sharex=ax['ax2'], sharey=ax['ax2'])

	ax[f'ax{nc_idx}'] = ax_nc
	ax[f'ax{cc_idx}'] = ax_cc

	#load the data from the numpy archive files
	data_nc = np.load(nc_files[i])
	data_cc = np.load(cc_files[i])
	#get the bin edges and calculate the bin centres
	bin_edges = data_nc['bin_edges']
	bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.

	#labels in intended order for legend
	labels_ord = []

	################################
	#### INDIVIDUAL RQ GALAXIES ####
	################################

	#get the IDs of all the RAGERS RL galaxies
	IDs = [s for s in data_nc.files if 'bin' not in s and s[:2] != 'w_' and s != 'ALL']

	#cycle through the individual galaxies
	for ID in IDs:
		#settings to plot the results as a grey line
		data_kwargs = dict(color=ps.grey, alpha=0.2, zorder=0)

		#retrieve the results for the current dataset
		y_nc, ey_nc_lo, ey_nc_hi = data_nc[ID]
		#create masks for the sake of visualisation
		plot_masks_nc = nc.mask_numcounts(bin_centres, y_nc, limits=False, exclude_all_zero=False, Smin=Smin)
		#plot the results
		nc.plot_numcounts(
			bin_centres,
			y_nc,
			ax=ax_nc,
			masks=plot_masks_nc,
			data_kwargs=data_kwargs
			)

		#retrieve the results for the current dataset
		y_cc, ey_cc_lo, ey_cc_hi = data_cc[ID]
		#create masks for the sake of visualisation
		plot_masks_cc = nc.mask_numcounts(bin_edges[:-1], y_cc, limits=False, exclude_all_zero=False, Smin=Smin)
		#plot the results
		nc.plot_numcounts(
			bin_edges[:-1],
			y_cc,
			ax=ax_cc,
			masks=plot_masks_cc,
			data_kwargs=data_kwargs
			)

	#########################
	#### ALL RQ GALAXIES ####
	#########################

	#settings to plot the combined results as black dots with errorbars
	label = 'All RQ analogues'
	labels_ord.append(label)
	data_kwargs = dict(color='k', label=label, linestyle='none', marker='o', ms=14., zorder=5)
	ebar_kwargs = dict(ecolor='k', zorder=4)

	#retrieve the results for the combined dataset
	y_nc, ey_nc_lo, ey_nc_hi = data_nc['ALL']
	#create masks for the sake of visualisation
	plot_masks_nc = nc.mask_numcounts(bin_centres, y_nc, limits=False, exclude_all_zero=False, Smin=Smin)
	nc.plot_numcounts(
		bin_centres,
		y_nc,
		yerr=(ey_nc_lo, ey_nc_hi),
		ax=ax_nc,
		masks=plot_masks_nc,
		data_kwargs=data_kwargs,
		ebar_kwargs=ebar_kwargs
		)

	#retrieve the results for the combined dataset
	y_cc, ey_cc_lo, ey_cc_hi = data_cc['ALL']
	#create masks for the sake of visualisation
	plot_masks_cc = nc.mask_numcounts(bin_edges[:-1], y_cc, limits=False, exclude_all_zero=False, Smin=Smin)
	nc.plot_numcounts(
		bin_edges[:-1],
		y_cc,
		yerr=(ey_cc_lo, ey_cc_hi),
		ax=ax_cc,
		masks=plot_masks_cc,
		data_kwargs=data_kwargs,
		ebar_kwargs=ebar_kwargs
		)

	#load the best-fit parameters for this radius
	nc_params = np.load(PATH_PARAMS + f'Differential_{r:.1f}am_{gen.n_rq}rq.npz')
	cc_params = np.load(PATH_PARAMS + f'Cumulative_{r:.1f}am_{gen.n_rq}rq.npz')

	#retrieve the best-fit parameters
	nc_popt, enc_popt_lo, enc_popt_hi = nc_params['ALL']
	cc_popt, ecc_popt_lo, ecc_popt_hi = cc_params['ALL']


	#plot the best-fit functions on the relevant axes
	x_range = np.linspace(bin_edges[0], bin_edges[-1], 100)
	ax_nc.plot(x_range, nc.schechter_model(x_range, nc_popt), c='k')
	ax_cc.plot(x_range, nc.cumulative_model(x_range, cc_popt), c='k')



	##################
	#### S2COSMOS ####
	##################

	#settings to plot the combined results as black dots with errorbars
	label = 'S2COSMOS (Simpson+19)'
	labels_ord.append(label)
	data_kwargs = dict(color=ps.crimson, label=label, linestyle='none', marker='D', ms=8., zorder=3, alpha=0.5)
	ebar_kwargs = dict(ecolor=ps.crimson, zorder=2, alpha=0.5)

	#plot the differential number counts
	nc.plot_numcounts(
		S19_bin_centres,
		N_S19,
		yerr=(eN_S19_lo,eN_S19_hi),
		ax=ax_nc,
		offset=plot_offset_S19,
		masks=plot_masks_nc_S19,
		weights=weights_S19,
		data_kwargs=data_kwargs,
		ebar_kwargs=ebar_kwargs
		)
	#plot the best-fit Schechter function
	ax_nc.plot(x_range, nc.schechter_model(x_range, nc_popt_S19), c=ps.crimson, alpha=0.5, linestyle='--')

	#plot the cumulative number counts
	nc.plot_numcounts(
		S19_bin_edges[:-1],
		c_S19,
		yerr=(ec_S19_lo,ec_S19_hi),
		ax=ax_cc,
		offset=plot_offset_S19,
		masks=plot_masks_cc_S19,
		weights=np.full(len(S19_bin_edges)-1, gen.A_s2c),
		data_kwargs=data_kwargs,
		ebar_kwargs=ebar_kwargs
		)
	#plot the best-fit Schechter function
	ax_cc.plot(x_range, nc.cumulative_model(x_range, cc_popt_S19), c=ps.crimson, alpha=0.5, linestyle='--')


	############################
	#### FORMATTING OF AXES ####
	############################

	#add labels indicating the serch radius used
	ax_nc.text(0.95, 0.95, r'$R = %.0f^{\prime}$'%r, transform=ax_nc.transAxes, ha='right', va='top')
	ax_cc.text(0.95, 0.95, r'$R = %.0f^{\prime}$'%r, transform=ax_cc.transAxes, ha='right', va='top')

	#add a legend in the bottom left corner, removing duplicate labels
	handles, labels = ax_nc.get_legend_handles_labels()
	labels_ord = [s for s in labels_ord if s in labels]
	by_label = dict(zip(labels, handles))
	ax_nc.legend([by_label[l] for l in labels_ord], [l for l in labels_ord], loc=3)
	ax_cc.legend([by_label[l] for l in labels_ord], [l for l in labels_ord], loc=3)


##############################
#### FORMATTING OF FIGURE ####
##############################

#increase the spacing between the columns
f.subplots_adjust(wspace=0.1)

#set the axes to log scale
ax['ax1'].set_xscale('log')
ax['ax1'].set_yscale('log')
ax['ax2'].set_xscale('log')
ax['ax2'].set_yscale('log')
#set the minor tick locations on the x-axis
ax['ax1'].set_xticks(xtick_min_locs, labels=xtick_min_labels, minor=True)
ax['ax2'].set_xticks(xtick_min_locs, labels=xtick_min_labels, minor=True)
#set the axes limits
ax['ax1'].set_xlim(1.5, 25.)
ax['ax2'].set_xlim(1.5, 25.)
ax['ax1'].set_ylim(0.05, 2500.)
ax['ax2'].set_ylim(0.05, 4000.)
	
#force matplotlib to label with the actual numbers
ax['ax1'].get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
ax['ax2'].get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
ax['ax1'].get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:g}'))
ax['ax2'].get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:g}'))
#minimise unnecesary whitespace
f.tight_layout()
figname = PATH_PLOTS + f'S850_number_counts_by_radius_{gen.n_rq}rq.png'
f.savefig(figname, bbox_inches='tight', dpi=300)

