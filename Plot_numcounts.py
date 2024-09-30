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

#toggle `switches' for varying functionality
plot_med = False		#plots the Schechter functions with the median parameter values instead of the true best-fit
plot_sims = False		#plot the simulated number counts required to detect overdensity 
plot_var_N0 = True	#plot the number counts fitted by varying N0 on the blank-field function
save_separate = False	#makes separate plots for each radius in addition to the main plot

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
#directory containing the simulated number counts (relevant only if plot_sims = True)
PATH_SIM_NC = PATH_CATS + 'Significance_tests/'

#get the radii used
radii = gen.r_search_all

#list the files containing results to plot
nc_files = [PATH_COUNTS + f'Differential_with_errs_{r:.1f}am_{gen.n_gal}{gen.gal_type}.npz' for r in radii]
cc_files = [PATH_COUNTS + f'Cumulative_with_errs_{r:.1f}am_{gen.n_gal}{gen.gal_type}.npz' for r in radii]

#if told to plot the simulated number counts, load the tables containing the relevant information
if plot_sims:
	nc_min_gals = Table.read(PATH_SIM_NC + 'Differential_min_gals_for_signal.txt', format='ascii')
	cc_min_gals = Table.read(PATH_SIM_NC + 'Cumulative_min_gals_for_signal.txt', format='ascii')
	#also define a suffix to add to the figure name if simulated counts included
	sim_suffix = '_with_sim_od'
else:
	sim_suffix = ''
plot_offset_sim = -0.01

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
ax_big1.set_ylabel(r'${\rm d}N/{\rm d}S$ (deg$^{-2}$ mJy$^{-1}$)', labelpad=20.)

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
if plot_med:
	nc_popt_S19 = nc_params_bf['S2COSMOS'][0]
	cc_popt_S19 = cc_params_bf['S2COSMOS'][0]
else:
	nc_popt_S19 = nc_params_bf['best_S2COSMOS']
	cc_popt_S19 = cc_params_bf['best_S2COSMOS']

plot_offset_S19 = 0.01


#table containing the best-fit N0 values when S0 and gamma are fixed to the blank-field values
if plot_var_N0:
	t_N0_nc = Table.read(PATH_PARAMS + 'Differential_N0_fits.txt', format='ascii')
	t_N0_cc = Table.read(PATH_PARAMS + 'Cumulative_N0_fits.txt', format='ascii')



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

	if save_separate:
		#differential counts
		fr1, axr_nc = plt.subplots(1, 1)
		#label x and y axes
		axr_nc.set_xlabel(r'$S_{850}$ (mJy)')
		axr_nc.set_ylabel(r'${\rm d}N/{rm d}S$ (deg$^{-2}$ mJy$^{-1}$)')

		#differential counts
		fr2, axr_cc = plt.subplots(1, 1)
		#label x and y axes
		axr_cc.set_xlabel(r'$S_{850}$ (mJy)')
		axr_cc.set_ylabel(r'$N(>S)$ (deg$^{-2}$)')

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

		if save_separate:
			#plot the results
			nc.plot_numcounts(
				bin_centres,
				y_nc,
				ax=axr_nc,
				masks=plot_masks_nc,
				data_kwargs=data_kwargs
				)
			nc.plot_numcounts(
				bin_edges[:-1],
				y_cc,
				ax=axr_cc,
				masks=plot_masks_cc,
				data_kwargs=data_kwargs
				)

	#########################
	#### ALL RQ GALAXIES ####
	#########################

	#settings to plot the combined results as black dots with errorbars
	if gen.gal_type == 'rq':
		label = 'All RQ analogues'
	else:
		label = 'All MLAGN/HLAGN analogues'
	labels_ord.append(label)
	data_kwargs = dict(color='k', label=label, linestyle='none', marker='o', ms=14., zorder=1)
	ebar_kwargs = dict(ecolor='k', zorder=0)

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
	nc_params = np.load(PATH_PARAMS + f'Differential_{r:.1f}am_{gen.n_gal}{gen.gal_type}.npz')
	cc_params = np.load(PATH_PARAMS + f'Cumulative_{r:.1f}am_{gen.n_gal}{gen.gal_type}.npz')

	#retrieve the best-fit parameters
	if plot_med:
		nc_popt = nc_params['ALL'][0]
		cc_popt = cc_params['ALL'][0]
	else:
		nc_popt = nc_params['best_ALL']
		cc_popt = cc_params['best_ALL']


	#plot the best-fit functions on the relevant axes
	x_range = np.linspace(bin_edges[0], bin_edges[-1], 100)
	ax_nc.plot(x_range, nc.schechter_model(x_range, nc_popt), c='k', zorder=1, lw=2)
	ax_cc.plot(x_range, nc.cumulative_model(x_range, cc_popt), c='k', zorder=1, lw=2)


	if save_separate:
		nc.plot_numcounts(
			bin_centres,
			y_nc,
			yerr=(ey_nc_lo, ey_nc_hi),
			ax=axr_nc,
			masks=plot_masks_nc,
			data_kwargs=data_kwargs,
			ebar_kwargs=ebar_kwargs
			)
		nc.plot_numcounts(
			bin_edges[:-1],
			y_cc,
			yerr=(ey_cc_lo, ey_cc_hi),
			ax=axr_cc,
			masks=plot_masks_cc,
			data_kwargs=data_kwargs,
			ebar_kwargs=ebar_kwargs
			)
		axr_nc.plot(x_range, nc.schechter_model(x_range, nc_popt), c='k', zorder=1, lw=2)
		axr_cc.plot(x_range, nc.cumulative_model(x_range, cc_popt), c='k', zorder=1, lw=2)


	#plot the rescaled blank-field function if told to do so
	if plot_var_N0:
		N0_nc = t_N0_nc['N0'][t_N0_nc['r'] == r][0]
		N0_cc = t_N0_cc['N0'][t_N0_cc['r'] == r][0]
		nc_popt_new = np.array([N0_nc, nc_popt_S19[1], nc_popt_S19[2]])
		cc_popt_new = np.array([N0_cc, cc_popt_S19[1], cc_popt_S19[2]])
		ax_nc.plot(x_range, nc.schechter_model(x_range, nc_popt_new), c='k', linestyle=':', zorder=1, lw=2)
		ax_cc.plot(x_range, nc.cumulative_model(x_range, cc_popt_new), c='k', linestyle=':', zorder=1, lw=2)

		if save_separate:
			axr_nc.plot(x_range, nc.schechter_model(x_range, nc_popt_new), c='k', linestyle=':', zorder=1, lw=2)
			axr_cc.plot(x_range, nc.cumulative_model(x_range, cc_popt_new), c='k', linestyle=':', zorder=1, lw=2)

	##################
	#### S2COSMOS ####
	##################

	#settings to plot the combined results as black dots with errorbars
	label = 'S2COSMOS (Simpson+19)'
	labels_ord.append(label)
	data_kwargs = dict(color=ps.crimson, label=label, linestyle='none', marker='D', ms=8., zorder=3, alpha=0.7)
	ebar_kwargs = dict(ecolor=ps.crimson, zorder=2, alpha=0.7)

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
	ax_nc.plot(x_range, nc.schechter_model(x_range, nc_popt_S19), c=ps.crimson, alpha=0.7, linestyle='--', zorder=3, lw=2)

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
	ax_cc.plot(x_range, nc.cumulative_model(x_range, cc_popt_S19), c=ps.crimson, alpha=0.7, linestyle='--', zorder=3, lw=2)



	if save_separate:
		#plot the differential number counts
		nc.plot_numcounts(
			S19_bin_centres,
			N_S19,
			yerr=(eN_S19_lo,eN_S19_hi),
			ax=axr_nc,
			offset=plot_offset_S19,
			masks=plot_masks_nc_S19,
			weights=weights_S19,
			data_kwargs=data_kwargs,
			ebar_kwargs=ebar_kwargs
			)
		#plot the best-fit Schechter function
		axr_nc.plot(x_range, nc.schechter_model(x_range, nc_popt_S19), c=ps.crimson, alpha=0.7, linestyle='--', zorder=3, lw=2)

		#plot the cumulative number counts
		nc.plot_numcounts(
			S19_bin_edges[:-1],
			c_S19,
			yerr=(ec_S19_lo,ec_S19_hi),
			ax=axr_cc,
			offset=plot_offset_S19,
			masks=plot_masks_cc_S19,
			weights=np.full(len(S19_bin_edges)-1, gen.A_s2c),
			data_kwargs=data_kwargs,
			ebar_kwargs=ebar_kwargs
			)
		#plot the best-fit Schechter function
		axr_cc.plot(x_range, nc.cumulative_model(x_range, cc_popt_S19), c=ps.crimson, alpha=0.7, linestyle='--', zorder=3, lw=2)

	#################################
	#### SIMULATED NUMBER COUNTS ####
	#################################

	if plot_sims:
		#load the differential and cumulative simulated number counts for this radius
		nc_sim_data = np.load(PATH_SIM_NC + f'Differential_with_errs_{r:.1f}am.npz')
		cc_sim_data = np.load(PATH_SIM_NC + f'Cumulative_with_errs_{r:.1f}am.npz')
		#get the minimum numbers of galaxies required to generate a signal in each
		Nmin_nc = int(nc_min_gals[nc_min_gals['r'] == r]['Nmin'][-1])
		Nmin_cc = int(cc_min_gals[cc_min_gals['r'] == r]['Nmin'][-1])
		#load the table containing the best-fit N0 values for each simulated number counts
		t_N0_nc = Table.read(PATH_SIM_NC + f'Differential_sig_test_results_{r:.1f}am.txt', format='ascii')
		t_N0_cc = Table.read(PATH_SIM_NC + f'Cumulative_sig_test_results_{r:.1f}am.txt', format='ascii')
		#get the best-fit value of N0
		N0_nc = t_N0_nc[t_N0_nc['N'] == Nmin_nc]['N0'][0]
		N0_cc = t_N0_cc[t_N0_cc['N'] == Nmin_cc]['N0'][0]
		#get the number counts bin info
		sim_bin_edges = nc_sim_data['bin_edges']
		sim_bin_centres = (sim_bin_edges[1:] + sim_bin_edges[:-1]) / 2.


		#settings to plot the results as green squares with errorbars
		label = 'Simulated overdensities'
		labels_ord.append(label)
		data_kwargs = dict(color=ps.teal, label=label, linestyle='none', marker='s', ms=8., zorder=5, alpha=0.7)
		ebar_kwargs = dict(ecolor=ps.teal, zorder=4, alpha=0.7)

		#retrieve the results for the combined dataset
		y_nc, ey_nc_lo, ey_nc_hi = nc_sim_data[f'{Nmin_nc}gals']
		#create masks for the sake of visualisation
		plot_masks_nc = nc.mask_numcounts(sim_bin_centres, y_nc, limits=False, exclude_all_zero=False, Smin=Smin)
		nc.plot_numcounts(
			sim_bin_centres,
			y_nc,
			yerr=(ey_nc_lo, ey_nc_hi),
			ax=ax_nc,
			offset=plot_offset_sim,
			masks=plot_masks_nc,
			data_kwargs=data_kwargs,
			ebar_kwargs=ebar_kwargs
			)
		#plot the best fit
		nc_popt_sim = [N0_nc, *nc_popt_S19[1:]]
		ax_nc.plot(x_range, nc.schechter_model(x_range, nc_popt_sim), c=ps.teal, alpha=0.7, linestyle=':', zorder=5, lw=2)


		#retrieve the results for the combined dataset
		y_cc, ey_cc_lo, ey_cc_hi = cc_sim_data[f'{Nmin_cc}gals']
		#create masks for the sake of visualisation
		plot_masks_cc = nc.mask_numcounts(sim_bin_edges[:-1], y_cc, limits=False, exclude_all_zero=False, Smin=Smin)
		nc.plot_numcounts(
			sim_bin_edges[:-1],
			y_cc,
			yerr=(ey_cc_lo, ey_cc_hi),
			ax=ax_cc,
			offset=plot_offset_sim,
			masks=plot_masks_cc,
			data_kwargs=data_kwargs,
			ebar_kwargs=ebar_kwargs
			)
		#plot the best fit
		cc_popt_sim = [N0_cc, *cc_popt_S19[1:]]
		ax_cc.plot(x_range, nc.cumulative_model(x_range, cc_popt_sim), c=ps.teal, alpha=0.7, linestyle=':', zorder=5, lw=2)


		if save_separate:
			nc.plot_numcounts(
				sim_bin_centres,
				y_nc,
				yerr=(ey_nc_lo, ey_nc_hi),
				ax=axr_nc,
				offset=plot_offset_sim,
				masks=plot_masks_nc,
				data_kwargs=data_kwargs,
				ebar_kwargs=ebar_kwargs
				)
			axr_nc.plot(x_range, nc.schechter_model(x_range, nc_popt_sim), c=ps.teal, alpha=0.7, linestyle=':', zorder=5, lw=2)

			nc.plot_numcounts(
				sim_bin_edges[:-1],
				y_cc,
				yerr=(ey_cc_lo, ey_cc_hi),
				ax=axr_cc,
				offset=plot_offset_sim,
				masks=plot_masks_cc,
				data_kwargs=data_kwargs,
				ebar_kwargs=ebar_kwargs
				)
			axr_cc.plot(x_range, nc.cumulative_model(x_range, cc_popt_sim), c=ps.teal, alpha=0.7, linestyle=':', zorder=5, lw=2)



	############################
	#### FORMATTING OF AXES ####
	############################

	#add labels indicating the serch radius used
	ax_nc.text(0.95, 0.95, r'$R = %.0f^{\prime}$'%r, transform=ax_nc.transAxes, ha='right', va='top')
	ax_cc.text(0.95, 0.95, r'$R = %.0f^{\prime}$'%r, transform=ax_cc.transAxes, ha='right', va='top')

	if i == (n_rows - 1):
		#add a legend in the bottom left corner, removing duplicate labels
		handles, labels = ax_nc.get_legend_handles_labels()
		labels_ord = [s for s in labels_ord if s in labels]
		by_label = dict(zip(labels, handles))
		ax_nc.legend([by_label[l] for l in labels_ord], [l for l in labels_ord], loc=3, prop={'size':19})
		ax_cc.legend([by_label[l] for l in labels_ord], [l for l in labels_ord], loc=3, prop={'size':19})

	if save_separate:
		axr_nc.text(0.95, 0.95, r'$R = %.0f^{\prime}$'%r, transform=axr_nc.transAxes, ha='right', va='top')
		axr_cc.text(0.95, 0.95, r'$R = %.0f^{\prime}$'%r, transform=axr_cc.transAxes, ha='right', va='top')
		axr_nc.legend([by_label[l] for l in labels_ord], [l for l in labels_ord], loc=3)
		axr_cc.legend([by_label[l] for l in labels_ord], [l for l in labels_ord], loc=3)

		#set the axes to log scale
		axr_nc.set_xscale('log')
		axr_nc.set_yscale('log')
		axr_cc.set_xscale('log')
		axr_cc.set_yscale('log')
		#set the minor tick locations on the x-axis
		axr_nc.set_xticks(xtick_min_locs, labels=xtick_min_labels, minor=True)
		axr_cc.set_xticks(xtick_min_locs, labels=xtick_min_labels, minor=True)
		#set the axes limits
		axr_nc.set_xlim(1.5, 25.)
		axr_cc.set_xlim(1.5, 25.)
		axr_nc.set_ylim(0.05, 2500.)
		axr_cc.set_ylim(0.05, 4000.)
			
		#force matplotlib to label with the actual numbers
		axr_nc.get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
		axr_cc.get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
		axr_nc.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:g}'))
		axr_cc.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:g}'))
		#minimise unnecesary whitespace
		fr1.tight_layout()
		fr2.tight_layout()
		frname1 = PATH_PLOTS + f'S850_number_counts_{r:.1f}_{gen.n_gal}{gen.gal_type}{sim_suffix}.png'
		frname2 = PATH_PLOTS + f'S850_cumulative_counts_{r:.1f}_{gen.n_gal}{gen.gal_type}{sim_suffix}.png'
		fr1.savefig(frname1, bbox_inches='tight', dpi=300)
		fr2.savefig(frname2, bbox_inches='tight', dpi=300)


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
figname = PATH_PLOTS + f'S850_number_counts_by_radius_{gen.n_gal}{gen.gal_type}{sim_suffix}_300dpi.png'
f.savefig(figname, bbox_inches='tight', dpi=300)

