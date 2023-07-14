############################################################################################################
# A script for trying to recreate the number counts plot from Simpson+19.
############################################################################################################

#import modules/packages
import os, sys
import general as gen
import stats
import plotstyle as ps
import numcounts as nc
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy.interpolate import LinearNDInterpolator
from astropy.table import Table
from astropy.io import fits
import glob
from multiprocessing import Pool, cpu_count


if __name__ == '__main__':
		
	##################
	#### SETTINGS ####
	##################

	#toggle `switches' for additional functionality
	plot_cumulative = True		#make cumulative number counts as well as differential number counts
	randomise_fluxes = True		#randomly draw flux densities from possible values
	comp_corr = True			#apply the completeness corrections to the number counts
	randomise_comp = True		#calculate completeness corrections for each randomly drawn flux density
	fit_schechter = True		#fits Schechter functions to the results
	exclude_faint = True		#exclude faintest bins from Schechter fitting
	compare_errors = True		#compare the uncertainties from S19 with the reconstruction performed here
	main_only = gen.main_only	#use only sources from the MAIN region of S2COSMOS
	settings = [
		plot_cumulative,
		randomise_fluxes,
		comp_corr,
		randomise_comp,
		fit_schechter,
		exclude_faint,
		compare_errors,
		main_only
	]

	#print the chosen settings to the Terminal
	print(gen.colour_string('CHOSEN SETTINGS:', 'white'))
	settings_print = [
		'Construct cumulative number counts: ',
		'Randomise source flux densities for error analysis: ',
		'Apply completeness corrections to number counts: ',
		'Calculate completeness corrections for each randomly drawn flux density: ',
		'Fit Schechter functions to data: ',
		'Exclude faintest bins from Schechter fitting: ',
		'Compare uncertainties with Simpson+19: ',
		'Use MAIN region only: '
	]
	for i in range(len(settings_print)):
		if settings[i]:
			settings_print[i] += 'y'
		else:
			settings_print[i] += 'n'

	nsim = gen.nsim		#number of iterations to use if randomising the flux densities/completeness

	if randomise_fluxes:
		settings_print.append(f'Number of iterations for randomisation: {nsim}')

	if fit_schechter:
		#number of walkers and iterations to use in MCMC fitting
		nwalkers = 100
		niter = gen.nsim
		#offsets for the initial walker positions from the initial guess values
		offsets_init = [10., 0.01, 0.01]

		settings_print.append(f'Number of walkers to use for MCMC: {nwalkers}')
		settings_print.append(f'Number of iterations to use for MCMC: {niter}')

	print(gen.colour_string('\n'.join(settings_print), 'white'))

	#formatting of plots
	plt.style.use(ps.styledict)

	#######################################################
	###############    START OF SCRIPT    #################
	#######################################################

	#relevant paths
	PATH_RAGERS = gen.PATH_RAGERS
	PATH_CATS = gen.PATH_CATS
	PATH_DATA = gen.PATH_DATA
	PATH_PLOTS = gen.PATH_PLOTS
	PATH_SIMS = gen.PATH_SIMS

	if randomise_fluxes:
		#see if file exists containing randomised flux densities already
		npz_filename = PATH_SIMS + 'S2COSMOS_randomised_S850.npz'
		if not os.path.exists(npz_filename):
			gen.error_message(sys.argv[0], 'No file found for randomised flux densities. Run step 2 of the pipeline to create one.')
			exit()

	if fit_schechter:
		#see if directories exist for containing the outputs of the MCMC code
		PATH_PARAMS = PATH_CATS + 'Schechter_params/'
		PATH_POSTS = PATH_SIMS + 'Schechter_posteriors/'
		for P in [PATH_PARAMS, PATH_POSTS]:
			if not os.path.exists(P):
				os.system(f'mkdir -p {P}')

		#destination files for the best-fit parameters and uncertainties, and for the posteriors
		nc_params_file = PATH_PARAMS + f'Differential_S2COSMOS.npz'
		nc_post_file = PATH_POSTS + f'Differential_S2COSMOS.npz'
		#create dictionaries for containing these reuslts
		nc_params_dict, nc_post_dict = {}, {}

		if plot_cumulative:
			#destination files for the best-fit parameters and uncertainties, and for the posteriors
			cc_params_file = PATH_PARAMS + f'Cumulative_S2COSMOS.npz'
			cc_post_file = PATH_POSTS + f'Cumulative_S2COSMOS.npz'
			#set up dictionaries for these results
			cc_params_dict, cc_post_dict = {}, {}


	######################
	#### FIGURE SETUP ####
	######################

	#create the main figure (differential number counts)
	f1, ax1 = plt.subplots(1, 1)
	#label the axes
	ax1.set_xlabel(r'$S_{850~\mu{\rm m}}$ (mJy)')
	ax1.set_ylabel(r'$dN/dS$ (deg$^{-2}$ mJy$^{-1}$)')
	#set the axes to log scale
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	#override the weird formatting to which pyplot defaults when the scale is logged
	ax1.get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
	ax1.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:g}'))
	#suffix to use for figures in which the faintest bins have been excluded
	suffix_ef = ''

	if plot_cumulative:
		#create the figure (cumulative number counts)
		f2, ax2 = plt.subplots(1, 1)
		#label the axes
		ax2.set_xlabel(r'$S_{850~\mu{\rm m}}$ (mJy)')
		ax2.set_ylabel(r'$N(>S)$ (deg$^{-2}$)')
		#set the axes to log scale
		ax2.set_xscale('log')
		ax2.set_yscale('log')
		#override the weird formatting to which pyplot defaults when the scale is logged
		ax2.get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
		ax2.get_yaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:g}'))


	if compare_errors:
		#create the figure (errorbar comparison)
		f3, ax3 = plt.subplots(1, 1)
		#label the axes
		ax3.set_xlabel(r'$S_{S850~\mu{\rm m}}$ (mJy)')
		ax3.set_ylabel(r'$\sigma_{S850}^{\rm S19} / \sigma_{S850}^{\rm TC}$')
		#set the x-axis to log scale
		ax3.set_xscale('log')
		#override the weird formatting to which pyplot defaults when the scale is logged
		ax3.get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))


	#################################
	#### RESULTS FROM SIMPSON+19 ####
	#################################

	#load the table summarising the nmber counts results
	S19_results = Table.read(gen.S19_results_file, format='ascii')

	#bin edges and centres for the differential number counts
	bin_edges = np.concatenate([np.array(S19_results['S850']), [22.]])
	bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.
	#bin widths
	dS = bin_edges[1:] - bin_edges[:-1]

	if main_only:
		#differential number counts in each bin
		bin_heights = np.array(S19_results['dNM/dSp'])
		#upper and lower uncertainties on the bin counts
		e_upper = np.array(S19_results['E_dNM/dSp'])
		e_lower = np.array(S19_results['e_dNM/dSp'])
		#survey area in square degrees
		A = np.full(len(bin_centres), 1.6)
	else:
		#differential number counts in each bin
		bin_heights = np.array(S19_results['dNMS/dSp'])
		#upper and lower uncertainties on the bin counts
		e_upper = np.array(S19_results['E_dNMS/dSp'])
		e_lower = np.array(S19_results['e_dNMS/dSp'])
		#survey area in square degrees
		A = np.full(len(bin_centres), 2.6)

	#best-fit Schechter parameters (N0, S0, with their upper and lower uncertainties
	S19_fit_params = np.array([
		[5000., 3., 1.6],
		[1300., 0.6, 0.3],
		[1400., 0.5, 0.4]])

	x_range_plot = np.logspace(np.log10(bin_edges[0]), np.log10(bin_edges[-1]), 100)
	ax1.plot(x_range_plot, nc.schechter_model(x_range_plot, S19_fit_params[0]), c='k', zorder=11, label='Simpson+19 best fit')

	#create masks for plotting included vs excluded bins
	plot_masks = nc.mask_numcounts(bin_centres, bin_heights, limits=False, Smin=3.)
	#plot the number counts
	nc.plot_numcounts(
		bin_centres,
		bin_heights,
		yerr=(e_lower,e_upper),
		ax=ax1,
		offset=0.,
		data_kwargs=dict(
			color='k',
			label='Simpson+19 results',
			linestyle='none',
			marker='o',
			zorder=10),
		ebar_kwargs=dict(
			ecolor='k',
			elinewidth=2.,
			zorder=9
			)
		)


	if fit_schechter:
		print(gen.colour_string('Fitting to Simpson+19 differential number counts...', 'purple'))
		
		popt, elo_popt, ehi_popt = nc.fit_schechter_mcmc(
			bin_centres,
			bin_heights,
			(e_upper+e_lower)/2.,
			nwalkers,
			niter,
			S19_fit_params[0],
			offsets=offsets_init,
			plot_on_axes=True,
			ax=ax1,
			linestyle='--',
			x_range=x_range_plot,
			color='k',
			zorder=11,
			label='Simpson+19; my fit',
			add_text=False,
			fontsize=18.,
			xyfont=(2., 30.)
			)
		best_fit_str = [
			r'$N_{0} = %.0f^{+%.0f}_{-%.0f}$'%tuple(S19_fit_params[:,0]),
			r'$S_{0} = %.1f^{+%.1f}_{-%.1f}$'%tuple(S19_fit_params[:,1]),
			r'$\gamma = %.1f^{+%.1f}_{-%.1f}$'%tuple(S19_fit_params[:,2])
		]
		#add text with the best-fit parameters from MCMC
		for i in range(len(best_fit_str)):
			if i == 0:
				best_fit_str[i] += r' $(%.0f^{+%.0f}_{-%.0f})$'%(popt[i],ehi_popt[i],elo_popt[i])
			else:
				best_fit_str[i] += r' $(%.1f^{+%.1f}_{-%.1f})$'%(popt[i],ehi_popt[i],elo_popt[i])
		best_fit_str = '\n'.join(best_fit_str)
		ax1.text(2., 30., best_fit_str, color='k', ha='left', va='top', fontsize=18.)



	if plot_cumulative:
		if main_only:
			#differential number counts in each bin
			bin_heights_c = np.array(S19_results['NM'])
			#upper and lower uncertainties on the bin counts
			e_upper_c = np.array(S19_results['E_NM'])
			e_lower_c = np.array(S19_results['e_NM'])
		else:
			#differential number counts in each bin
			bin_heights_c = np.array(S19_results['NMS'])
			#upper and lower uncertainties on the bin counts
			e_upper_c = np.array(S19_results['E_NMS'])
			e_lower_c = np.array(S19_results['e_NMS'])

		#create masks for plotting included vs excluded bins
		plot_masks_c = nc.mask_numcounts(bin_edges[:-1], bin_heights_c, limits=False, Smin=3.)
		#plot the cumulative counts
		nc.plot_numcounts(
			bin_edges[:-1],
			bin_heights_c,
			yerr=(e_lower_c,e_upper_c),
			ax=ax2,
			offset=0.,
			masks=plot_masks_c,
			data_kwargs=dict(
				color='k',
				label='Simpson+19 results',
				linestyle='none',
				marker='o',
				zorder=10
				),
			ebar_kwargs=dict(
				ecolor='k',
				elinewidth=2.,
				zorder=9
				)
			)

		if fit_schechter:
			print(gen.colour_string('Fitting to Simpson+19 cumulative number counts...', 'purple'))
			
			popt, elo_popt, ehi_popt = nc.fit_cumulative_mcmc(
				bin_edges[:-1],
				bin_heights_c,
				(e_upper_c+e_lower_c)/2.,
				nwalkers,
				niter,
				S19_fit_params[0],
				offsets=offsets_init,
				plot_on_axes=True,
				ax=ax2,
				linestyle='--',
				x_range=x_range_plot,
				color='k',
				zorder=11,
				label='Simpson+19; my fit',
				add_text=True,
				fontsize=18.,
				xyfont=(2., 100.)
				) 

	#####################################
	#### ATTEMPT TO RECREATE RESULTS ####
	#####################################

	#S2COSMOS catalogue
	S19_cat = gen.S19_cat
	data_submm = Table.read(S19_cat, format='fits')
	#retrieve the deboosted flux densities and uncertainties, and RMS noise
	S850, eS850_lo, eS850_hi, RMS, *_, comp_cat = gen.get_relevant_cols_S19(data_submm, main_only=main_only)

	if randomise_fluxes:
		rand_data = np.load(npz_filename)
		S850_rand = rand_data['S850_rand']
	else:
		S850_rand = S850[:]

	#construct the differential number counts
	N, eN_lo, eN_hi, counts, weights = nc.differential_numcounts(S850_rand, bin_edges, A, incl_poisson=True)
	#plot these results
	plot_offset = 0.01

	plot_masks = nc.mask_numcounts(bin_centres, N, limits=False, Smin=3.)
	#plot the number counts
	nc.plot_numcounts(
		bin_centres,
		N,
		yerr=(eN_lo,eN_hi),
		ax=ax1,
		offset=plot_offset,
		masks=plot_masks,
		weights=weights,
		data_kwargs=dict(
			color=ps.grey,
			label='S2COSMOS catalogue',
			linestyle='none',
			marker='s',
			zorder=2),
		ebar_kwargs=dict(
			ecolor=ps.grey,
			elinewidth=2.
			)
		)

	if plot_cumulative:
		c, ec_lo, ec_hi, cumcounts = nc.cumulative_numcounts(counts=counts, A=A)

		plot_masks_c = nc.mask_numcounts(bin_edges[:-1], c, limits=False, Smin=3.)

		nc.plot_numcounts(
			bin_edges[:-1],
			c,
			yerr=(ec_lo,ec_hi),
			ax=ax2,
			offset=plot_offset,
			masks=plot_masks_c,
			weights=weights,
			data_kwargs=dict(
				color=ps.grey,
				label='S2COSMOS catalogue',
				linestyle='none',
				marker='s',
				zorder=2),
			ebar_kwargs=dict(
				ecolor=ps.grey,
				elinewidth=2.,
				zorder=1
				)
			)


	##################################
	#### COMPLETENESS CORRECTIONS ####
	##################################

	if comp_corr:
		print(gen.colour_string('Retrieving completeness corrections...', 'purple'))

		if randomise_comp and randomise_fluxes:
			if 'comp_rand' in rand_data.files:
				comp_s2c = rand_data['comp_rand']
			else:
				gen.error_message(sys.argv[0], 'Randomised completenesses have not been generated. Run step 2 of the pipeline.')
		else:
			comp_s2c = comp_cat[:]	

		print(gen.colour_string('Applying completeness corrections...', 'purple'))

		N, eN_lo, eN_hi, counts_comp_corr, weights = nc.differential_numcounts(S850_rand, bin_edges, A, comp=comp_s2c, incl_poisson=True)
			
		#write the results as a table to a FITS file
		t_results = Table([bin_centres, N, eN_lo, eN_hi], names=['S850', 'N_comp_corr', 'eN_lo', 'eN_hi'])
		t_results.write(PATH_CATS + 'S19_differential_number_counts_recreated.fits', overwrite=True)

		#plot these results
		plot_offset = -0.01

		plot_masks = nc.mask_numcounts(bin_centres, N, limits=False, Smin=3.)
		#plot the number counts
		nc.plot_numcounts(
			bin_centres,
			N,
			yerr=(eN_lo,eN_hi),
			ax=ax1,
			offset=plot_offset,
			masks=plot_masks,
			weights=weights,
			data_kwargs=dict(
				color=ps.crimson,
				label='S2COSMOS catalogue (comp. corr.)',
				linestyle='none',
				marker='D',
				zorder=3),
			ebar_kwargs=dict(
				ecolor=ps.crimson,
				elinewidth=2.,
				zorder=2
				)
			)

		if plot_cumulative:
			c, ec_lo, ec_hi, cumcounts = nc.cumulative_numcounts(counts_comp_corr, A=A)

			plot_masks_c = nc.mask_numcounts(bin_edges[:-1], c, limits=False, Smin=3.)
			#plot these results
			nc.plot_numcounts(
				bin_edges[:-1],
				c,
				yerr=(ec_lo,ec_hi),
				ax=ax2,
				offset=plot_offset,
				masks=plot_masks_c,
				weights=weights,
				data_kwargs=dict(
					color=ps.crimson,
					label='S2COSMOS catalogue (comp. corr.)',
					linestyle='none',
					marker='D',
					zorder=2),
				ebar_kwargs=dict(
					ecolor=ps.crimson,
					elinewidth=2.,
					zorder=1
					)
				)


	#####################################
	#### FITTING SCHECHTER FUNCTIONS ####
	#####################################

	if fit_schechter:
		print(gen.colour_string('Fitting Schechter functions...', 'purple'))
		
		#label to use for the best-fit
		fit_label = 'Fit to catalogue'
		#colour to use for the best-fit curve
		if comp_corr:
			fit_colour = ps.crimson
			fit_label += ' (comp. corr.)'
		else:
			fit_colour = ps.grey

		#if told to exclude faint bins, set the lowest index to 2
		if exclude_faint:
			#suffix to use for figures in which the faintest bins have been excluded
			suffix_ef = '_excl_faint'
		else:
			suffix_ef = ''

		popt_diff, elo_popt_diff, ehi_popt_diff, sampler_diff = nc.fit_schechter_mcmc(
			bin_centres[plot_masks[0]], 
			N[plot_masks[0]], 
			(eN_hi+eN_lo)[plot_masks[0]]/2., 
			nwalkers,
			niter,
			S19_fit_params[0],
			offsets=offsets_init,
			return_sampler=True,
			plot_on_axes=True,
			ax=ax1,
			x_range=x_range_plot,
			x_offset=plot_offset,
			color=fit_colour,
			zorder=11,
			label=fit_label,
			add_text=True,
			fontsize=18.,
			xyfont=(9., 1000.),
			fc=fit_colour
			)
		#add the results to the relevant dictionaries
		nc_params_dict[gen.s2c_key] = np.array([popt_diff, elo_popt_diff, ehi_popt_diff])
		nc_post_dict[gen.s2c_key] = sampler_diff.flatchain
		#save the results to the files
		np.savez_compressed(nc_params_file, **nc_params_dict)
		np.savez_compressed(nc_post_file, **nc_post_dict)

		if plot_cumulative:

			popt_cumul, elo_popt_cumul, ehi_popt_cumul, sampler_cumul = nc.fit_cumulative_mcmc(
				bin_edges[:-1][plot_masks_c[0]], 
				c[plot_masks_c[0]], 
				(ec_hi+ec_lo)[plot_masks_c[0]]/2., 
				nwalkers,
				niter,
				S19_fit_params[0],
				offsets=offsets_init,
				return_sampler=True,
				plot_on_axes=True,
				ax=ax2,
				x_range=x_range_plot,
				x_offset=plot_offset,
				color=fit_colour,
				zorder=11,
				label=fit_label,
				add_text=True,
				fontsize=18.,
				xyfont=(9., 1000.),
				fc=fit_colour
				) 
			cc_params_dict[gen.s2c_key] = np.array([popt_cumul, elo_popt_cumul, ehi_popt_cumul])
			cc_post_dict[gen.s2c_key] = sampler_cumul.flatchain
			#save the results to the files
			np.savez_compressed(cc_params_file, **cc_params_dict)
			np.savez_compressed(cc_post_file, **cc_post_dict)


	##################################
	#### UNCERTAINTIES COMPARISON ####
	##################################

	if compare_errors:
		print(gen.colour_string('Comparing uncertainties...', 'purple'))
		#calculate ratio of the uncertainties to those from Simpson+19
		r_err_upper = e_upper / eN_hi
		r_err_lower = e_lower / eN_lo

		#plot the 1:1 ratio
		ax3.axhline(1., c=ps.grey, alpha=0.4)
		#plot these as functions of the flux density
		ax3.plot(bin_centres, r_err_upper, marker='D', c=ps.magenta, label=r'$\sigma_{\rm upper}$')
		ax3.plot(bin_centres, r_err_lower, marker='s', c=ps.dark_blue, label=r'$\sigma_{\rm lower}$')



	##################################
	#### SAVING FIGURES AND FILES ####
	##################################

	print(gen.colour_string('Formatting and saving figures...', 'purple'))

	ax1.legend()
	#set the minor tick locations on the x-axis
	xtick_min_locs = list(np.arange(2,10,1)) + [20]
	xtick_min_labels = [2, 5, 20]
	ax1.set_xticks(xtick_min_locs, labels=[f'{s:g}' if s in xtick_min_labels else '' for s in xtick_min_locs], minor=True)

	#minimise unnecesary whitespace
	f1.tight_layout()
	#save the figure
	f1.savefig(PATH_PLOTS + f'Simpson+19_number_counts_improved_comp{suffix_ef}.png', dpi=300)

	if plot_cumulative:
		ax2.legend()
		ax2.set_xticks(xtick_min_locs, labels=[f'{s:g}' if s in xtick_min_labels else '' for s in xtick_min_locs], minor=True)

		#minimise unnecesary whitespace
		f2.tight_layout()
		#save the figure
		f2.savefig(PATH_PLOTS + f'Simpson+19_cumulative_counts_improved_comp{suffix_ef}.png', dpi=300)


	if compare_errors:
		ax3.legend()
		ax3.set_xticks(xtick_min_locs, labels=[f'{s:g}' if s in xtick_min_labels else '' for s in xtick_min_locs], minor=True)
		#minimise unnecesary whitespace
		f3.tight_layout()
		#save the figure
		f3.savefig(PATH_PLOTS + 'Simpson+19_number_counts_uncertainty_comparison_improved_comp.png', dpi=300)

	print(gen.colour_string('Done!', 'purple'))


