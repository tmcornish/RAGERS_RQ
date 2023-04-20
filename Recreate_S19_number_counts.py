############################################################################################################
# A script for trying to recreate the number counts plot from Simpson+19.
############################################################################################################

#import modules/packages
import general as gen
import stats
import plotstyle as ps
import numcounts as nc
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from astropy.table import Table
import glob
from multiprocessing import Pool, cpu_count

##################
#### SETTINGS ####
##################

#toggle `switches' for additional functionality
use_cat_from_paper = True	#use the unedited catalogue downloaded from the Simpson+19 paper website
plot_cumulative = True		#make cumulative number counts as well as differential number counts
randomise_fluxes = True		#randomly draw flux densities from possible values
comp_corr = True			#apply the completeness corrections to the number counts
plot_comp = True			#plot the completeness as a function of flux density and RMS
plot_data_on_comp = True	#plot the positions of each source in RMS-S850 space on the completeness figure
fit_schechter = True		#fits Schechter functions to the results
compare_errors = True		#compare the uncertainties from S19 with the reconstruction performed here
settings = [
	use_cat_from_paper,
	plot_cumulative,
	randomise_fluxes,
	comp_corr,
	plot_comp,
	plot_data_on_comp,
	fit_schechter,
	compare_errors
]

#print the chosen settings to the Terminal
print(gen.colour_string('CHOSEN SETTINGS:', 'white'))
settings_print = [
	'Use catalogue linked to Simpson+19 paper: ',
	'Construct cumulative number counts: ',
	'Randomise source flux densities for error analysis: ',
	'Apply completeness corrections to number counts: ',
	'Plot the completeness as a function S850 and RMS: ',
	'Show sources on completeness plot: ',
	'Fit Schechter functions to data: ',
	'Compare uncertainties with Simpson+19: '
]
for i in range(len(settings_print)):
	if settings[i]:
		settings_print[i] += 'y'
	else:
		settings_print[i] += 'n'

nsim = 10000		#number of iterations to use if randomising the flux densities/completeness

if randomise_fluxes:
	settings_print.append(f'Number of iterations for randomisation: {nsim}')

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

if plot_comp:
	#create the figure (completeness plot)
	f3, ax3 = plt.subplots(1, 1)
	#label the axes
	ax3.set_xlabel(r'$S_{850~\mu{\rm m}}$ (mJy)')
	ax3.set_ylabel(r'$\sigma_{\rm inst}$ (mJy)')

if compare_errors:
	#create the figure (errorbar comparison)
	f4, ax4 = plt.subplots(1, 1)
	#label the axes
	ax4.set_xlabel(r'$S_{S850~\mu{\rm m}}$ (mJy)')
	ax4.set_ylabel(r'$\sigma_{S850}^{\rm S19} / \sigma_{S850}^{\rm TC}$')
	#set the x-axis to log scale
	ax4.set_xscale('log')
	#override the weird formatting to which pyplot defaults when the scale is logged
	ax4.get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))


#################################
#### RESULTS FROM SIMPSON+19 ####
#################################

#load the table summarising the nmber counts results
S19_results_file = PATH_CATS + 'Simpson+19_number_counts_tab.txt'
S19_results = Table.read(S19_results_file, format='ascii')

#bin edges and centres for the differential number counts
bin_edges = np.concatenate([np.array(S19_results['S850']), [22.]])
bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.
#bin widths
dS = bin_edges[1:] - bin_edges[:-1]

#differential number counts in each bin
bin_heights = np.array(S19_results['dNMS/dSp'])
#upper and lower uncertainties on the bin counts
e_upper = np.array(S19_results['E_dNMS/dSp'])
e_lower = np.array(S19_results['e_dNMS/dSp'])

#best-fit Schechter parameters (N0, S0, with their upper and lower uncertainties
S19_fit_params = np.array([
	[5000., 3., 1.6],
	[1300., 0.6, 0.3],
	[1400., 0.5, 0.4]])

#plot the differential number counts
ax1.plot(bin_centres, bin_heights, marker='o', c='k', linestyle='none', zorder=10, label='Simpson+19 results')
ax1.errorbar(bin_centres, bin_heights, yerr=(e_lower, e_upper), ecolor='k', fmt='none', zorder=9)

#plot the best-fit Schechter function
x_range_plot = np.logspace(np.log10(bin_edges[0]), np.log10(bin_edges[-1]), 100)
ax1.plot(x_range_plot, nc.schechter_model(x_range_plot, S19_fit_params[0]), c='k', zorder=11, label='Simpson+19 best fit')
#add text with the best-fit parameters
best_fit_str = [
	r'$N_{0} = %.0f^{+%.0f}_{-%.0f}$'%tuple(S19_fit_params[:,0]),
	r'$S_{0} = %.1f^{+%.1f}_{-%.1f}$'%tuple(S19_fit_params[:,1]),
	r'$\gamma = %.1f^{+%.1f}_{-%.1f}$'%tuple(S19_fit_params[:,2])
]
best_fit_str = '\n'.join(best_fit_str)
ax1.text(9., 600., best_fit_str, color='k', ha='left', va='top', fontsize=18.)


if plot_cumulative:
	#cumulative number counts in each bin
	bin_heights_c = np.array(S19_results['NMS'])
	#upper and lower uncertainties
	e_upper_c = np.array(S19_results['E_NMS'])
	e_lower_c = np.array(S19_results['e_NMS'])
	#plot these data
	ax2.plot(bin_edges[:-1], bin_heights_c, marker='o', c='k', linestyle='none', zorder=10, label='Simpson+19 results')
	ax2.errorbar(bin_edges[:-1], bin_heights_c, yerr=(e_lower_c, e_upper_c), ecolor='k', fmt='none', zorder=9)

	if fit_schechter:
		#attempt a fit to the S19 results and print/plot the best fit
		popt, _ = stats.chisq_minimise(bin_edges[:-1], bin_heights_c, (e_upper_c + e_lower_c) / 2., nc.schechter_model, S19_fit_params[0])
		ax2.plot(x_range_plot, nc.schechter_model(x_range_plot, popt), c='k', zorder=11, label='Simpson+19 best fit')
		#add text with the best-fit parameters
		best_fit_str = [
			r'$N_{0} = %.0f$'%popt[0],
			r'$S_{0} = %.1f$'%popt[1],
			r'$\gamma = %.1f$'%popt[2]]
		best_fit_str = '\n'.join(best_fit_str)
		ax2.text(10., 1000., best_fit_str, color='k', ha='left', va='top', fontsize=18.)



#####################################
#### ATTEMPT TO RECREATE RESULTS ####
#####################################

if use_cat_from_paper:
	#catalogue containing submm data for S2COSMOS sources
	SUBMM_CAT = PATH_CATS + 'Simpson+19_S2COSMOS_source_cat.fits'
	data_submm = Table.read(SUBMM_CAT, format='fits')
	#get the (deboosted) 850 µm flux densities and the uncertainties
	S850 = data_submm['S850-deb']
	eS850_lo = data_submm['e_S850-deb']
	eS850_hi = data_submm['E_S850-deb']
	#also get the RMS
	RMS = data_submm['e_S850-obs']
else:
	#catalogue containing submm data for S2COSMOS sources
	SUBMM_CAT = PATH_CATS + 'S2COSMOS_sourcecat850_Simpson18.fits'
	data_submm = Table.read(SUBMM_CAT, format='fits')
	#get the (deboosted) 850 µm flux densities and the uncertainties
	S850 = data_submm['S_deboost']
	eS850_lo = data_submm['S_deboost_errlo']
	eS850_hi = data_submm['S_deboost_errhi']
	#also get the RMS
	RMS = data_submm['RMS']

if randomise_fluxes:
	S850_rand = np.array([stats.random_asymmetric_gaussian(S850[i], eS850_lo[i], eS850_hi[i], nsim) for i in range(len(S850))]).T
else:
	S850_rand = S850[:]

#survey area in square degrees
A = np.full(len(bin_centres), 2.6)

#construct the differential number counts
N, eN_lo, eN_hi, counts, weights = nc.differential_numcounts(S850_rand, bin_edges, A)
#plot these results
x_bins = bin_centres * 10. **  0.004
ax1.plot(x_bins, N, marker='s', c=ps.grey, linestyle='none', zorder=2, label='S2COSMOS catalogue')
ax1.errorbar(x_bins, N, yerr=(eN_lo, eN_hi), ecolor=ps.grey, fmt='none', zorder=1)

if plot_cumulative:
	c, ec_lo, ec_hi, cumcounts = nc.cumulative_numcounts(counts=counts, A=A)
	x_bins = bin_edges[:-1] * 10. ** 0.004
	ax2.plot(x_bins, c, marker='s', c=ps.grey, linestyle='none', zorder=2, label='S2COSMOS catalogue')
	ax2.errorbar(x_bins, c, yerr=(ec_lo, ec_hi), ecolor=ps.grey, fmt='none', zorder=1)



##################################
#### COMPLETENESS CORRECTIONS ####
##################################

if comp_corr:
	print(gen.colour_string('Calculating completeness corrections...', 'purple'))

	#list of files contaning the flux densities and RMS values corresponding to a given completeness
	comp_files = sorted(glob.glob(PATH_CATS + 'Simpson+19_completeness_curves/*'))
	#corresponding completenesses
	defined_comps = [0.1, 0.3, 0.5, 0.7, 0.9]
	#min, max and step of the axes in the completeness grid: x (flux density) and y (RMS)
	xparams = [-15., 30., 0.01]
	yparams = [0.45, 3.05, 0.01]

	#run the function that reconstructs the completeness grid from Simpson+19
	comp_interp, zgrid = nc.recreate_S19_comp_grid(comp_files, defined_comps, xparams, yparams, plot_grid=plot_comp, other_axes=ax3)

	if plot_data_on_comp:
		#plot the submm sources
		ax3.plot(S850, RMS, marker='.', c='k', linestyle='none', alpha=0.3)
		ax3.errorbar(S850, RMS, xerr=(eS850_lo, eS850_hi), fmt='none', ecolor='k', alpha=0.1)
		#suffix to add to the figure filename
		suffix_comp = '_with_data'
	else:
		suffix_comp = ''

	#save the completeness grid to a FITS file
	compgrid_file = PATH_DATA + 'Completeness_at_S850_and_rms.fits'
	gen.array_to_fits(zgrid.T, compgrid_file, CRPIX=[1,1], CRVAL=[xparams[0],yparams[0]], CDELT=[xparams[2],yparams[2]])

	print(gen.colour_string('Done!', 'purple'))


	print(gen.colour_string('Applying completeness corrections...', 'purple'))

	if randomise_fluxes:
		#set up a multiprocessing Pool using all but one CPU
		pool = Pool(cpu_count()-1)
		#calculate the completeness for the randomly generated flux densities
		comp_s2c = np.array(pool.starmap(comp_interp, [[S850_rand[i], RMS] for i in range(len(S850_rand))]))
		'''
		#construct completeness-corrected bins, removing any randomly generated sources with 0 completeness
		zero_comp = comp_s2c == 0.
		counts_comp_corr = np.array([np.histogram(S850_rand[i][~zero_comp[i]], bin_edges, weights=1./comp_s2c[i][~zero_comp[i]])[0] for i in range(nsim)])
		#convert these to differential number counts
		N_comp_corr = counts_comp_corr * weights
		#take the median values to be the true values and use the 16th and 84th percentiles to estiamte the uncertainties
		N16, N, N84 = np.nanpercentile(N_comp_corr, q=[stats.p16, 50, stats.p84], axis=0)
		eN_lo = N - N16
		eN_hi = N84 - N
		'''
	else:
		comp_s2c = comp_interp(S850, RMS)
		'''
		zero_comp = comp_s2c == 0.
		counts_comp_corr, _  = np.histogram(S850[~zero_comp], bin_edges, weights=1./comp_s2c[~zero_comp])
		N = counts_comp_corr * weights
		eN_lo = eN_hi = np.sqrt(counts_comp_corr) * weights
		'''
	N, eN_lo, eN_hi, counts_comp_corr, weights = nc.differential_numcounts(S850_rand, bin_edges, A, comp=comp_s2c)
		
	#write the results as a table to a FITS file
	t_results = Table([bin_centres, N, eN_lo, eN_hi], names=['S850', 'N_comp_corr', 'eN_lo', 'eN_hi'])
	t_results.write(PATH_CATS + 'S19_differential_number_counts_recreated.fits', overwrite=True)

	#plot these results
	x_bins = bin_centres * 10. ** (-0.004)
	ax1.plot(x_bins, N, marker='D', c=ps.crimson, linestyle='none', zorder=3, label='S2COSMOS catalogue (comp. corr.)')
	ax1.errorbar(x_bins, N, yerr=(eN_lo, eN_hi), ecolor=ps.crimson, fmt='none', zorder=2)


	if plot_cumulative:
		c, ec_lo, ec_hi, cumcounts = nc.cumulative_numcounts(counts_comp_corr, A=A)
		#plot these results
		x_bins = bin_edges[:-1] * 10. ** (-0.004)
		ax2.plot(x_bins, c, marker='D', c=ps.crimson, linestyle='none', zorder=2, label='S2COSMOS catalogue (comp. corr.)')
		ax2.errorbar(x_bins, c, yerr=(ec_lo, ec_hi), ecolor=ps.crimson, fmt='none', zorder=1)

	print(gen.colour_string('Done!', 'purple'))


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

	#best-fit parameters for the differential number counts
	popt_diff, _ = stats.chisq_minimise(bin_centres, N, (eN_hi+eN_lo)/2., nc.schechter_model, S19_fit_params[0])
	#plot the fit
	x_range_plot_fit = x_range_plot*10.**(-0.004)
	ax1.plot(x_range_plot_fit, nc.schechter_model(x_range_plot_fit, popt_diff), c=fit_colour, zorder=11, label=fit_label)
	#add text with the best-fit parameters
	best_fit_str = [
		r'$N_{0} = %.0f$'%popt_diff[0],
		r'$S_{0} = %.1f$'%popt_diff[1],
		r'$\gamma = %.1f$'%popt_diff[2]]
	best_fit_str = '\n'.join(best_fit_str)
	ax1.text(4., 11., best_fit_str, color=fit_colour, ha='left', va='top', fontsize=18.)

	if plot_cumulative:
		#best-fit parameters for the differential number counts
		popt_cumul, _ = stats.chisq_minimise(bin_edges[:-1], c, (ec_hi+ec_lo)/2., nc.schechter_model, S19_fit_params[0])
		#plot the fit
		ax2.plot(x_range_plot_fit, nc.schechter_model(x_range_plot_fit, popt_cumul), c=fit_colour, zorder=11, label=fit_label)
		#add text with the best-fit parameters
		best_fit_str = [
			r'$N_{0} = %.0f$'%popt_cumul[0],
			r'$S_{0} = %.1f$'%popt_cumul[1],
			r'$\gamma = %.1f$'%popt_cumul[2]]
		best_fit_str = '\n'.join(best_fit_str)
		ax2.text(3., 70., best_fit_str, color=fit_colour, ha='left', va='top', fontsize=18.)

	print(gen.colour_string('Done!', 'purple'))

##################################
#### UNCERTAINTIES COMPARISON ####
##################################

if compare_errors:
	print(gen.colour_string('Comparing uncertainties...', 'purple'))
	#calculate ratio of the uncertainties to those from Simpson+19
	r_err_upper = e_upper / eN_hi
	r_err_lower = e_lower / eN_lo

	#plot the 1:1 ratio
	ax4.axhline(1., c=ps.grey, alpha=0.4)
	#plot these as functions of the flux density
	ax4.plot(bin_centres, r_err_upper, marker='D', c=ps.magenta, label=r'$\sigma_{\rm upper}$')
	ax4.plot(bin_centres, r_err_lower, marker='s', c=ps.dark_blue, label=r'$\sigma_{\rm lower}$')
	print(gen.colour_string('Done!', 'purple'))



########################
#### SAVING FIGURES ####
########################

ax1.legend()
#set the minor tick locations on the x-axis
xtick_min_locs = list(np.arange(2,10,1)) + [20]
xtick_min_labels = [2, 5, 20]
ax1.set_xticks(xtick_min_locs, labels=[f'{s:g}' if s in xtick_min_labels else '' for s in xtick_min_locs], minor=True)

#minimise unnecesary whitespace
f1.tight_layout()
#save the figure
f1.savefig(PATH_PLOTS + f'Simpson+19_number_counts_improved_comp.png', dpi=300)

if plot_cumulative:
	ax2.legend()
	ax2.set_xticks(xtick_min_locs, labels=[f'{s:g}' if s in xtick_min_labels else '' for s in xtick_min_locs], minor=True)

	#minimise unnecesary whitespace
	f2.tight_layout()
	#save the figure
	f2.savefig(PATH_PLOTS + f'Simpson+19_cumulative_counts_improved_comp.png', dpi=300)

if plot_comp:
	#set the axes limits (completeness grid)
	xmin, xmax = ax3.set_xlim(0., 13.)
	ymin, ymax = ax3.set_ylim(yparams[0], yparams[1])
	f3.tight_layout()
	f3.savefig(PATH_PLOTS + f'Simpson+19_completeness_extrapolated{suffix_comp}.png', dpi=300)

if compare_errors:
	ax4.legend()
	ax4.set_xticks(xtick_min_locs, labels=[f'{s:g}' if s in xtick_min_labels else '' for s in xtick_min_locs], minor=True)
	#minimise unnecesary whitespace
	f4.tight_layout()
	#save the figure
	f4.savefig(PATH_PLOTS + 'Simpson+19_number_counts_uncertainty_comparison_improved_comp.png', dpi=300)






