############################################################################################################
# A script for trying to recreate the number counts plot from Simpson+19.
############################################################################################################

#import modules/packages
import os
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


##################
#### SETTINGS ####
##################

#toggle `switches' for additional functionality
use_cat_from_paper = True	#use the unedited catalogue downloaded from the Simpson+19 paper website
plot_cumulative = True		#make cumulative number counts as well as differential number counts
randomise_fluxes = True		#randomly draw flux densities from possible values
comp_corr = True			#apply the completeness corrections to the number counts
randomise_comp = True		#calculate completeness corrections for each randomly drawn flux density
plot_comp = True			#plot the completeness as a function of flux density and RMS
plot_data_on_comp = True	#plot the positions of each source in RMS-S850 space on the completeness figure
fit_schechter = True		#fits Schechter functions to the results
exclude_faint = True		#exclude faintest bins from Schechter fitting
compare_errors = True		#compare the uncertainties from S19 with the reconstruction performed here
main_only = gen.main_only	#use only sources from the MAIN region of S2COSMOS
settings = [
	use_cat_from_paper,
	plot_cumulative,
	randomise_fluxes,
	comp_corr,
	randomise_comp,
	plot_comp,
	plot_data_on_comp,
	fit_schechter,
	exclude_faint,
	compare_errors,
	main_only
]

#print the chosen settings to the Terminal
print(gen.colour_string('CHOSEN SETTINGS:', 'white'))
settings_print = [
	'Use catalogue linked to Simpson+19 paper: ',
	'Construct cumulative number counts: ',
	'Randomise source flux densities for error analysis: ',
	'Apply completeness corrections to number counts: ',
	'Calculate completeness corrections for each randomly drawn flux density: ',
	'Plot the completeness as a function S850 and RMS: ',
	'Show sources on completeness plot: ',
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

if fit_schechter:
	print(gen.colour_string('Fitting to Simpson+19 differential number counts...', 'purple'))
	'''
	#attempt a fit to the S19 results and print/plot the best fit
	popt, _ = stats.chisq_minimise(bin_centres, bin_heights, nc.schechter_model, S19_fit_params[0], yerr=(e_upper+e_lower)/2.)
	ax1.plot(x_range_plot, nc.schechter_model(x_range_plot, popt), c='k', linestyle='--', zorder=11, label='Simpson+19; my fit')
	#estimate the uncertainties on each fit parameter
	elo_popt, ehi_popt, *_ = stats.uncertainties_in_fit(bin_centres, bin_heights, (e_upper, e_lower), nc.schechter_model, S19_fit_params[0], use_yerr_in_fit=False)
	#add text with the best-fit parameters
	for i in range(len(best_fit_str)):
		if i == 0:
			best_fit_str[i] += r' $(%.0f^{+%.0f}_{-%.0f})$'%(popt[i],ehi_popt[i],elo_popt[i])
		else:
			best_fit_str[i] += r' $(%.1f^{+%.1f}_{-%.1f})$'%(popt[i],ehi_popt[i],elo_popt[i])
	'''
	popt, elo_popt, ehi_popt = nc.fit_schechter_mcmc(bin_centres, bin_heights, (e_upper+e_lower)/2., 100, 10000, S19_fit_params[0], pool=None) 
	ax1.plot(x_range_plot, nc.schechter_model(x_range_plot, popt), c='k', linestyle='--', zorder=11, label='Simpson+19; my fit')
	#add text with the best-fit parameters
	for i in range(len(best_fit_str)):
		if i == 0:
			best_fit_str[i] += r' $(%.0f^{+%.0f}_{-%.0f})$'%(popt[i],ehi_popt[i],elo_popt[i])
		else:
			best_fit_str[i] += r' $(%.1f^{+%.1f}_{-%.1f})$'%(popt[i],ehi_popt[i],elo_popt[i])
	print(gen.colour_string('Done!', 'purple'))

best_fit_str = '\n'.join(best_fit_str)
ax1.text(2., 30., best_fit_str, color='k', ha='left', va='top', fontsize=18.)

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
		print(gen.colour_string('Fitting to Simpson+19 cumulative number counts...', 'purple'))
		'''
		#attempt a fit to the S19 results and print/plot the best fit
		popt, _ = stats.chisq_minimise(bin_edges[:-1], bin_heights_c, nc.schechter_model, S19_fit_params[0], yerr=(e_upper_c + e_lower_c) / 2.)
		ax2.plot(x_range_plot, nc.schechter_model(x_range_plot, popt), c='k', zorder=11, label='Simpson+19 best fit')
		#estimate the uncertainties on each fit parameter
		elo_popt, ehi_popt, *_ = stats.uncertainties_in_fit(bin_edges[:-1], bin_heights_c, (e_upper_c, e_lower_c), nc.schechter_model, S19_fit_params[0], use_yerr_in_fit=False)
		#add text with the best-fit parameters
		best_fit_str = [
			r'$N_{0} = %.0f^{+%.0f}_{-%.0f}$'%(popt[0],ehi_popt[0],elo_popt[0]),
			r'$S_{0} = %.1f^{+%.1f}_{-%.1f}$'%(popt[1],ehi_popt[1],elo_popt[1]),
			r'$\gamma = %.1f^{+%.1f}_{-%.1f}$'%(popt[2],ehi_popt[2],elo_popt[2])
			]
		best_fit_str = '\n'.join(best_fit_str)
		ax2.text(10., 1000., best_fit_str, color='k', ha='left', va='top', fontsize=18.)
		'''
		popt, elo_popt, ehi_popt = nc.fit_schechter_mcmc(bin_edges[:-1], bin_heights_c, (e_upper_c+e_lower_c)/2., 100, 10000, S19_fit_params[0], pool=None) 
		ax2.plot(x_range_plot, nc.schechter_model(x_range_plot, popt), c='k', zorder=11, label='Simpson+19 best fit')
		#add text with the best-fit parameters
		best_fit_str = [
			r'$N_{0} = %.0f^{+%.0f}_{-%.0f}$'%(popt[0],ehi_popt[0],elo_popt[0]),
			r'$S_{0} = %.1f^{+%.1f}_{-%.1f}$'%(popt[1],ehi_popt[1],elo_popt[1]),
			r'$\gamma = %.1f^{+%.1f}_{-%.1f}$'%(popt[2],ehi_popt[2],elo_popt[2])
			]
		best_fit_str = '\n'.join(best_fit_str)
		ax2.text(10., 1000., best_fit_str, color='k', ha='left', va='top', fontsize=18.)
		print(gen.colour_string('Done!', 'purple'))

#####################################
#### ATTEMPT TO RECREATE RESULTS ####
#####################################

if use_cat_from_paper:
	#catalogue containing submm data for S2COSMOS sources
	SUBMM_CAT = PATH_CATS + 'Simpson+19_S2COSMOS_source_cat.fits'
	data_submm = Table.read(SUBMM_CAT, format='fits')
	if main_only:
		data_submm = data_submm[data_submm['Sample'] == 'MAIN']
	#get the (deboosted) 850 µm flux densities and the uncertainties
	S850 = data_submm['S850-deb']
	eS850_lo = data_submm['e_S850-deb'] #/ (2. * np.sqrt(2. * np.log(2.)))
	eS850_hi = data_submm['E_S850-deb'] #/ (2. * np.sqrt(2. * np.log(2.)))
	#also get the RMS
	RMS = data_submm['e_S850-obs']
else:
	#catalogue containing submm data for S2COSMOS sources
	SUBMM_CAT = PATH_CATS + 'S2COSMOS_sourcecat850_Simpson18.fits'
	data_submm = Table.read(SUBMM_CAT, format='fits')
	if main_only:
		data_submm = data_submm[data_submm['CATTYPE'] == 'MAIN']
	#get the (deboosted) 850 µm flux densities and the uncertainties
	S850 = data_submm['S_deboost']
	eS850_lo = data_submm['S_deboost_errlo'] #/ (2. * np.sqrt(2. * np.log(2.)))
	eS850_hi = data_submm['S_deboost_errhi'] #/ (2. * np.sqrt(2. * np.log(2.)))
	#also get the RMS
	RMS = data_submm['RMS']

if randomise_fluxes:
	#see if file exists containing randomised flux densities already
	npz_filename = PATH_CATS + 'S2COSMOS_randomised_S850.npz'
	if os.path.exists(npz_filename):
		rand_data = np.load(npz_filename)
		S850_rand = rand_data['S850_rand']
	else:
		S850_rand = np.array([stats.random_asymmetric_gaussian(S850[i], eS850_lo[i], eS850_hi[i], nsim) for i in range(len(S850))]).T
		#set up a dictionary containing the randomised data
		dict_rand = {'S850_rand':S850_rand}
else:
	S850_rand = S850[:]

#construct the differential number counts
N, eN_lo, eN_hi, counts, weights = nc.differential_numcounts(S850_rand, bin_edges, A, incl_poisson=True)
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

	#see if a file exists with the reconstructed S2COSMOS completeness grid (should have been created
	#in previous step of pipeline)
	compgrid_file = PATH_DATA + 'Completeness_at_S850_and_rms.fits'
	if os.path.exists(compgrid_file):
		CG_HDU = fits.open(compgrid_file)[0]
		zgrid = CG_HDU.data.T
		hdr = CG_HDU.header
		#get the min, max and step along the S850 and RMS axes
		xmin = hdr['CRVAL1']
		xstep = hdr['CDELT1']
		xmax = xmin + (hdr['NAXIS1'] - 1) * xstep
		xparams = [xmin, xmax, xstep]
		ymin = hdr['CRVAL2']
		ystep = hdr['CDELT2']
		ymax = ymin + (hdr['NAXIS2'] - 1) * ystep
		yparams = [ymin, ymax, ystep]
		#create a grid in flux density-RMS space
		xgrid, ygrid = np.mgrid[xmin:xmax+xstep:xstep, ymin:ymax+ystep:ystep]
		#interpolate the completeness values from the file w.r.t. S850 and RMS
		points = np.array([xgrid.flatten(), ygrid.flatten()])
		values = zgrid.flatten()
		comp_interp = LinearNDInterpolator(points.T, values)
	#otherwise, reconstruct the completeness from scratch
	else:
		#list of files contaning the flux densities and RMS values corresponding to a given completeness
		comp_files = sorted(glob.glob(PATH_CATS + 'Simpson+19_completeness_curves/*'))
		#corresponding completenesses
		defined_comps = [0.1, 0.3, 0.5, 0.7, 0.9]
		#min, max and step of the axes in the completeness grid: x (flux density) and y (RMS)
		xparams = [-15., 30., 0.01]
		yparams = [0.45, 3.05, 0.01]
		#run the function that reconstructs the completeness grid from Simpson+19
		comp_interp, zgrid = nc.recreate_S19_comp_grid(comp_files, defined_comps, xparams, yparams, plot_grid=plot_comp, other_axes=ax3)
		#save the completeness grid to a FITS file
		gen.array_to_fits(zgrid.T, compgrid_file, CRPIX=[1,1], CRVAL=[xparams[0],yparams[0]], CDELT=[xparams[2],yparams[2]])

	if plot_data_on_comp:
		#plot the submm sources
		ax3.plot(S850, RMS, marker='.', c='k', linestyle='none', alpha=0.3)
		ax3.errorbar(S850, RMS, xerr=(eS850_lo, eS850_hi), fmt='none', ecolor='k', alpha=0.1)
		#suffix to add to the figure filename
		suffix_comp = '_with_data'
	else:
		suffix_comp = ''

	print(gen.colour_string('Done!', 'purple'))


	print(gen.colour_string('Applying completeness corrections...', 'purple'))

	if randomise_fluxes and randomise_comp:
		#see if the completeness has already been calculated for the randomised flux densities
		if 'rand_data' in globals():
			comp_s2c = rand_data['comp_rand']
		else:
			#set up a multiprocessing Pool using all but one CPU
			pool = Pool(cpu_count()-1)
			#calculate the completeness for the randomly generated flux densities
			comp_s2c = np.array(pool.starmap(comp_interp, [[S850_rand[i], RMS] for i in range(len(S850_rand))]))
			#add these completenesses to the dictionary of randomly generated data
			dict_rand['comp_rand'] = comp_s2c
	else:
		comp_s2c = comp_interp(S850, RMS)

	N, eN_lo, eN_hi, counts_comp_corr, weights = nc.differential_numcounts(S850_rand, bin_edges, A, comp=comp_s2c, incl_poisson=True)
		
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

	#if told to exclude faint bins, set the lowest index to 2
	if exclude_faint:
		imin = 3
		#suffix to use for figures in which the faintest bins have been excluded
		suffix_ef = '_excl_faint3'
	else:
		imin = 0
	'''
	#best-fit parameters for the differential number counts
	popt_diff, _ = stats.chisq_minimise(bin_centres[imin:], N[imin:], nc.schechter_model, S19_fit_params[0], yerr=(eN_hi+eN_lo)[imin:]/2.)
	#plot the fit
	x_range_plot_fit = x_range_plot*10.**(-0.004)
	ax1.plot(x_range_plot_fit, nc.schechter_model(x_range_plot_fit, popt_diff), c=fit_colour, zorder=11, label=fit_label)
	#estimate the uncertainties on each fit parameter
	elo_popt_diff, ehi_popt_diff, *_ = stats.uncertainties_in_fit(bin_centres[imin:], N[imin:], (eN_hi[imin:], eN_lo[imin:]), nc.schechter_model, S19_fit_params[0], use_yerr_in_fit=False)
	'''

	popt_diff, elo_popt_diff, ehi_popt_diff = nc.fit_schechter_mcmc(bin_centres[imin:], N[imin:], (eN_hi+eN_lo)[imin:]/2., 100, 10000, S19_fit_params[0], pool=None) 
	#plot the fit
	x_range_plot_fit = x_range_plot*10.**(-0.004)
	ax1.plot(x_range_plot_fit, nc.schechter_model(x_range_plot_fit, popt_diff), c=fit_colour, zorder=11, label=fit_label)
	#add text with the best-fit parameters
	best_fit_str = [
		r'$N_{0} = %.0f^{+%.0f}_{-%.0f}$'%(popt_diff[0],ehi_popt_diff[0],elo_popt_diff[0]),
		r'$S_{0} = %.1f^{+%.1f}_{-%.1f}$'%(popt_diff[1],ehi_popt_diff[1],elo_popt_diff[1]),
		r'$\gamma = %.1f^{+%.1f}_{-%.1f}$'%(popt_diff[2],ehi_popt_diff[2],elo_popt_diff[2])
		]
	best_fit_str = '\n'.join(best_fit_str)
	ax1.text(9., 1000., best_fit_str, color=fit_colour, ha='left', va='top', fontsize=18.)

	if plot_cumulative:
		'''
		#best-fit parameters for the differential number counts
		popt_cumul, _ = stats.chisq_minimise(bin_edges[imin:-1], c[imin:], nc.schechter_model, S19_fit_params[0], yerr=(ec_hi+ec_lo)[imin:]/2.)
		#plot the fit
		ax2.plot(x_range_plot_fit, nc.schechter_model(x_range_plot_fit, popt_cumul), c=fit_colour, zorder=11, label=fit_label)
		#estimate the uncertainties on each fit parameter
		elo_popt_cumul, ehi_popt_cumul, *_ = stats.uncertainties_in_fit(bin_edges[imin:-1], c[imin:], (ec_hi[imin:], ec_lo[imin:]), nc.schechter_model, S19_fit_params[0], use_yerr_in_fit=False)
		'''
		popt_cumul, elo_popt_cumul, ehi_popt_cumul = nc.fit_schechter_mcmc(bin_edges[imin:-1], c[imin:], (ec_hi+ec_lo)[imin:]/2., 100, 10000, S19_fit_params[0], pool=None) 
		#plot the fit
		ax2.plot(x_range_plot_fit, nc.schechter_model(x_range_plot_fit, popt_cumul), c=fit_colour, zorder=11, label=fit_label)
		#add text with the best-fit parameters
		best_fit_str = [
			r'$N_{0} = %.0f^{+%.0f}_{-%.0f}$'%(popt_cumul[0],ehi_popt_cumul[0],elo_popt_cumul[0]),
			r'$S_{0} = %.1f^{+%.1f}_{-%.1f}$'%(popt_cumul[1],ehi_popt_cumul[1],elo_popt_cumul[1]),
			r'$\gamma = %.1f^{+%.1f}_{-%.1f}$'%(popt_cumul[2],ehi_popt_cumul[2],elo_popt_cumul[2])
			]
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



##################################
#### SAVING FIGURES AND FILES ####
##################################

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


if randomise_fluxes and ('dict_rand' in globals()):
	#save the randomised flux densities (and completenesses if generated) to a compressed numpy archive
	np.savez_compressed(npz_filename, **dict_rand)



