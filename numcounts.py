############################################################################################################
# Module containing functions relating to the construction of number counts.
###########################################################################################################

import general as gen
import stats
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
import scipy.optimize as opt
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from astropy.table import Table
from matplotlib.ticker import AutoMinorLocator

def FD_like(x, theta):
	'''
	Fermi-Dirac-like distribution to model a completeness curve.
	
	Parameters
	----------
	x: array-like
		Values (of flux density) at which the distribution is analysed.
	
	theta: array-like
		Contains the fit parameters defining the slope (A) and the x-position (B) at which the 
		completeness = 0.5, respectively.

	Returns
	----------
	FD: array-like
		Function values evaluated at the provided x-values.
	'''
	A, B = theta
	FD = 1. / (np.exp(A * (-x + B)) + 1.)
	return FD


def recreate_S19_comp_grid(
	comp_files,
	defined_comps=[0.1, 0.3, 0.5, 0.7, 0.9],
	xparams=[-15., 30., 0.01],
	yparams=[0.45, 3.05, 0.01],
	plot_grid=False,
	**plot_kwargs
):
	'''
	Recreates the S2COSMOS completeness as a function of deboosted flux density and RMS from Simpson+19.

	Parameters
	----------
	comp_files: list
		List of files containing the flux densities and RMS values corresponding to a given completeness.
		Each file corresponds to a one value of completeness and must be provided in the same order as
		defined_comps.

	defined_comps: list
		List of completeness values for which the functions are well-defined in Simpson+19. Must have
		same length as comp_files, and must be provided in the same order.

	xparams: list
		Specifies the parameters of the grid in the x (flux density) direction: [min,max,step].

	yparams: list
		Specifies the parameters of the grid in the y (RMS) direction: [min,max,step].

	plot_grid: bool
		Whether to create a plot of the completeness grid.

	plot_filename: str
		Filename to be given to the plot if saved.

	**plot_kwargs
		Any remaining keyword arguments will be used to format the plot (if generated).

	Returns
	----------
	comp_interp: callable
		Completeness as a 2D interpolated function of deboosted flux density and RMS.

	zgrid: 2d array
		Grid of completeness values at given (flux density, RMS) coordinates.
	'''

	######################
	#### FIGURE SETUP ####
	######################

	#if told to plot the completeness grid, set up the figure
	if plot_grid:
		#create the figure (completeness plot)
		fig, ax = plt.subplots(1, 1)
		#label the axes
		ax.set_xlabel(r'$S_{850~\mu{\rm m}}$ (mJy)')
		ax.set_ylabel(r'$\sigma_{\rm inst}$ (mJy)')

		#colourmap for the completeness plot
		if 'cmap' in plot_kwargs:
			cmap = plot_kwargs['cmap']
		else:
			interval = np.arange(0.1, 0.501, 0.001)[::-1]
			colors = plt.cm.RdYlBu(interval)
			cmap = LinearSegmentedColormap.from_list('name', colors)
		if 'other_axes' in plot_kwargs:
			ax_other = plot_kwargs['other_axes']
		else:
			ax_other = None

	#######################
	#### INTERPOLATION ####
	#######################

	#create a grid in flux density-RMS space
	xmin, xmax, xstep = xparams
	ymin, ymax, ystep = yparams
	xgrid, ygrid = np.mgrid[xmin:xmax+xstep:xstep, ymin:ymax+ystep:ystep]
	#flat arrays of intervals in flux density and RMS
	xspace = np.arange(xmin, xmax+xstep, xstep)
	yspace = np.arange(ymin, ymax+ystep, ystep)

	#list to which the flux density, RMS and completeness values will be appended
	values_list = []
	#list to which the interpolated RMS as functions of flux density will be appended
	x_interp_list = []
	y_interp_list = []

	#lists to which interpolated values of x and y will be appended for each completeness curve
	x_interp_vals = []
	y_interp_vals = []
	for f,comp in zip(comp_files,defined_comps):
		t_now = Table.read(f, format='ascii')
		#interpolate the RMS as a function of flux density
		y_interp = interp1d(t_now['col1'], t_now['col2'], fill_value='extrapolate')
		#reverse the interpolation to get flux density as a function of RMS as well
		x_interp = interp1d(t_now['col2'], t_now['col1'], fill_value='extrapolate')
		#get the completeness from the filename
		#comp = float(f[-8:-6]) / 100.
		#use the interpolated functions to calculate the flux density at regular intervals of rms and vice versa
		Y = y_interp(xspace)
		X = x_interp(yspace)
		#append the relevant values/functions to the lists
		values_list.append([xspace, Y, np.full(len(xspace), comp)])
		x_interp_list.append(x_interp)
		y_interp_list.append(y_interp)
		x_interp_vals.append(X)
		y_interp_vals.append(Y)

		if plot_grid:
			#plot the curve
			ax.plot(xspace, Y, color='k', zorder=10)
			ax.plot(X, yspace, color='k', linestyle=':', zorder=10)
			if ax_other is not None:
				ax_other.plot(xspace, Y, color='k', zorder=10)
				ax_other.plot(X, yspace, color='k', linestyle=':', zorder=10)

	#stack the results for each completeness curve
	t_all = np.hstack(values_list)
	#split the array into x-y coordinate pairs and the corresponding z values
	points = t_all[:2]
	values = t_all[-1]
	#interpolate completeness as a function of flux density and RMS
	comp_interp = LinearNDInterpolator(points.T, values)
	#create a grid of completeness values
	zgrid = comp_interp(xgrid, ygrid)


	#######################
	#### EXTRAPOLATION ####
	#######################
	
	#uncertainties to give the observed completeness values, arbitrarily chosen such that the 10% and 90% 
	#values have the highest weighting
	comp_err = np.array([0.001, 0.01, 0.01, 0.01, 0.001])

	#completeness values at which new curves are to be estimated
	comp_to_find = np.concatenate(([0.0001], np.arange(0.01, 0.1, 0.01), np.arange(0.91, 1., 0.01), [0.99999]))

	S_comp_to_find = []
	#cycle through values of RMS at which to collapse the 2D completeness function to 1D functions of flux density
	for i in range(len(yspace)):
		rms_now = yspace[i]
		#flux densities at which the completeness curves are well-defined by Simpson et al.
		S_list = np.array([x_interp_list[j](rms_now) for j in range(len(x_interp_list))])
		#fit a curve to the data, weighting the 10% and 90% points more heavily
		initial = [1., S_list[2]]
		popt, _ = stats.chisq_minimise(S_list, defined_comps, FD_like, initial, yerr=comp_err)
		#create x-ranges at which to plot things
		xrange_lo = xspace[xspace < S_list[0]]
		xrange_mid = xspace[(xspace >= S_list[0]) * (xspace <= S_list[-1])]
		xrange_hi = xspace[xspace > S_list[-1]]
		#evaluate the fitted curve over the low and high x-ranges, and the original completeness curve in the mid range
		x_range = np.concatenate((xrange_lo, xrange_mid, xrange_hi))
		#z_range = np.concatenate((FD_model(popt, xrange_lo), comp_interp(xrange_mid, rms_now), FD_model(popt, xrange_hi)))
		z_range = np.concatenate((FD_like(xrange_lo, popt), comp_interp(xrange_mid, rms_now), FD_like(xrange_hi, popt)))
		#interpolate x w.r.t. z
		S_interp = interp1d(z_range, x_range, fill_value='extrapolate')
		#get the flux densities at the desired completenesses
		S_at_comp = S_interp(comp_to_find)
		S_comp_to_find.append(S_at_comp)

	S_comp_to_find = np.array(S_comp_to_find).T

	#interpolate the flux density w.r.t. rms at each value of completeness
	for i in range(len(comp_to_find)):
		ctf = comp_to_find[i]
		y_interp_new = interp1d(S_comp_to_find[i], yspace, fill_value=(ymin-1000*ystep, ymax+1000*ystep), bounds_error=False)
		#y_interp_new = interp1d(S_comp_to_find[i], yspace, fill_value='extrapolate', bounds_error=False)
		Y_new = y_interp_new(xspace)
		if ctf < comp_to_find[0]:
			y_interp_list.insert(i, y_interp_new)
			values_list.insert(i, [xspace, Y_new, np.full(len(xspace), ctf)])
		else:
			y_interp_list.append(y_interp_new)
			values_list.append([xspace, Y_new, np.full(len(xspace), ctf)])
		
		if plot_grid:
			ax.plot(xspace, Y_new, c='k', linestyle='--')
			if ax_other is not None:
				ax_other.plot(xspace, Y_new, c='k', linestyle='--')

	#stack the results for each completeness curve
	t_all = np.hstack(values_list)
	#split the array into x-y coordinate pairs and the corresponding z values
	points = t_all[:2]
	values = t_all[-1]
	#interpolate completeness as a function of flux density and RMS
	comp_interp = LinearNDInterpolator(points.T, values)
	#create a grid of completeness values
	zgrid = comp_interp(xgrid, ygrid)

	#get the interpolated ~100% completeness curve
	interp_100 = y_interp_list[-1]
	#identify all elements in the grid that lie below the 100% completeness curve
	lower_mask = ygrid < interp_100(xgrid)
	#fill these values with 100%
	zgrid[lower_mask] = 1.
	#identify all elements in the grid that lie below a flux density of 0 mJy
	zero_mask = xgrid <= 0.
	#fill these values with 0
	zgrid[zero_mask] = 0.

	#a few NaNs exist at the upper right corner of the grid; replace them with 1s
	zgrid[np.isnan(zgrid)] = 1.
	
	if plot_grid:
		if 'levels' in plot_kwargs:
			levels = plot_kwargs['levels']
		else:
			levels = 100
		#plot the filled contours
		C = ax.contourf(xgrid, ygrid, zgrid*100., levels=levels, cmap=cmap, zorder=1)
		#add colourbar
		cbar = plt.colorbar(C, ax=ax)
		cbar.ax.set_ylabel('Completeness (\%)')
		cbar.set_ticks(np.arange(0., 110., 10.))
		cbar_minor_ticks = AutoMinorLocator(2)
		cbar.ax.yaxis.set_minor_locator(cbar_minor_ticks)

		if ax_other is not None:
			#plot the filled contours
			C = ax_other.contourf(xgrid, ygrid, zgrid*100., levels=levels, cmap=cmap, zorder=1)
			#add colourbar
			cbar = plt.colorbar(C, ax=ax_other)
			cbar.ax.set_ylabel('Completeness (\%)')
			cbar.set_ticks(np.arange(0., 110., 10.))
			cbar_minor_ticks = AutoMinorLocator(2)
			cbar.ax.yaxis.set_minor_locator(cbar_minor_ticks)


	#re-interpolate the completeness now that the NaNs have been replaced
	points = np.array([xgrid.flatten(), ygrid.flatten()])
	values = zgrid.flatten()
	comp_interp = LinearNDInterpolator(points.T, values)


	#######################
	#### SAVING FIGURE ####
	#######################

	if plot_grid:
		#axis limits
		if 'xlims' in plot_kwargs:
			ax.set_xlim(*plot_kwargs['xlims'])
		if 'ylims' in plot_kwargs:
			ax.set_ylim(*plot_kwargs['ylims'])

		if 'plot_filename' in plot_kwargs:
			#minimise unnecesary whitespace
			fig.tight_layout()
			plotname = plot_kwargs['plot_filename']
			#save the figure
			fig.savefig(plotname, dpi=300)
		else:
			if ax_other is None:
				gen.error_message(
					'numcounts.recreate_S19_comp_grid',
					'No axes or figure filename provided. Figure not saved.'
					)


	return (comp_interp, zgrid)



def schechter_model(S, params):
	'''
	Model Schechter function of the form used for number counts.

	Parameters
	----------
	S: array-like 
		Flux density values at which the function is to be evaluated.

	params: array_like
		List/tuple/array containing the fit parameters (normalisation, flux density at the 'knee',
		faint-end slope).

	Returns
	----------
	y: array-like
		Model Schechter function values evaluated at the specified flux densities.
	'''

	N0, S0, gamma = params
	with np.errstate(all='ignore'):
		y = (N0 / S0) * (S0 / S) ** gamma * np.exp(-S / S0)
	return y



def differential_numcounts(S, bin_edges, A, comp=None, poisson=False):
	'''
	Constructs differential number counts for a given set of flux densities. Optionally also
	randomises the flux densities according to their uncertainties to get an estimate of the
	uncertainties on each bin (otherwise just assumes Poissonian uncertainties).

	Parameters
	----------
	S: array-like 
		Flux density values to be binned.

	bin_edges: array_like
		Edges of the bins to be used for constructing the number counts (includes rightmost edge).

	A: float or array-like
		Area probed for all flux density bins, or for each individual bin.

	comp: array-like or None
		Completeness of each source. If None, will simply create an array of 1s with the same shape
		as counts or S (whichever is provided).

	poisson: bool
		Whether to include Poissonian uncertainties in the errorbars. 

	Returns
	----------
	N: array-like
		Differential number counts.

	eN_lo, eN_hi: array-like
		Lower and upper uncertainties on each bin height.

	counts: array-like
		Raw counts in each flux density bin.

	weights: array-like
		Factors by which the raw counts in each bin are multiplied to get the differential number 
		counts.
	'''

	#calculate the bin widths
	dS = bin_edges[1:] - bin_edges[:-1]
	#weights for each bin (i.e. divide the counts by the area times the width of the bin)
	weights = 1. / (A * dS)

	#if completeness values not provided, create an array of 1s with the same shape as the flux densities
	if comp is None:
		try:
			comp = np.ones(S.shape)
		except AttributeError:
			comp = np.ones(np.array(S).shape)
	#want to exclude any sources with 0 completeness
	nonzero_comp = comp > 0.

	#get the number of dimensions for the imported flux density array
	ndim = gen.get_ndim(S)
	#if 2-dimensional, assume each row is one dataset
	if ndim == 2:
		#get the dimensions of the provided completeness array
		ndim_comp = gen.get_ndim(comp)
		if ndim_comp == 2:
			#bin the sources in each dataset
			counts = np.array([np.histogram(S[i][nonzero_comp[i]], bin_edges, weights=1./comp[i][nonzero_comp[i]])[0] for i in range(len(S))])
		else:
			counts = np.array([np.histogram(S[i][nonzero_comp], bin_edges, weights=1./comp[nonzero_comp])[0] for i in range(len(S))])
		'''
		if poisson:
			#take the median counts in each bin and calculate the Poissonian uncertainties
			counts_med = np.median(counts, axis=0)
			ecounts_lo_p, ecounts_hi_p = np.array(stats.poisson_errs_1sig(counts_med))
			#randomly draw values from a distribution defined by these uncertainties
			counts_p = np.array([stats.random_asymmetric_gaussian(counts_med[i], ecounts_lo_p[i], ecounts_hi_p[i], len(S)) for i in range(len(counts_med))]).T
			#concatenate with the array of existing counts
			counts = np.concatenate([counts, counts_p], axis=0)
			np.random.shuffle(counts)
		'''
		'''
		if poisson:
			#take the median counts in each bin and randomly draw from Poisson distributions
			counts_med = np.median(counts, axis=0)
			counts_p = np.random.poisson(counts_med, size=(len(counts), len(counts_med)))
			counts = np.concatenate([counts, counts_p], axis=0)
			np.random.shuffle(counts)
		'''
		'''
		if poisson:
			ecounts_lo_p, ecounts_hi_p = np.array(stats.poisson_errs_1sig(counts))
			ecounts_lo_p[np.isnan(ecounts_lo_p)] = 0.
			counts = np.array([[stats.random_asymmetric_gaussian(c[j], ec_lo[j], ec_hi[j], 1)[0] for j in range(len(c))] for c,ec_lo,ec_hi in zip(counts,ecounts_lo_p,ecounts_hi_p)])
		'''
		'''
		if poisson:
			#permute each count 1000 times according to its Poissonian uncertainties
			counts_p = np.random.poisson(counts, size=(1000, *counts.shape))
			N_rand = counts * weights
			N16, N, N84 = np.percentile(N_rand, q=[stats.p16, 50, stats.p84], axis=[0,1])
		else:
			N_rand = counts * weights
			N16, N, N84 = np.percentile(N_rand, q=[stats.p16, 50, stats.p84], axis=0)
		'''

		N_rand = counts * weights
		#take the median values to be the true values and use the 16th and 84th percentiles to estiamte the uncertainties
		N16, N, N84 = np.percentile(N_rand, q=[stats.p16, 50, stats.p84], axis=0)
		eN_lo = N - N16
		eN_hi = N84 - N
		
		
		if poisson:
			counts_med = np.median(counts, axis=0)
			eN_lo_p, eN_hi_p = np.array(stats.poisson_errs_1sig(counts_med)) * weights
			eN_lo = np.sqrt(eN_lo ** 2. + eN_lo_p ** 2.)
			eN_hi = np.sqrt(eN_hi ** 2. + eN_hi_p ** 2.)
		

	#if 1-dimensonal, only one dataset
	elif ndim == 1:
		#bin the sources in the catalogue by their flux densities using the bins defined above
		counts, _ = np.histogram(S, bins=bin_edges, weights=1./comp)

		#convert to number density
		N = counts * weights
		#Poissonian uncertainties
		eN_lo, eN_hi = np.array(stats.poisson_errs_1sig(counts)) * weights


	return N, eN_lo, eN_hi, counts, weights




def cumulative_numcounts(counts=None, S=None, bin_edges=None, A=1., comp=None, poisson=True):
	'''
	Constructs cumulative number counts, either by taking the cumulative sum of the provided counts or,
	if counts aren't provided, by calculating from scratch for a given set of flux densities.
	Optionally also randomises the flux densities according to their uncertainties to get an estimate 
	of the uncertainties on each bin (otherwise just assumes Poissonian uncertainties).

	Parameters
	----------
	counts: array-like or None
		If provided, will simply perform the cumulative sum of these counts. If None, will calculate
		the cumulative number counts from scratch.

	S: array-like or None
		Flux density values to be binned if counts is None.

	bin_edges: array_like or None 
		Edges of the bins to be used for constructing the number counts (includes rightmost edge)
		if counts is None.

	A: float or array-like
		Area probed for all flux density bins, or for each individual bin, if counts is None.

	comp: array-like or None
		Completeness of each source. If None, will simply create an array of 1s with the same shape
		as counts or S (whichever is provided).

	Returns
	----------
	N: array-like
		Differential number counts.

	eN_lo, eN_hi: array-like
		Lower and upper uncertainties on each bin height.

	cumcounts: array-like
		Raw cumulative counts in each flux density bin.
	'''

	if (counts is None):
		#calculate the bin widths
		dS = bin_edges[1:] - bin_edges[:-1]

		if None not in [S, bin_edges]:
			#if completeness values not provided, create an array of 1s with the same shape as the flux densities
			if comp is None:
				try:
					comp = np.ones(S.shape)
				except AttributeError:
					comp = np.ones(np.array(S).shape)
			#want to exclude any sources with 0 completeness
			nonzero_comp = comp > 0.

			#get the number of dimensions for the imported flux density array
			ndim = gen.get_ndim(S)
			#if 2-dimensional, assume each row is one dataset
			if ndim == 2:
				#bin the sources in each dataset
				counts = np.array([np.histogram(S[i][nonzero_comp[i]], bin_edges, weights=1./comp[i][nonzero_comp[i]])[0] for i in range(len(S))])
			#if 1-dimensonal, only one dataset
			elif ndim == 1:
				#bin the sources in the catalogue by their flux densities using the bins defined above
				counts, _ = np.histogram(S, bins=bin_edges)
		else:
			gen.error_message(
				'numcounts.cumulative_numcounts',
				'Must provide either counts (array-like), or both S (array-like) and bin_edges (array-like).'
				)
			return None

	#determine whether the array of counts is 1-dimensional or 2-dimensional
	ndim = gen.get_ndim(counts)
	if ndim == 2:
		#calculate the cumulative counts for each randomly generated dataset
		cumcounts = np.cumsum(counts[:,::-1]/A, axis=1)[:,::-1]
		#take the median values to be the true values and use the 16th and 84th percentiles to estiamte the uncertainties
		N16, N, N84 = np.percentile(cumcounts, q=[stats.p16, 50, stats.p84], axis=0)
		eN_lo = N - N16
		eN_hi = N84 - N

		if poisson:
			cumcounts_med = np.median(cumcounts, axis=0)
			eN_lo_p, eN_hi_p = np.array(stats.poisson_errs_1sig(cumcounts_med))
			eN_lo = np.sqrt(eN_lo ** 2. + eN_lo_p ** 2.)
			eN_hi = np.sqrt(eN_hi ** 2. + eN_hi_p ** 2.)
	else:
		#calculate the cumulative counts
		N = cumcounts = np.cumsum(counts[::-1]/A)[::-1]
		eN_lo, eN_hi = np.array(stats.poisson_errs_1sig(cumcounts))

	return N, eN_lo, eN_hi, cumcounts










