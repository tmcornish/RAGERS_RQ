############################################################################################################
# Module containing functions relating to the construction of number counts.
###########################################################################################################

import general as gen
import stats
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
import scipy.optimize as opt
from scipy.stats import poisson
from scipy.integrate import quad
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from astropy.table import Table
from matplotlib.ticker import AutoMinorLocator
import emcee
import plotstyle as ps
import mpmath
from scipy.special import gamma, gammaincc, expn 

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
		#see if additional axes were provided for the completeness to be plotted on
		if 'other_axes' in plot_kwargs:
			ax_other = plot_kwargs['other_axes']
		else:
			ax_other = None
		#see if a plotstyle was provided; otherwise use a preset one
		if 'plotstyle' in plot_kwargs:
			plt.style.use(plot_kwargs['plotstyle'])
		else:
			plt.style.use(ps.styledict)
		#if not already using LaTeX formatting, do so
		if not mpl.rcParams['text.usetex']:
			mpl.rcParams['text.usetex'] = True


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
		
		if 'plot_extrap' in plot_kwargs:
			if plot_grid and plot_kwargs['plot_extrap']:
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



def differential_numcounts(S, bin_edges, A, comp=None, incl_poisson=True, return_dist=False):
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

	incl_poisson: bool
		Whether to include Poissonian uncertainties in the errorbars. 

	return_dist: bool
		If True, just returns the distributions of counts and bin heights.

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

		N_rand = counts * weights
		
		if return_dist:
			return N_rand, counts

		#take the median values to be the true values and use the 16th and 84th percentiles to estiamte the uncertainties
		N16, N, N84 = np.percentile(N_rand, q=[stats.p16, 50, stats.p84], axis=0)
		eN_lo = N - N16
		eN_hi = N84 - N
		
		#includes Poissonian uncertainties - NOTE: currently treats the uncertainties as if they're
		#symmetric, which isn't necessarily true. Workarounds exist (see e.g. Barlow 2003) but they
		#give similar enough results for this to be okay for now.
		if incl_poisson:
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

		if return_dist:
			return N, counts

		#Poissonian uncertainties
		eN_lo, eN_hi = np.array(stats.poisson_errs_1sig(counts)) * weights

	return N, eN_lo, eN_hi, counts, weights




def cumulative_numcounts(counts=None, S=None, bin_edges=None, A=1., comp=None, incl_poisson=True, return_dist=False):
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

	incl_poisson: bool
		Whether to include Poissonian uncertainties in the errorbars. 

	return_dist: bool
		If True, just returns the distributions of counts and bin heights.

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
		cumcounts = np.cumsum(counts[:,::-1], axis=1)[:,::-1]
		N_rand = cumcounts / A

		if return_dist:
			return N_rand, cumcounts

		#take the median values to be the true values and use the 16th and 84th percentiles to estiamte the uncertainties
		N16, N, N84 = np.percentile(cumcounts, q=[stats.p16, 50, stats.p84], axis=0) / A
		eN_lo = N - N16
		eN_hi = N84 - N

		if incl_poisson:
			cumcounts_med = np.median(cumcounts, axis=0)
			eN_lo_p, eN_hi_p = np.array(stats.poisson_errs_1sig(cumcounts_med)) / A
			eN_lo = np.sqrt(eN_lo ** 2. + eN_lo_p ** 2.)
			eN_hi = np.sqrt(eN_hi ** 2. + eN_hi_p ** 2.)
	else:
		#calculate the cumulative counts
		cumcounts = np.cumsum(counts[::-1])[::-1]
		N = cumcounts / A

		if return_dist:
			return N, cumcounts

		eN_lo, eN_hi = np.array(stats.poisson_errs_1sig(cumcounts)) / A


	return N, eN_lo, eN_hi, cumcounts



def fit_schechter_mcmc(x, y, yerr, nwalkers, niter, initial, offsets=0.01, return_sampler=False, plot_on_axes=False, **plot_kwargs):
	'''
	Uses MCMC to fit a Schechter function to data.

	Parameters
	----------
	x: array-like
		Independent variable (flux density).

	y: array-like
		Dependent variable (surface density).

	yerr: array-like
		(Symmetric) uncertainties in the dependent variable.

	nwalkers: int
		Number of walkers to use when running the MCMC.

	niter: int
		Number of steps for each walker to take before the simulation completes.

	initial: list
		Initial guesses for each fit parameter.

	offsets: float or array-like
		Offsets in each dimension of the parameter space from an initial fit, used to determine
		the starting positions of each walker.

	return_sampler: bool
		If True, returns the full sampler output from running the MCMC in addition to the best-fit
		parameters and uncertainties.

	plot_on_axes: bool
		Plot the line of best fit on a provided set of axes.

	**plot_kwargs
		Any remaining keyword arguments will be used to format the plot (if generated).

	Returns
	----------
	best: array
			Best-fit parameters.

	e_lo: array
		Lower uncertainties on the best-fit parameters.

	e_hi: array
		Upper uncertainties on the best-fit parameters.

	sampler: EnsembleSamper
		EnsembleSampler object obtained by running emcee. Only returned if return_sampler=True.

	'''

	def model(theta, S):
		'''
		The model form of the Schechter function to be fitted to the data.

		Parameters
		----------
		theta: tuple
			Fit parameters (N0, S0, gamma).

		S: array-like
			Flux density.

		Returns
		----------
		y: array-like
			Model y-values for each value in x.
		'''

		N0, S0, gamma = theta
		y = (N0 / S0) * (S0 / S) ** gamma * np.exp(-S / S0)
		return y


	def lnlike(theta, x, y, yerr):
		'''
		Calculates the log-likelihood of the model.

		Parameters
		----------
		theta: tuple
			Fit parameters (N0, S0, gamma).

		x: array-like
			Independent variable (flux density).

		y: array-like
			Dependent variable (surface density).

		yerr: array-like
			(Symmetric) uncertainties in the dependent variable.

		Returns
		----------
		L: float
			Log-likelihood.
		'''

		ymodel = model(theta, x)
		L = -0.5 * np.sum(((y - ymodel) / yerr) ** 2.)
		return L

	def lnprior(theta):
		'''
		Sets the priors on the fit parameters.

		Parameters
		----------
		theta: tuple
			Fit parameters (N0, S0, gamma).

		Returns
		----------
		0 or inf: float
			Returns 0 if the parameters satisfy the priors, and returns -inf if one or more of 
			them does not.
		'''

		N0, S0, gamma = theta
		if (1000. <= N0 <= 15000.) and (1. <= S0 <= 10.) and (-1. <= gamma <= 6.):
			return 0.
		else:
			return -np.inf

	def lnprob(theta, x, y, yerr):
		'''
		Calculates the logged probability of the data matching the fit.

		Parameters
		----------
		theta: tuple
			Fit parameters (N0, S0, gamma).

		x: array-like
			Independent variable (flux density).

		y: array-like
			Dependent variable (surface density).

		yerr: array-like
			(Symmetric) uncertainties in the dependent variable.

		Returns
		----------
		P: float
			Logged probability of the data matching the fit.
		'''

		lp = lnprior(theta)
		if not np.isfinite(lp):
			return -np.inf
		else:
			return lp + lnlike(theta, x, y, yerr)


	def main(p0, nwalkers, niter, ndim, lnprob, data):
		'''
		Actually runs the MCMC to perform the fit.

		Parameters
		----------

		p0: list
			List of initial starting points for each walker.

		nwalkers: int
			Number of walkers to use when running the MCMC.

		niter: int
			Number of steps for each walker to take before the simulation completes.

		ndim: int
			Number of parameters to fit.

		lnprob: callable
			Function for calculating the logged probability of the data matching the fit.

		data: tuple or array-like 
			Contains (as three separate entries) the x values, the y values, and the uncertainties
			in y.

		'''

		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

		print("Running burn-in...")
		p0, _, _ = sampler.run_mcmc(p0, 100)
		sampler.reset()

		print("Running production...")
		pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)

		return sampler, pos, prob, state

	#convert initial conditions to an array if not provided as one already
	initial = np.asarray(initial)
	offsets = np.asarray(offsets)
	#number of parameters to be fitted
	ndim = len(initial)

	#use an optimiser to get a better initial estimate for the fit parameters
	nll = lambda *args: -lnlike(*args)
	#add an error catcher to reattempt with different initial conditions if first attempt fails
	while True:
		result = opt.minimize(nll, x0=initial, args=(x, y, yerr))
		if result.success:
			break
		else:
			initial = initial + offsets * np.random.randn(ndim)
	initial = result['x']

	#initial starting points for each walker in the MCMC
	p0 = [initial + offsets * np.random.randn(ndim) for i in range(nwalkers)]

	data = (x, y, yerr)
	sampler, pos, prob, state = main(p0,nwalkers,niter,ndim,lnprob,data)
	samples = sampler.flatchain
	theta_max  = samples[np.argmax(sampler.flatlnprobability)]

	#get the median, 16th and 84th percentiles for each parameter
	'''
	q = np.percentile(samples, [stats.p16, 50., stats.p84], axis=0)
	best = q[1]
	uncert = np.diff(q, axis=0)
	e_lo = uncert[0]
	e_hi = uncert[1]
	'''
	q = np.percentile(samples, [stats.p16, stats.p84], axis=0)
	best = theta_max
	e_lo = theta_max - q[0]
	e_hi = q[1] - theta_max

	if plot_on_axes:
		#see if a set of axes has been provided - if not, give error message
		if 'ax' in plot_kwargs:
			#see if linestyle, linewidth, colour, alpha, label and zorder were provided as kwargs
			ls_key = [s for s in ['linestyle', 'ls'] if s in plot_kwargs]
			lw_key = [s for s in ['linewidth', 'lw'] if s in plot_kwargs]
			c_key = [s for s in ['colour', 'color', 'c'] if s in plot_kwargs]
			if len(ls_key) > 0:
				ls = plot_kwargs[ls_key[0]]
			else:
				ls = '-'
			if len(lw_key) > 0:
				lw = plot_kwargs[lw_key[0]]
			else:
				lw = 1.
			if len(c_key) > 0:
				c = plot_kwargs[c_key[0]]
			else:
				c = 'k'
			if 'alpha' in plot_kwargs:
				alpha = plot_kwargs['alpha']
			else:
				alpha = 1.
			if 'label' in plot_kwargs:
				label = plot_kwargs['label']
			else:
				label = ''
			if 'zorder' in plot_kwargs:
				zorder = plot_kwargs['zorder']
			else:
				zorder = 100

			#see if a range of x-values has been provided
			if 'x_range' in plot_kwargs:
				x_range = plot_kwargs['x_range']
			else:
				x_range = np.linspace(2., 20., 100)
			#see if an offset has beeen provided to avoid overlapping curves
			if 'x_offset' in plot_kwargs:
				x_range_plot = x_range * 10. ** plot_kwargs['x_offset']
			else:
				x_range_plot = x_range[:]

			#plot the line
			plot_kwargs['ax'].plot(x_range_plot, schechter_model(x_range, best), c=c, linestyle=ls, linewidth=lw, label=label, zorder=zorder)

			#see if told to also add text to the axes showing the best-fit values
			if 'add_text' in plot_kwargs:
				if plot_kwargs['add_text']:
					#see if the fontsize, position and colour of the text have been provided
					fs_key = [s for s in ['fontsize', 'fs'] if s in plot_kwargs]
					pos_key = [s for s in ['fontposition', 'fp', 'xyfont'] if s in plot_kwargs]
					fc_key = [s for s in ['fontcolour', 'fc'] if s in plot_kwargs]
					if len(fs_key) > 0:
						fs = plot_kwargs[fs_key[0]]
					else:
						fs = 14.
					if len(pos_key) > 0:
						xtext, ytext = plot_kwargs[pos_key[0]]
						if 'transform' in plot_kwargs:
							if plot_kwargs['transform'] == 'axes':
								transform = plot_kwargs['ax'].transAxes
						else:
							transform = plot_kwargs['ax'].transData
					else:
						xtext, ytext = 0.05, 0.5
						transform = plot_kwargs['ax'].transAxes
					if len(fc_key) > 0:
						fc = plot_kwargs[fc_key[0]]
					else:
						fc = 'k'

					best_fit_str = [
						r'$N_{0} = %.0f^{+%.0f}_{-%.0f}$'%(best[0],e_hi[0],e_lo[0]),
						r'$S_{0} = %.1f^{+%.1f}_{-%.1f}$'%(best[1],e_hi[1],e_lo[1]),
						r'$\gamma = %.1f^{+%.1f}_{-%.1f}$'%(best[2],e_hi[2],e_lo[2])
						]
					best_fit_str = '\n'.join(best_fit_str)
					plot_kwargs['ax'].text(xtext, ytext, best_fit_str, color=fc, ha='left', va='top', fontsize=fs, transform=transform)

	if return_sampler:
		return best, e_lo, e_hi, sampler
	else:
		return best, e_lo, e_hi


def mask_numcounts(x, y, limits=True, exclude_all_zero=True, Smin=None):
	'''
	Creates masks for the number counts data to account for the various visualisation possibilities
	- e.g. `normal' data points, data points excluded from subsequent analyses, limits. 

	Parameters
	----------
	x: array-like
		The x-values (flux density) for the number counts.

	y: array-like
		The y-values (dN/dS for differential or N(>S) for cumulative number counts).

	limits: bool
		Whether to replace with a limit the first zero bin outside the range of bins containing
		data.

	exclude_all_zero: bool
		Whether to remove bins with zero sources even if they lie between two non-zero bins.

	Smin: float or None
		If provided, sets a flux density limit below which bins are to be excluded from subsequent
		analyses.

	Returns
	-------
	masks: tuple
		Contains all the masks produced. By default this includes a mask for selecting `normal'
		data points, and a mask for selecting bins below a flux limit (mask is all False if no
		limit is provided, but optionally returns a mask for bins to be plotted as limits if told 
		to do so.

	'''

	#make a boolean array initially filled with all True
	true_array = np.full(len(x), True)

	#indices of all bins containing sources
	idx_data = np.where(y > 0)[0]

	#make a copy for identifying `normal' data points
	normal_data = true_array.copy()
	
	#remove sources according to the conditions provided
	if exclude_all_zero:
		zero_sources = y == 0.
	else:
		zero_sources = true_array.copy()
		try:
			zero_sources[idx_data.min():idx_data.max()+1] = False
		except IndexError:
			pass
	normal_data *= ~zero_sources

	if Smin is not None:
		faint_bins = x < Smin
	else:
		faint_bins = ~true_array
	normal_data *= ~faint_bins
	
	masks = (normal_data, faint_bins)
	
	if limits:
		limits = ~true_array
		try:
			limits[idx_data.max()+1] = True
		except IndexError:
			pass
		masks += (limits,)

	return masks




def plot_numcounts(x, y, xerr=None, yerr=None, ax=None, cumulative=False, masks=None, weights=None, offset=0., data_kwargs={}, ebar_kwargs={}, limit_kwargs={}, **plot_kwargs):
	'''
	Plots the number counts data, either on an existing set of axes or on a new figure.

	Parameters
	----------
	x: array-like
		The x-values (flux density) for the number counts.

	y: array-like
		The y-values (dN/dS for differential or N(>S) for cumulative number counts).

	xerr: array-like or None
		The x uncertainties on the data points.

	yerr: array-like or None
		The y uncertainties on the data points.

	ax: Axes object or None
		Axes on which the data should be plotted. If None will make a new figure and return it.
	
	cumulative: bool
		Number counts are cumulative rather than differential

	masks: array or tuple or None
		Contains masks for different plotting conditions. If 1D, will only plot those bins as normal
		data points. If 2D, the masks must follow the order of (1) normal data points, (2) faint bins,
		and (3) limits.

	weights: array-like or None
		Weights used to convert from counts to bin heights - to be used when plotting limits.

	offset: float
		Offset to apply in the x-direction for all bins (used to avoid overlap between different
		datasets).

	data_kwargs: dict
		Dictionary of kwargs for formatting the plotted data.

	ebar_kwargs: dict
		Dictionary of kwargs for formatting the plotted errorbars.

	**plot_kwargs
		Any remaining keyword arguments will be used to format the plot (if generated).

	Returns
	-------
	f, ax: Figure and Axes
		Returns the new figure and axes if produced.
	'''

	#see if a plotstyle has been provided
	if 'plotstyle' in plot_kwargs:
		plt.style.use(plot_kwargs['plotstyle'])
	else:
		plt.style.use(ps.styledict)

	#see if a set of axes has been provided; if not, make a new figure
	if ax is None:
		f, ax = plt.subplots()
		ax.set_xlabel(r'$S_{850~\mu{\rm m}}$ (mJy)')
		if cumulative:
			ax.set_ylabel(r'$N(>S)$ (deg$^{-2}$)')
		else:
			ax.set_ylabel(r'$dN/dS$ (deg$^{-2}$ mJy$^{-1}$)')
		ax.set_xscale('log')
		ax.set_yscale('log')

	#if no errors provided in x and/or y, set them to zero
	if xerr is None:
		xerr = np.zeros((2,len(x)))
	#if 1D array of uncertainties provided, broadcast to 2D for convenience
	elif gen.get_ndim(xerr) == 1:
		xerr = np.tile(xerr, (2,1))
	if yerr is None:
		yerr = np.zeros((2,len(x)))
	#if 1D array of uncertainties provided, broadcast to 2D for convenience
	elif gen.get_ndim(yerr) == 1:
		yerr = np.tile(yerr, (2,1))

	#apply the provided x-offset to the bins (accounting for log space)
	x_plot = x * 10. ** offset

	#see if any masks were provided
	if masks is not None:
		#find out if more than one mask was provided
		ndim_masks = gen.get_ndim(masks)
		#if only one provided, plot only these sources as normal data points
		if ndim_masks == 1:
			#plot these bins
			ax.plot(x_plot[masks], y[masks], **data_kwargs)
			ax.errorbar(x_plot[masks], y[masks], xerr=(xerr[0][masks], xerr[1][masks]), yerr=(yerr[0][masks], yerr[1][masks]), fmt='none', **ebar_kwargs)
		#otherwise, use the first mask to plot the normal data points
		else:
			ax.plot(x_plot[masks[0]], y[masks[0]], **data_kwargs)
			ax.errorbar(x_plot[masks[0]], y[masks[0]], xerr=(xerr[0][masks[0]], xerr[1][masks[0]]), yerr=(yerr[0][masks[0]], yerr[1][masks[0]]), fmt='none', **ebar_kwargs)
			#see if a second mask has been provided
			if len(masks) > 1:
				#remove the 'label' keyword from data_kwargs if it exists
				if 'label' in data_kwargs:
					del data_kwargs['label']
				#use the second mask to plot bins below a previously defined flux limit as open data points
				ax.plot(x_plot[masks[1]], y[masks[1]], mfc='none', **data_kwargs)
				ax.errorbar(x_plot[masks[1]], y[masks[1]], xerr=(xerr[0][masks[1]], xerr[1][masks[1]]), yerr=(yerr[0][masks[1]], yerr[1][masks[1]]), fmt='none', **ebar_kwargs)
				#see if a third mask is provided
				if len(masks) > 2:
					#calculate the y-values at which these limits should be plotted
					if weights is not None:
						#use the third mask to plot bins as limits
						lim_data, = ax.plot(x_plot[masks[2]], weights[masks[2]], **data_kwargs)
						#ensure the arrows have the same colour as the data points
						limit_kwargs['color'] = lim_data.get_color()
						ax.quiver(x_plot[masks[2]], weights[masks[2]], 0., -1., **limit_kwargs)
					#else:
						#gen.error_message('numcounts.plot_numcounts', 'Need to provide weights in order to plot limits.')

	#if no masks provided, just plot the data
	else:
		ax.plot(x, y, **data_kwargs)
		ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='none', **ebar_kwargs)


	#if a new figure was made, return that and the axes
	if 'f' in locals():
		return f, ax
	else:
		return None


def Gamma_integrand(t, s):
	return t ** (s - 1.) * np.exp(-t)

def Gamma_upper(s, x):
	'''
	Upper incomplete gamma function.

	Parameters
	----------

	s: float
		Lower bound of the integral used to compute the Gamma function.

	x: float
		Controls the exponent of the variable in the integrand.
	'''
	#return float(mpmath.gammainc(s, x))

	G = quad(Gamma_integrand, x, np.inf, args=(s,))[0]
	return G


#create a new version of the above function that's compatible with numpy arrays
Gamma_upper_vec = np.frompyfunc(Gamma_upper, 2, 1)

'''
def gamma_upper(s, x):
	return float(mpmath.gammainc(s, x, np.inf))

gamma_upper_vec = np.frompyfunc(gamma_upper, 2, 1)

def gamma_upper_vec_(s, x):
	return gamma_upper_vec(s, x).astype(float)
'''

def gammainc_up(a, x):
	'''
	Upper incomplete gamma function, generalised to include a <= 0.

	Parameters
	----------
	a: float
		Lower bound of the integral used to compute the Gamma function.
	
	x: float or array-like
		Controls the exponent of the variable in the integrand.

	Returns
	-------
	G: float or array-like
		Value(s) of the upper incomplete gamma function evaluated at x.
	'''

	#if a is positive, simply use scipy's functions to calculate the result
	if a > 0:
		return gammaincc(a,x) * gamma(a)
	#if a = 0, use relation Gamma(0,x) = E1(x)
	elif a == 0:
		return expn(1, x)
	#if a < 0, use recursion relation to calculate the result
	else:
		a_ = a - np.floor(a)
		j = int(a_ - a)
		if a_ == 0:
			G = expn(1, x)
		else:
			G = gammaincc(a_, x) * gamma(a_)
		for i in range(j):
			a_ -= 1
			G = (1/a_) * (G - x**(a_) * np.exp(-x))
		return G


def cumulative_model(S, params):
	'''
	Integral of a model Schechter function of the form used for number counts.

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
		#y = N0 * Gamma_upper_vec(-gamma + 1., S / S0)
		y = N0 * gammainc_up(-gamma + 1., S / S0)
	return y


def fit_cumulative_mcmc(x, y, yerr, nwalkers, niter, initial, offsets=0.01, return_sampler=False, plot_on_axes=False, **plot_kwargs):
	'''
	Uses MCMC to fit a renormalised incomplete Gamma function to data. (Used for cumulative number counts.)

	Parameters
	----------
	x: array-like
		Independent variable (flux density).

	y: array-like
		Dependent variable (surface density).

	yerr: array-like
		(Symmetric) uncertainties in the dependent variable.

	nwalkers: int
		Number of walkers to use when running the MCMC.

	niter: int
		Number of steps for each walker to take before the simulation completes.

	initial: list
		Initial guesses for each fit parameter.

	offsets: float or array-like
		Offsets in each dimension of the parameter space from an initial fit, used to determine
		the starting positions of each walker.

	return_sampler: bool
		If True, returns the full sampler output from running the MCMC in addition to the best-fit
		parameters and uncertainties.

	plot_on_axes: bool
		Plot the line of best fit on a provided set of axes.

	**plot_kwargs
		Any remaining keyword arguments will be used to format the plot (if generated).

	Returns
	----------
	best: array
			Best-fit parameters.

	e_lo: array
		Lower uncertainties on the best-fit parameters.

	e_hi: array
		Upper uncertainties on the best-fit parameters.

	sampler: EnsembleSamper
		EnsembleSampler object obtained by running emcee. Only returned if return_sampler=True.

	'''

	def model(theta, S):
		'''
		The model form of the Schechter function to be fitted to the data.

		Parameters
		----------
		theta: tuple
			Fit parameters (N0, S0, gamma).

		S: array-like
			Flux density.

		Returns
		----------
		y: array-like
			Model y-values for each value in x.
		'''

		N0, S0, gamma = theta
		y = N0 * gammainc_up(-gamma + 1., S / S0)
		return y


	def lnlike(theta, x, y, yerr):
		'''
		Calculates the log-likelihood of the model.

		Parameters
		----------
		theta: tuple
			Fit parameters (N0, S0, gamma).

		x: array-like
			Independent variable (flux density).

		y: array-like
			Dependent variable (surface density).

		yerr: array-like
			(Symmetric) uncertainties in the dependent variable.

		Returns
		----------
		L: float
			Log-likelihood.
		'''

		ymodel = model(theta, x)
		L = -0.5 * np.sum(((y - ymodel) / yerr) ** 2.)
		return L

	def lnprior(theta):
		'''
		Sets the priors on the fit parameters.

		Parameters
		----------
		theta: tuple
			Fit parameters (N0, S0, gamma).

		Returns
		----------
		0 or inf: float
			Returns 0 if the parameters satisfy the priors, and returns -inf if one or more of 
			them does not.
		'''

		N0, S0, gamma = theta
		if (1000. <= N0 <= 15000.) and (1. <= S0 <= 10.) and (-1. <= gamma <= 6.):
			return 0.
		else:
			return -np.inf

	def lnprob(theta, x, y, yerr):
		'''
		Calculates the logged probability of the data matching the fit.

		Parameters
		----------
		theta: tuple
			Fit parameters (N0, S0, gamma).

		x: array-like
			Independent variable (flux density).

		y: array-like
			Dependent variable (surface density).

		yerr: array-like
			(Symmetric) uncertainties in the dependent variable.

		Returns
		----------
		P: float
			Logged probability of the data matching the fit.
		'''

		lp = lnprior(theta)
		if not np.isfinite(lp):
			return -np.inf
		else:
			return lp + lnlike(theta, x, y, yerr)


	def main(p0, nwalkers, niter, ndim, lnprob, data):
		'''
		Actually runs the MCMC to perform the fit.

		Parameters
		----------

		p0: list
			List of initial starting points for each walker.

		nwalkers: int
			Number of walkers to use when running the MCMC.

		niter: int
			Number of steps for each walker to take before the simulation completes.

		ndim: int
			Number of parameters to fit.

		lnprob: callable
			Function for calculating the logged probability of the data matching the fit.

		data: tuple or array-like 
			Contains (as three separate entries) the x values, the y values, and the uncertainties
			in y.

		'''

		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

		print("Running burn-in...")
		p0, _, _ = sampler.run_mcmc(p0, 100)
		sampler.reset()

		print("Running production...")
		pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)

		return sampler, pos, prob, state

	#convert initial conditions to an array if not provided as one already
	initial = np.asarray(initial)
	offsets = np.asarray(offsets)
	#number of parameters to be fitted
	ndim = len(initial)
	
	#use an optimiser to get a better initial estimate for the fit parameters
	nll = lambda *args: -lnlike(*args)
	#add a catch for in case the initial conditions cause an error
	while True:
		try:
			result = opt.minimize(nll, x0=initial, args=(x, y, yerr))
			break
		except TypeError:
			initial = initial + offsets * np.random.randn(ndim)

	initial = result['x']

	#initial starting points for each walker in the MCMC
	p0 = [initial + offsets * np.random.randn(ndim) for i in range(nwalkers)]

	data = (x, y, yerr)
	sampler, pos, prob, state = main(p0,nwalkers,niter,ndim,lnprob,data)
	samples = sampler.flatchain
	theta_max  = samples[np.argmax(sampler.flatlnprobability)]

	#get the median, 16th and 84th percentiles for each parameter
	'''
	q = np.percentile(samples, [stats.p16, 50., stats.p84], axis=0)
	best = q[1]
	uncert = np.diff(q, axis=0)
	e_lo = uncert[0]
	e_hi = uncert[1]
	'''
	q = np.percentile(samples, [stats.p16, stats.p84], axis=0)
	best = theta_max
	e_lo = theta_max - q[0]
	e_hi = q[1] - theta_max

	if plot_on_axes:
		#see if a set of axes has been provided - if not, give error message
		if 'ax' in plot_kwargs:
			#see if linestyle, linewidth, colour, alpha, label and zorder were provided as kwargs
			ls_key = [s for s in ['linestyle', 'ls'] if s in plot_kwargs]
			lw_key = [s for s in ['linewidth', 'lw'] if s in plot_kwargs]
			c_key = [s for s in ['colour', 'color', 'c'] if s in plot_kwargs]
			if len(ls_key) > 0:
				ls = plot_kwargs[ls_key[0]]
			else:
				ls = '-'
			if len(lw_key) > 0:
				lw = plot_kwargs[lw_key[0]]
			else:
				lw = 1.
			if len(c_key) > 0:
				c = plot_kwargs[c_key[0]]
			else:
				c = 'k'
			if 'alpha' in plot_kwargs:
				alpha = plot_kwargs['alpha']
			else:
				alpha = 1.
			if 'label' in plot_kwargs:
				label = plot_kwargs['label']
			else:
				label = ''
			if 'zorder' in plot_kwargs:
				zorder = plot_kwargs['zorder']
			else:
				zorder = 100

			#see if a range of x-values has been provided
			if 'x_range' in plot_kwargs:
				x_range = plot_kwargs['x_range']
			else:
				x_range = np.linspace(2., 20., 100)
			#see if an offset has beeen provided to avoid overlapping curves
			if 'x_offset' in plot_kwargs:
				x_range_plot = x_range * 10. ** plot_kwargs['x_offset']
			else:
				x_range_plot = x_range[:]

			#plot the line
			plot_kwargs['ax'].plot(x_range_plot, cumulative_model(x_range, best), c=c, linestyle=ls, linewidth=lw, label=label, zorder=zorder)

			#see if told to also add text to the axes showing the best-fit values
			if 'add_text' in plot_kwargs:
				if plot_kwargs['add_text']:
					#see if the fontsize, position and colour of the text have been provided
					fs_key = [s for s in ['fontsize', 'fs'] if s in plot_kwargs]
					pos_key = [s for s in ['fontposition', 'fp', 'xyfont'] if s in plot_kwargs]
					fc_key = [s for s in ['fontcolour', 'fc'] if s in plot_kwargs]
					if len(fs_key) > 0:
						fs = plot_kwargs[fs_key[0]]
					else:
						fs = 14.
					if len(pos_key) > 0:
						xtext, ytext = plot_kwargs[pos_key[0]]
						if 'transform' in plot_kwargs:
							if plot_kwargs['transform'] == 'axes':
								transform = plot_kwargs['ax'].transAxes
						else:
							transform = plot_kwargs['ax'].transData
					else:
						xtext, ytext = 0.05, 0.5
						transform = plot_kwargs['ax'].transAxes
					if len(fc_key) > 0:
						fc = plot_kwargs[fc_key[0]]
					else:
						fc = 'k'

					best_fit_str = [
						r'$N_{0} = %.0f^{+%.0f}_{-%.0f}$'%(best[0],e_hi[0],e_lo[0]),
						r'$S_{0} = %.1f^{+%.1f}_{-%.1f}$'%(best[1],e_hi[1],e_lo[1]),
						r'$\gamma = %.1f^{+%.1f}_{-%.1f}$'%(best[2],e_hi[2],e_lo[2])
						]
					best_fit_str = '\n'.join(best_fit_str)
					plot_kwargs['ax'].text(xtext, ytext, best_fit_str, color=fc, ha='left', va='top', fontsize=fs, transform=transform)

	if return_sampler:
		return best, e_lo, e_hi, sampler
	else:
		return best, e_lo, e_hi







