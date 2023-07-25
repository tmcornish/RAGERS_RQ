############################################################################################################
# Module containing functions for the analysis and propagation of uncertainties.
###########################################################################################################

import numpy as np
from scipy.stats import truncnorm
import scipy.special as special
from scipy import optimize as opt
from multiprocessing import Pool, cpu_count
import general as gen
import warnings
import random
import pandas as pd
import itertools
from functools import reduce
from scipy.interpolate import interp1d

#DataFrame containing key values of Delta Chi^2 corresponding to different sigma for different degrees
#of freedom, k. NOTE: For testing goodness of fit, k = N_data - N_params, but for estimating paramter
#uncertainties, k = N_params.
chi2_dict = {
	'k'	   : list(range(1,11)),
	'1sig' : [1., 2.3, 3.53, 4.72, 5.89, 7.04, 8.18, 9.3, 10.42, 11.54],
	'2sig' : [4., 6.18, 8.02, 9.72, 11.31, 12.85, 14.34, 15.79, 17.21, 18.61],
	'3sig' : [9., 11.83, 14.16, 16.25, 18.21, 20.06, 21.85, 23.57, 25.26, 26.9],
	'4sig' : [16., 19.33, 22.06, 24.50, 26.77, 28.91, 30.96, 32.93, 34.85, 36.72],
	'5sig' : [25., 28.74, 31.81, 34.56, 37.09, 39.49, 41.78, 43.98, 46.12, 48.19]
}
chi2_df = pd.DataFrame(chi2_dict).set_index('k')

def percentiles_nsig(n):
	'''
	Returns the lower and upper percentiles corresponding to the n-sigma bounds for a normal distribution.
		n: The multiple of sigma for which the percentiles are to be calculated.
	'''
	#fraction of population within the range [mu-n*sigma, mu+n*sigma]
	f = special.erf(n / np.sqrt(2.))
	#percentiles
	p_lo = 0.5 * (1. - f) * 100.
	p_hi = 0.5 * (1. + f) * 100.
	return (p_lo, p_hi)

#percentiles corresponding to 1 sigma about the median
p16, p84 = percentiles_nsig(1.)


def gaussian(x, x0, sigma):
	'''
	Creates a Gaussian distribution in x about the mean and returns its values.
		x: The values at which the Gaussian is to be calculated.
		x0: The mean of the distribution.
		sigma: The standard deviation of the distribution.
	'''
	gauss = np.exp(-((x - x0) ** 2.) / (2. * sigma ** 2.))
	return gauss


def random_gaussian(mu, sigma, N):
	'''
	Randomly draws values from a normal distribution or set of normal distributions.
		mu: Mean(s) of the distribution(s).
		sigma: Standard deviation(s) of the distribution(s).
		N: Number of random values to be drawn from the distribution(s).
	'''
	#case where mu and sigma are arrays
	try:
		l = len(mu)
		rand = np.random.normal(size=(l,N)) * sigma[:,None] + mu[:,None]
	#case where mu and sigma are single values
	except TypeError:
		rand = np.random.normal(size=N) * sigma + mu

	#return the randomly selected values
	return rand


def random_asymmetric_gaussian(mu, sigma_lo, sigma_hi, N, xmin=None, xmax=None):
	'''
	Randomly draws values from an asymmetric 'normal' distribution. To be used e.g. when propagating 
	asymmetric uncertainties. NOTE: Only works for single-value arguments.
		mu: Mean(s) of the distribution(s).
		sigma_lo: Standard deviation(s) of the distribution(s) below the mean(s).
		sigma_hi: Standard deviation(s) of the distribution(s) above the mean(s).
		N: Number of random values to be drawn from the distribution(s).
		xmin: Minimum allowed value for x; if None, assumes -inf.
		xmax: Maximum allowed value for x; if None, assumes inf.
	'''
	#rescale the limiting values of x (if provided)
	if xmin is not None:
		xmin = (xmin - mu) / sigma_lo
	else:
		xmin = -np.inf
	if xmax is not None:
		xmax = (xmax - mu) / sigma_hi
	else:
		xmax = np.inf
	#choose the number of values to draw from each side of the distribution based on the ratio of the uncertainties
	#N_hi = int(np.nan_to_num(np.ceil(N / (1. + sigma_lo / sigma_hi)), nan=0.))
	#N_lo = int(np.floor(N - N_hi))
	#if told to drawn >1 values, draw half from each side of the distribution
	if N > 1:
		N_lo = N_hi = int(N/2)
	else:
		#otherwise, use a 'coin flip' to choose which side to draw the single value from
		coin_flip = random.random()
		if coin_flip < 0.5:
			N_lo = 1
			N_hi = 0
		else:
			N_lo = 0
			N_hi = 1
	if (xmin < 0.) and (xmax > 0.):
		rand_lo = truncnorm.rvs(xmin, 0., size=N_lo) * sigma_lo + mu
		rand_hi = truncnorm.rvs(0., xmax, size=N_hi) * sigma_hi + mu
	elif (xmin >= 0.) and (xmax > 0.):
		rand_lo = []
		rand_hi = truncnorm.rvs(0., xmax, size=N) * sigma_hi + mu
	elif (xmin < 0.) and (xmax <= 0.):
		rand_hi = []
		rand_lo = truncnorm.rvs(xmin, 0., size=N) * sigma_lo + mu
	else:
		gen.error_message('stats.random_asymmetric_gaussian', 'Attempting to draw from a distribution with width of 0. Try loosening the constraints.')
		rand_lo = []
		rand_hi = []
	#scale and shift each distribution according to the standard deviations and means
	#rand_lo = rand_lo * sigma_lo + mu
	#rand_hi = rand_hi * sigma_hi + mu
	#concatenate the two randomly drawn samples
	rand_all = np.concatenate([rand_lo, rand_hi])
	#shuffle the concatenated array so that draws from the two half-gaussians are mixed
	np.random.shuffle(rand_all)
	return rand_all



def chisq(x, y, yerr, ymodel):
	'''
	Calculates chi squared for a set of observed and model values at given x-coordinates.

	Parameters
	----------
	x: array-like
		x-values at which the observations were taken.

	y: array-like
		y-values for each observation.

	yerr: array-like
		Uncertainties on each observed y-value (assumed symmetric).

	ymodel: array-like
		Model y-values for each x-value, calculated using some model function.

	Returns
	----------
	chisq: float
		The chi-squared value for the function.
	'''
	#calculate chi-squared
	chi2 = np.sum(((ymodel - y)/yerr) ** 2.)
	return chi2


def chisq_minimise(x, y, func, initial, yerr=None):
	'''
	Takes a function and performs chi-squared minimisation to determine the best-fit parameters
	for a set of observations.

	Parameters
	----------
	x: array-like
		x-values at which the observations were taken.

	y: array-like
		y-values for each observation.

	func: callable
		Function to be fitted. Must have the form func(x, params) where params is array-like and
		comprises the function parameters.

	initial: array-like
		Initial guesses for the best-fit parameters (must have the same order as when they are fed
		to the function).

	yerr: array-like or None
		Uncertainties on each observed y-value. If a 2D array, assumed to contain the lower and
		upper uncertainties. If None, defines chi-squared as chi^2 = sum((obs-exp)^2/exp).

	Returns
	----------
	popt: array
		The best-fit parameters.

	chi2min: float
		The chi-squared value for the best fit.

	'''

	if yerr is not None:
		#get the dimensions of yerr
		ndim = gen.get_ndim(yerr)
		#symmetrise the uncertainties if asymmetric uncertainties provided
		if ndim == 2:
			yerr = (yerr[0] + yerr[1]) / 2.

		def chisq_model(params, x_, y_, yerr_):
			'''
			Calculates chi-squared for the model function and a given dataset.

			Parameters
			----------
			params: array-like
				List/tuple/array containing the model function parameters.
		
			x_: array-like
				x-values at which the observations were taken.

			y_: array-like
				y-values for each observation.

			yerr_: array-like
				Uncertainties on each observed y-value (assumed symmetric).

			Returns
			----------
			chisq: float
				The chi-squared value for the function.
			'''
			ymodel = func(x_, params)
			chi2 = np.sum(((ymodel - y_)/yerr_) ** 2.)
			return chi2

		#reformat the chisq_model function so that it is compatible with scipy.optimize.minimize
		nll = lambda *args : chisq_model(*args)
		#run the optimisation to find the best-fit parameters that minimize chi squared
		res = opt.minimize(nll, x0=initial, args=(x, y, yerr))

	else:
		def chisq_model(params, x_, y_):
			'''
			Calculates chi-squared for the model function and a given dataset.

			Parameters
			----------
			params: array-like
				List/tuple/array containing the model function parameters.
		
			x_: array-like
				x-values at which the observations were taken.

			y_: array-like
				y-values for each observation.

			Returns
			----------
			chisq: float
				The chi-squared value for the function.
			'''
			ymodel = func(x_, params)
			chi2 = np.sum(((y - ymodel) ** 2.) / ymodel)
			return chi2

		#reformat the chisq_model function so that it is compatible with scipy.optimize.minimize
		nll = lambda *args : chisq_model(*args)
		#run the optimisation to find the best-fit parameters that minimize chi squared
		res = opt.minimize(nll, x0=initial, args=(x, y))

	#retrieve the best-fit parameters and minimum chi squared value
	popt = res.x
	chi2min = res.fun
	return popt, chi2min


def uncertainties_in_fit(x, y, yerr, func, initial, use_yerr_in_fit=True, nsim=10000, nsigma=1., return_dist=False, ncpu=0):
	'''
	Estimates the uncertainties on a fit by perturbing the observations according to their
	uncertainties. NOTE: currently assumes uncorrelated uncertainties. To account for correlated
	uncertainties, will need to reperform the fit letting only one parameter vary at a time (fixing
	the others to their best-fit values).

	Parameters
	----------
	x: array-like
		x-values at which the observations were taken.

	y: array-like
		y-values for each observation.

	yerr: array-like
		Uncertainties on each observed y-value. If a 2D array, assumed to contain the lower and
		upper uncertainties.

	func: callable
		Function to be fitted. Must have the form func(x, params) where params is array-like and
		comprises the function paramters.

	initial: array-like
		Initial guesses for the best-fit parameters (must have the same order as when they are fed
		to the function).

	use_yerr_in_fit: bool
		Use the uncertainties in y when fitting the function to each simulated dataset.

	nsim: int
		The number of iterations to perform when estimating the uncertainties.

	nsigma: float
		The significance level for which the uncertainties are being estimated.

	return_dist: bool
		Also return the full distribution for each fit parameter.

	Returns
	----------
	e_lo: array-like
		The lower uncertainties on each fit parameter.

	e_hi: array-like
		The upper uncertainties on each fit parameter.

	p_all: array
		Values of each fit parameter at the desired percentiles. Has shape (N, 3) where N is
		the number of fit parameters.

	popt_all: array
		Best-fit parameters for all simulated datasets.

	chi2min_all: array
		Chi-squared values for the fits to all simulated datasets.

	'''

	#set up a multiprocessing Pool using the desired number of CPUs
	if ncpu == 0:
		pool = Pool(cpu_count()-1)
	else:
		pool = Pool(ncpu)

	#get the dimensions of yerr
	ndim = gen.get_ndim(yerr)
	#if 2-dimensional, contains lower and upper uncertainties
	if ndim == 2:
		#created nsim randomly generated datasets
		y_rand = np.array([random_asymmetric_gaussian(y[i], yerr[0][i], yerr[1][i], nsim) for i in range(len(y))]).T
		#symmetrise the uncertainties
		yerr = (yerr[0] + yerr[1]) / 2.
	#if only 1-dimensional, uncertainties are symmetric
	else:
		y_rand = random_gaussian(y, yerr, nsim).T

	#for each generated dataset, perform a fit and store the best-fit parameters
	with np.errstate(all='ignore'):
		if use_yerr_in_fit:
			popt_all, chi2min_all = zip(*pool.starmap(chisq_minimise, [[x, y_rand[i], func, initial, yerr] for i in range(nsim)]))
		else:
			popt_all, chi2min_all = zip(*pool.starmap(chisq_minimise, [[x, y_rand[i], func, initial, None] for i in range(nsim)]))
	pool.close()
	pool.join()

	popt_all = np.array(popt_all)
	chi2min_all = np.array(chi2min_all)

	#calculate the desired percentiles (and the median) for each parameter 
	perc_lo, perc_hi = percentiles_nsig(nsigma)
	perc = [perc_lo, 50., perc_hi]
	if gen.get_ndim(popt_all) > 1:
		p_lo, p_med, p_hi = np.array([np.nanpercentile(popt_all[:,i], perc) for i in range(len(popt_all[0]))]).T
	else:
		p_lo, p_med, p_hi = np.nanpercentile(popt_all, perc)
	#combine the percentiles into one array
	p_all = np.array([p_lo, p_med, p_hi]).T
	#calculate the uncertainties
	e_lo = p_med - p_lo
	e_hi = p_hi - p_med

	return e_lo, e_hi, p_all, popt_all, chi2min_all


def poisson_CI_1sig(N):
	'''
	Using the approximations from Gehrels (1986), calculates the bounds of the 1-sigma confidence
	interval of the Poisson distribution for N occurrences.

	Parameters
	----------
	N: int or float or array-like
		Number of occurrences.

	Returns
	----------
	Nlo, Nhi: floats or array-like
		Lower and upper bounds of the 1-sigma confidence interval, respectively.
	'''

	#account for possibility of 0 occurrences
	if gen.get_ndim(N) == 0:
		try:
			Nlo = N * (1. - 1./(9.*N) - 1./(3.*np.sqrt(N))) ** 3.
		except ZeroDivisionError:
			Nlo = 0.
	else:
		with np.errstate(all='ignore'):
			Nlo = N * (1. - 1./(9.*N) - 1./(3.*np.sqrt(N))) ** 3.
		Nlo[np.isinf(Nlo) + np.isnan(Nlo)] = 0.
	Nhi = N + np.sqrt(N + 0.75) + 1.

	return Nlo, Nhi


def poisson_errs_1sig(N):
	'''
	Using the approximations from Gehrels (1986), calculates the 1-sigma Poissonian uncertainties
	for N occurrences.

	Parameters
	----------
	N: int or float or array-like
		Number of occurrences.

	Returns
	----------
	elo, ehi: floats or array-like
		Lower and upper 1-sigma uncertainties, respectively.
	'''
	elo, ehi = np.abs(np.array(poisson_CI_1sig(N)) - N)
	return elo, ehi



def hpd(trace, mass_frac) :
	'''
	Returns highest probability density region given by
	a set of samples.

	Parameters
	----------
	trace : array
		1D array of MCMC samples for a single variable

	mass_frac : float with 0 < mass_frac <= 1
		The fraction of the probability to be included in
		the HPD.  For example, `massfrac` = 0.95 gives a
		95% HPD.

	Returns
	-------
	output : array, shape (2,)
		The bounds of the HPD
	'''
	# Get sorted list
	d = np.sort(np.copy(trace))

	# Number of total samples taken
	n = len(trace)

	# Get number of samples that should be included in HPD
	n_samples = np.floor(mass_frac * n).astype(int)

	# Get width (in units of data) of all intervals with n_samples samples
	int_width = d[n_samples:] - d[:n-n_samples]

	# Pick out minimal interval
	min_int = np.argmin(int_width)

	# Return interval
	return np.array([d[min_int], d[min_int+n_samples]])



def vals_and_errs_from_dist(dist, axis=0):
	'''
	Given a distribution of values, calculates the median along with upper and lower uncertainties.

	Parameters
	----------
	dist: ND array
		Distribution of values. If >1D, the various percentiles will be calculated along the specified
		axis.

	axis: int
		Axis along which the quantiles should be calculated.

	Returns
	-------
	val: float or array
		Median value(s) from the distribution(s).

	elo: float or array
		Lower uncertainties calculated from the distribution(s).

	ehi: float or array
		Upper uncertainties calculated from the distribution(s).
	'''

	val_lo, val, val_hi = np.nanpercentile(dist, [p16, 50., p84], axis=axis)
	#calculate the uncertainties
	elo = val - val_lo
	ehi = val_hi - val

	return val, elo, ehi


def chi2_func(params, x, y, yerr, func):
	ymodel = func(x, params)	
	chi2 = np.sum(((ymodel - y)/yerr) ** 2.)
	return chi2


def view1D(a, b):
	'''
	When used in conjunction with setdiff_nd (see below), removes from an array any entries which
	occur in another array.

	Parameters
	----------
	a: array
		Array from which elements are to be removed.

	b: array
		Array whose elements are to be excluded from array a.

	
	Returns
	-------
	1D versions of each array with flexible dtypes.

	'''
	a = np.ascontiguousarray(a)
	b = np.ascontiguousarray(b)
	void_dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
	return a.view(void_dt).ravel(),  b.view(void_dt).ravel()


def setdiff_nd(a,b):
	'''
	When used in conjunction with view1D (see above), removes from an array any entries which
	occur in another array.

	Parameters
	----------
	a: array
		Array from which elements are to be removed.

	b: array
		Array whose elements are to be excluded from array a.

	Returns
	-------
	Array a with any entries from array b removed.
	'''

	# a,b are the nD input arrays
	A,B = view1D(a,b)    
	return a[~np.isin(A,B)]



def chi2_min_fit(x, y, yerr, func, initial, res, nsig=3, Nmax=10000000):
	'''
	Takes a function and performs chi-squared minimisation to determine the best-fit parameters
	for a set of observations.

	Parameters
	----------
	x: array-like
		x-values at which the observations were taken.

	y: array-like
		y-values for each observation.

	yerr: array-like or None
		Uncertainties on each observed y-value. If a 2D array, assumed to contain the lower and
		upper uncertainties.

	func: callable
		Function to be fitted. Must have the form func(x, params) where params is array-like and
		comprises the function parameters.

	initial: array-like
		Initial guesses for the best-fit parameters (must have the same order as when they are fed
		to the function).

	res: float or array-like
		Desired resolution for each parameter when exploring the parameter space. If a single value,
		uses same resolution for all parameters.

	nsig: int
		The desired significance (in integer multiples of sigma) to be probed in the parameter space.

	Nmax: int
		The maximum number of entries to be allowed in any one given array when performing calculations
		(to conserve memory).
	'''	

	def chisq_model(params, x_, y_, yerr_):
		'''
		Calculates chi-squared for the model function and a given dataset.

		Parameters
		----------
		params: array-like
			List/tuple/array containing the model function parameters.
	
		x_: array-like
			x-values at which the observations were taken.

		y_: array-like
			y-values for each observation.

		yerr_: array-like
			Uncertainties on each observed y-value (assumed symmetric).

		Returns
		----------
		chisq: float
			The chi-squared value for the function.
		'''
		ymodel = func(x_, params)
		chi2 = np.sum(((ymodel - y_)/yerr_) ** 2.)
		return chi2

	#get the number of parameters
	N_param = len(initial)

	#get the dimensions of yerr
	ndim = gen.get_ndim(yerr)
	#symmetrise the uncertainties if asymmetric uncertainties provided
	if ndim == 2:
		yerr = (yerr[0] + yerr[1]) / 2.

	#find the value of Delta Chi^2 corresponding to the desired significance
	dchi2_nsig = chi2_df.loc[N_param][f'{int(nsig)}sig']

	#perform an initial chi-squared minimisation using an optimiser:
	#reformat the chisq_model function so that it is compatible with scipy.optimize.minimize
	nll = lambda *args : chisq_model(*args)
	#run the optimisation to find the best-fit parameters that minimize chi squared
	popt = opt.minimize(nll, x0=initial, args=(x, y, yerr))['x']

	#set the maximum number of entries along each axis an array with N_params dimensions can have
	#while still being smaller than Nmax
	N_per_dim = int(Nmax ** (1./N_param))

	#factors by which to scale the resolutions if the parameter space ends up being too small
	res_factor = np.full(N_param,1.)
	#convert the resolutions into an array if it is not one already
	res = np.array(res)

	#create an array to which all parameter combinations will be concatenated
	params_tried = np.empty((0, N_param), dtype=float)
	chi2_tried = np.empty(0, dtype=float)

	n_tries = 0
	while True:
		scaled_res = res_factor * res
		print(scaled_res)
		#define a grid in the parameter space centred on this initial estimate, firstly with far worse
		param_ranges = [
			np.arange(
				popt[i]-(N_per_dim/2)*scaled_res[i],
				popt[i]+(N_per_dim/2)*scaled_res[i],
				scaled_res[i]
				)
			for i in range(N_param)]
		#all unique combinations of the paramters in these ranges
		permut = np.array(list(itertools.product(*param_ranges)))
		#remove any rounding errors
		permut = permut - (permut % scaled_res)

		#remove any permutations that occurred in previous iterations
		permut = setdiff_nd(permut, params_tried)
		print(len(permut))

		'''
		#if a previous iteration has been run, remove all permutations within the previously probed param space
		if n_tries > 0:
			for i in range(N_param):
				p_mask_all = np.full(len(permut), False)
				#only mask the permutations using this parameter if extending the parameter space along this axis
				if not params_fine[i]:
					p_min = params_tried[-1][:,i].min()
					p_max = params_tried[-1][:,i].max()
					p_mask = (permut[:,i] < p_min) | (permut[:,i] > p_max)
					p_mask_all |= p_mask
				permut = permut[p_mask_all]
		'''
		#calculate the model y values for these combinations of parameters
		ymodel = func(x[:,None], permut.T)
		#get rid of any parameter combinations resulting in NaN model values for any bins
		params_keep = ~np.isnan(ymodel).all(axis=0)
		permut = permut.T[:,params_keep].T
		ymodel = ymodel[:,params_keep]
		#calculate chi^2 for each combination
		chi2 = np.sum(((ymodel - y[:,None]) / yerr[:,None]) ** 2., axis=0)

		#append the parameter combinations and corresponding chi^2 values to the list
		params_tried = np.concatenate([params_tried, permut])
		chi2_tried = np.concatenate([chi2_tried, chi2])

		#if previous iteration has been run, find min chi^2 across all iterations
		chi2_min = chi2_tried.min()

		#calculate Delta Chi^2
		dchi2 = chi2 - chi2_min

		#find where Delta Chi2 < the value corresponding to n sigma 
		dchi2_mask = dchi2 < dchi2_nsig

		print(dchi2_mask.sum(), len(chi2))

		#if there are no parameter combinations with Delta Chi^2 below the desired value, break
		if dchi2_mask.sum() == 0:
			break

		params_fine = []
		#cycle through the parameters
		for i in range(N_param):
			if (permut[:,i].min() == permut[:,i][dchi2_mask].min()) or (permut[:,i].max() == permut[:,i][dchi2_mask].max()):
				#res_factor[i] += 1.
				params_fine.append(False)
			else:
				params_fine.append(True)


		#add 1 to the number of attempts
		n_tries += 1

		if np.sum(params_fine) == N_param:
			break
		else:
			res_factor *= 2.

	return chi2_tried, params_tried




def chunks(iterable, size=10):
	'''
	Function for breaking a generator object into smaller chunks.

	Parameters
	----------
	iterable: generator
		Generator to split into chunks.

	size: int
		Size of each chunk.


	Yields
	------
	The next chunk in the sequence.

	'''
	iterator = iter(iterable)
	for first in iterator:
		yield itertools.chain([first], itertools.islice(iterator, size - 1))



def chi2_min_fit_simple(x, y, yerr, func, initial, res, nsig=3, Nmax=10000000):
	'''
	Takes a function and performs chi-squared minimisation to determine the best-fit parameters
	for a set of observations.

	Parameters
	----------
	x: array-like
		x-values at which the observations were taken.

	y: array-like
		y-values for each observation.

	yerr: array-like or None
		Uncertainties on each observed y-value. If a 2D array, assumed to contain the lower and
		upper uncertainties.

	func: callable
		Function to be fitted. Must have the form func(x, params) where params is array-like and
		comprises the function parameters.

	initial: array-like
		Initial guesses for the best-fit parameters (must have the same order as when they are fed
		to the function).

	res: float or array-like
		Desired resolution for each parameter when exploring the parameter space. If a single value,
		uses same resolution for all parameters.

	nsig: int
		The desired significance (in integer multiples of sigma) to be probed in the parameter space.

	Nmax: int
		The maximum number of entries to be allowed in any one given array when performing calculations
		(to conserve memory).
	'''	

	def chisq_model(params, x_, y_, yerr_):
		'''
		Calculates chi-squared for the model function and a given dataset.

		Parameters
		----------
		params: array-like
			List/tuple/array containing the model function parameters.
	
		x_: array-like
			x-values at which the observations were taken.

		y_: array-like
			y-values for each observation.

		yerr_: array-like
			Uncertainties on each observed y-value (assumed symmetric).

		Returns
		----------
		chisq: float
			The chi-squared value for the function.
		'''
		ymodel = func(x_, params)
		chi2 = np.sum(((ymodel - y_)/yerr_) ** 2.)
		return chi2

	#get the number of parameters
	N_param = len(initial)

	#get the dimensions of yerr
	ndim = gen.get_ndim(yerr)
	#symmetrise the uncertainties if asymmetric uncertainties provided
	if ndim == 2:
		yerr = (yerr[0] + yerr[1]) / 2.

	#find the value of Delta Chi^2 corresponding to the desired significance
	dchi2_nsig = chi2_df.loc[N_param][f'{int(nsig)}sig']

	#perform an initial chi-squared minimisation using an optimiser:
	#reformat the chisq_model function so that it is compatible with scipy.optimize.minimize
	nll = lambda *args : chisq_model(*args)
	#run the optimisation to find the best-fit parameters that minimize chi squared
	#popt = opt.minimize(nll, x0=initial, args=(x, y, yerr))['x']
	popt = np.array(initial)
	#set the maximum number of entries along each axis an array with N_params dimensions can have
	#while still being smaller than Nmax
	N_per_dim = int(Nmax ** (1./N_param))

	#factors by which to scale the resolutions if the parameter space ends up being too small
	res_factor = np.full(N_param,1.)
	#convert the resolutions into an array if it is not one already
	res = np.array(res)

	#create an array to which all parameter combinations will be concatenated
	params_tried = np.empty((0, N_param), dtype=float)
	chi2_tried = np.empty(0, dtype=float)

	#define a grid in the parameter space centred on this initial estimate, firstly with far worse
	param_ranges = [
		np.arange(
			popt[i]-(N_per_dim/2)*res[i],
			popt[i]+(N_per_dim/2)*res[i],
			res[i]
			)
		for i in range(N_param)]
	#all unique combinations of the paramters in these ranges
	permut = np.array(list(itertools.product(*param_ranges)))
	#remove any rounding errors
	permut = permut - (permut % res)


	#calculate the model y values for these combinations of parameters
	ymodel = func(x[:,None], permut.T)
	#get rid of any parameter combinations resulting in NaN model values for any bins
	params_keep = ~np.isnan(ymodel).all(axis=0)
	permut = permut.T[:,params_keep].T
	ymodel = ymodel[:,params_keep]
	#calculate chi^2 for each combination
	chi2 = np.sum(((ymodel - y[:,None]) / yerr[:,None]) ** 2., axis=0)

	#if previous iteration has been run, find min chi^2 across all iterations
	chi2_min = chi2.min()

	#calculate Delta Chi^2
	dchi2 = chi2 - chi2_min

	#find where Delta Chi2 < the value corresponding to n sigma 
	dchi2_mask = dchi2 < dchi2_nsig


	return chi2, permut





def chisq_fit(
	x,
	y,
	yerr,
	func,
	params_min,
	params_max,
	params_res,
	Nmax=10000000,
	save_output=False,
	output_name='chi2_results.npz',
	param_names=None,
	evaluate_errs=True):
	'''
	Parameters
	----------
	x: array-like
		Independent variable values at which the model is to be evaluated.

	y: array-like
		Observed values of the dependent variable at x.

	yerr: array-like
		Uncertainties in the observed values of y. Asymmetric uncertainties can be provided in a
		single array/list of the for [yerr_lo, yerr_hi].

	func: callable
		The model function. Should have the form func(x, params) where params is an array-like
		variable containing the function parameters. Function should also be able to take vector
		arguments for params.

	params_min: array-like
		Minimum values for each parameter in the model. Parameter space beyond these bounds will
		not be explored.

	params_max: array-like
		Minimum values for each parameter in the model. Parameter space beyond these bounds will
		not be explored.

	params_res: array-like
		Desired resolution for each parameter.

	Nmax: int
		The maximum number of entries to be allowed in any one given array when performing calculations
		(to conserve memory).

	save_output: bool
		Save the parameter combinations and chi^2 values to a .npz file.

	output_name: str
		Path to the output file if told to make one.

	param_names: list or None
		If not None, a list that contains the names of each parameter for the sake of saving to a file.

	evaluate_errs: bool
		Find the uncertainty on each parameter through marginalisation.

	PLAN:
		- Calculate model values using func(x, params).
		- Create array of all possible parameter combinations within the bounds defined by the arguments.
		- Calculate chi^2 for each parameter combination.
			- If asymmetric uncertainties provided, use lower uncertainty if ymodel < y, and use upper uncertainty if ymodel > y.
	  	- Find parameter combo with min chi^2.
	  	- For parameter uncertainties:
	  		- Convert chi^2 into probability values using P \propto exp(-chi^2 / 2) and normalise.
	  		- At each value for the current parameter, sum probabilities across all possible values of other params.
	  		- Turn this into a CDF.
	  		- Find values of parameter where CDF = 0.16, 0.84.
	'''

	#get the number of parameters
	N_param = len(params_min)
	#set the maximum number of entries along each axis an array with N_params dimensions can have
	#while still being smaller than Nmax
	N_per_dim = int(Nmax ** (1./N_param))

	#get the dimensions of yerr
	ndim = gen.get_ndim(yerr)
	#retrieve upper and lower uncertainties if yerr is 2D
	if ndim == 2:
		yerr_lo, yerr_hi = yerr
	else:
		yerr_lo, yerr_hi = [yerr, yerr]

	#create parameter ranges given the provided minima, maxima and resolutions
	param_ranges = [np.arange(pmin, pmax+pres, pres) for pmin,pmax,pres in zip(params_min, params_max, params_res)]
	#get the total number of permutations
	N_permut = reduce(lambda x,y: x*y, [len(p) for p in param_ranges])
	#use this to figure out how many steps the calculation of chi-squared should be broken into
	N_iter = int(np.ceil(N_permut / Nmax))

	#all unique combinations of the paramters in these ranges
	permut = itertools.product(*param_ranges)

	results_all = []
	for chunk in chunks(permut, Nmax):
		permut_now = np.array(list(chunk))
		#calculate chi^2 for these parameter combinations
		ymodel = func(x[:,None], permut_now.T).T
		chi2 = np.zeros(ymodel.shape)
		#identify all model values that are above/below the observed values
		mask_lo = ymodel < y
		mask_hi = ymodel >= y
		chi2[mask_lo] = ((ymodel - y)/yerr_lo)[mask_lo] ** 2.
		chi2[mask_hi] = ((ymodel - y)/yerr_hi)[mask_hi] ** 2.
		#take the sum across all bins to get the chi^2 values
		chi2 = np.sum(chi2, axis=1)
		results_all.append(np.concatenate([permut_now, chi2[:,None]], axis=1))
	#combine the results from all chunks
	results_all = np.vstack(results_all)

	#find the minimum chi^2 value
	idx_min = np.argmin(results_all[:,-1])
	chi2_min = results_all[idx_min][-1]
	#retrieve the best-fit parameters
	popt = results_all[idx_min][:-1]

	if save_output:
		#set up a dictionary for the results
		d_results = {}
		for i in range(N_param):
			if param_names is not None:
				d_results[param_names[i]] = results_all[:,i]
			else:
				d_results[f'param{i+1}'] = results_all[:,i]
		np.savez_compressed(output_name, **d_results)
		del d_results


	if evaluate_errs:
		#convert chi^2 values to probabilities and normalise
		P = np.exp(-results_all[:,-1] / 2.)
		P /= P.sum()
		#lists for the lower and upper uncertainties on each parameter
		errs_lo, errs_hi = [], []
		#cycle through the parameters
		for i in range(N_param):
			#get the unique parameter values
			param_u = param_ranges[i]
			#sum the marginalised probabilities at each unique value of the current parameter
			P_m = np.array([np.sum(P[np.abs(results_all[:,i] - p) < params_res[i]*1E-10]) for p in param_u])
			#convert this into a CDF
			CDF = np.cumsum(P_m)
			#interpolate the parameter values w.r.t. the CDF
			p_from_cdf = interp1d(CDF, param_u)
			#retrieve the 16th and 84th percentiles
			p_lo, p_hi = p_from_cdf([p16/100., p84/100.])
			#use these to calculate the uncertainties
			errs_lo.append(popt[i] - p_lo)
			errs_hi.append(p_hi - popt[i])
		errs_lo = np.array(errs_lo)
		errs_hi = np.array(errs_hi)
		return popt, chi2_min, errs_lo, errs_hi


	return popt, chi2_min








