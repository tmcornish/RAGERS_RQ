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
	N_lo = N_hi = int(N/2)
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
	#np.random.shuffle(rand_all)
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
	uncertainties.

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
		Nlo = N * (1. - 1./(9.*N) - 1./(3.*np.sqrt(N))) ** 3.
		Nlo[np.isinf(Nlo)] = 0.
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

