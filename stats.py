############################################################################################################
# Module containing functions for the analysis and propagation of uncertainties.
###########################################################################################################

import numpy as np
from scipy.stats import truncnorm
import scipy.special as special
from scipy import optimize as opt


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


def random_asymmetric_gaussian(mu, sigma_lo, sigma_hi, N):
	'''
	Randomly draws values from an asymmetric 'normal' distribution. To be used e.g. when propagating 
	asymmetric uncertainties. NOTE: Only works for single-value arguments.
		mu: Mean(s) of the distribution(s).
		sigma_lo: Standard deviation(s) of the distribution(s) below the mean(s).
		sigma_hi: Standard deviation(s) of the distribution(s) above the mean(s).
		N: Number of random values to be drawn from the distribution(s).
	'''
	#choose the number of values to draw from each side of the distribution based on the ratio of the uncertainties
	N_hi = int(np.nan_to_num(np.ceil(N / (1. + sigma_lo / sigma_hi)), nan=0.))
	N_lo = int(np.floor(N - N_hi))
	#randomly draw values from truncated normal distributions, corresponding to the lower and upper halves of a normal distribution
	rand_lo = truncnorm.rvs(-np.inf, 0., size=N_lo)
	rand_hi = truncnorm.rvs(0., np.inf, size=N_hi)
	#scale and shift each distribution according to the standard deviations and means
	rand_lo = rand_lo * sigma_lo + mu
	rand_hi = rand_hi * sigma_hi + mu
	#concatenate the two randomly drawn samples
	rand_all = np.concatenate([rand_lo, rand_hi])
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


def chisq_minimise(x, y, yerr, func, initial):
	'''
	Takes a function and performs chi-squared minimisation to determine the best-fit parameters
	for a set of observations.

	Parameters
	----------
	x: array-like
		x-values at which the observations were taken.

	y: array-like
		y-values for each observation.

	yerr: array-like
		Uncertainties on each observed y-value (assumed symmetric).

	func: callable
		Function to be fitted. Must have the form func(x, params) where params is array-like and
		comprises the function paramters.

	initial: array-like
		Initial guesses for the best-fit parameters (must have the same order as when they are fed
		to the function).

	Returns
	----------
	popt: array
		The best-fit parameters.

	chi2min: float
		The chi-squared value for the best fit.

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

	#reformat the chisq_model function so that it is compatible with scipy.optimize.minimize
	nll = lambda *args : chisq_model(*args)
	#run the optimisation to find the best-fit parameters that minimize chi squared
	res = opt.minimize(nll, x0=initial, args=(x, y, yerr))
	#retrieve the best-fit parameters and minimum chi squared value
	popt = res.x
	chi2min = res.fun
	return popt, chi2min



