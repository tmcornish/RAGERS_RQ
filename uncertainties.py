############################################################################################################
# Module containing functions for the analysis and propagation of uncertainties.
###########################################################################################################


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