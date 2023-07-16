############################################################################################################
# A script for determining the density of sources required to detect an overdensity using the method
# implemented in this pipeline.
############################################################################################################

'''
Plan for the script:
	- load blank field parameters
	- load in number counts data (one radius at a time)
	- generate 10^4 versions of the number counts using uncertainties on bin heights
	- use blank-field schechter parameters to create PDF for randomly drawing sources
	- randomly draw N sources from this PDF, 10^4 times
	- recalculate number counts including new sources for each realisation
	- combine results from all realisations into final bin heights with uncertainties
	- fit to the results
	- calculate ratio of N0 from new fit to blank field, with uncertainties
	- if ratio is <1sigma significant, increase N; if it is >1sigma, take midpoint between N and previous value of N (use N/2 if first iteration)
	- repeat until convergence reached 
'''



