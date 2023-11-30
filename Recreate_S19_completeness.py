############################################################################################################
# A script for trying to recreate the completeness plot from Simpson+19.
############################################################################################################

#import modules/packages
import os
import general as gen
import plotstyle as ps
import numcounts as nc
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from astropy.table import Table
import glob


##################
#### SETTINGS ####
##################

#toggle `switches' for additional functionality
plot_comp = True			#plot the completeness as a function of flux density and RMS
plot_data = False			#plot the S2COSMOS data on top of the completeness grid
plot_extrap = False			#show the extrapolated curves on the plot

settings = [
	plot_comp,
	plot_data,
	plot_extrap
	]

#print the chosen settings to the Terminal
print(gen.colour_string('CHOSEN SETTINGS:', 'white'))
settings_print = [
	'Plot the completeness as a function S850 and RMS: ',
	'Show sources on completeness plot: ',
	'Plot the extrapolated curves on the plot: '
	]
for i in range(len(settings_print)):
	if settings[i]:
		settings_print[i] += 'y'
	else:
		settings_print[i] += 'n'
print(gen.colour_string('\n'.join(settings_print), 'white'))

######################
#### FIGURE SETUP ####
######################

plt.style.use(ps.styledict)

#create the figure (completeness plot)
f, ax = plt.subplots(1, 1)
#label the axes
ax.set_xlabel(r'$S_{850~\mu{\rm m}}$ (mJy)')
ax.set_ylabel(r'$\sigma_{\rm inst}$ (mJy)')


#######################################################
###############    START OF SCRIPT    #################
#######################################################

#relevant paths
PATH_RAGERS = gen.PATH_RAGERS
PATH_CATS = gen.PATH_CATS
PATH_DATA = gen.PATH_DATA
PATH_PLOTS = gen.PATH_PLOTS

#S2COSMOS catalogue
S19_cat = gen.S19_cat
data_submm = Table.read(S19_cat, format='fits')
#retrieve the deboosted flux densities and uncertainties, and RMS noise
S850, eS850_lo, eS850_hi, RMS,  *_ = gen.get_relevant_cols_S19(data_submm, main_only=False)


##################################
#### COMPLETENESS CORRECTIONS ####
##################################

print_str = 'Generating'
if plot_comp:
	print_str += ' and plotting'
print_str += ' completeness grid...'

print(gen.colour_string(print_str, 'purple'))

#list of files contaning the flux densities and RMS values corresponding to a given completeness
comp_files = sorted(glob.glob(PATH_CATS + 'Simpson+19_completeness_curves/*'))
#corresponding completenesses
defined_comps = [0.1, 0.3, 0.5, 0.7, 0.9]
#min, max and step of the axes in the completeness grid: x (flux density) and y (RMS)
xparams = [-15., 30., 0.01]
yparams = [0.45, 3.05, 0.01]
#run the function that reconstructs the completeness grid from Simpson+19
comp_interp, zgrid = nc.recreate_S19_comp_grid(comp_files, defined_comps, xparams, yparams, plot_grid=plot_comp, other_axes=ax, plot_extrap=plot_extrap)
#save the completeness grid to a FITS file
compgrid_file = PATH_DATA + 'Completeness_at_S850_and_rms.fits'
gen.array_to_fits(zgrid.T, compgrid_file, CRPIX=[1,1], CRVAL=[xparams[0],yparams[0]], CDELT=[xparams[2],yparams[2]])

print(gen.colour_string('Calculating completeness for catalogue S850 values...', 'purple'))
#calculate the completeness for the catalogue flux densities and add them to the catalogue
comp_sc2 = comp_interp(S850, RMS)
data_submm['Completeness'] = comp_sc2
data_submm.write(S19_cat, overwrite=True)

if plot_comp:
	#if told to plot the data points as well, add them
	if plot_data:
		#plot the submm sources
		ax.plot(S850, RMS, marker='.', c='k', linestyle='none', alpha=0.3)
		ax.errorbar(S850, RMS, xerr=(eS850_lo, eS850_hi), fmt='none', ecolor='k', alpha=0.1)
		#suffix to add to the figure filename
		suffix_comp = '_with_data'
	else:
		suffix_comp = ''

	print(gen.colour_string('Formatting and saving figure...', 'purple'))
	#set the axes limits (completeness grid)
	xmin, xmax = ax.set_xlim(0., 13.)
	ymin, ymax = ax.set_ylim(yparams[0], yparams[1])
	f.tight_layout()
	f.savefig(PATH_PLOTS + f'Simpson+19_completeness_extrapolated{suffix_comp}.png', dpi=300)

print(gen.colour_string('Done!', 'purple'))










