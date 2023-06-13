############################################################################################################
# A script for making contour plots from the Schechter fits performed in a previous step of the
# pipeline.
############################################################################################################

#import modules/packages
import os, sys
import general as gen
import plotstyle as ps
import numcounts as nc
import stats
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

##################
#### SETTINGS ####
##################

#toggle switches for additional functionality
show_data = True				#show the best-fit parameters as data points with error bars
make_corners = False			#make corner plots for each dataset

#formatting for figures
plt.style.use(ps.styledict)

#bins for the 2D histograms
nbins = 50

#select a radius for which the results will be emphasised on the plot
r_emph = 4.

#######################################################
###############    START OF SCRIPT    #################
#######################################################

#relevant paths
PATH_CATS = gen.PATH_CATS
PATH_PLOTS = gen.PATH_PLOTS
PATH_SIMS = gen.PATH_SIMS
PATH_PARAMS = PATH_CATS + 'Schechter_params/'
PATH_COUNTS = PATH_CATS + 'Number_counts/'
PATH_POSTS = PATH_SIMS + 'Schechter_posteriors/'

#retrieve the number of radii used to make the number counts
N_radii = len(gen.r_search_all)
#contour levels to mark intervals of sigma
sig_levels = [1.]
clevels = np.array([np.diff(stats.percentiles_nsig(i))[0] for i in sig_levels])[::-1] / 100.

######################
#### FIGURE SETUP ####
######################

#create a figure with 3 panels
f, ax = plt.subplots(1, 3, figsize=(3.*ps.x_size, ps.y_size))

#x and y labels for each subplot
xlabels = [r'$N_{0}$', r'$S_{0}$', r'$\gamma$']
ylabels = [xlabels[(i+1)%len(xlabels)] for i in range(len(xlabels))]

#use colours from middle section of the GnBu colourmap
interval = np.arange(0.1, 1.001, 0.001)
colors = plt.cm.GnBu(interval)
cmap = LinearSegmentedColormap.from_list('name', colors)
#choose N colours from this map, where N is the number of radii used minus any that are to be emphasised
N_clr = len([r for r in gen.r_search_all if r != r_emph])
cycle_clr = [gen.scale_RGB_colour(cmap((j+1.)/N_radii)[:-1], scale_l=0.7) for j in range(N_clr)]

###############################
#### LOADING RELEVANT DATA ####
###############################

#retrieve the best-fit differential and cumulative parameters for each radius
nc_params_all = [np.load(PATH_PARAMS+f'Differential_{r:.1f}am.npz')['ALL'] for r in gen.r_search_all]
cc_params_all = [np.load(PATH_PARAMS+f'Cumulative_{r:.1f}am.npz')['ALL'] for r in gen.r_search_all]
#retrieve the posterior distributions from each MCMC fit
nc_posts_all = [np.load(PATH_POSTS+f'Differential_{r:.1f}am.npz')['ALL'] for r in gen.r_search_all]
cc_posts_all = [np.load(PATH_POSTS+f'Cumulative_{r:.1f}am.npz')['ALL'] for r in gen.r_search_all]

#blank field parameters (S2COSMOS)
if gen.main_only:
	key = 'MAIN'
else:
	key = 'ALL'
nc_params_s2c = np.load(PATH_PARAMS+f'Differential_S2COSMOS.npz')[key]
Cc_params_s2c = np.load(PATH_PARAMS+f'Cumulative_S2COSMOS.npz')[key]
#blank field posteriors (S2COSMOS)
nc_posts_s2c = np.load(PATH_POSTS+f'Differential_S2COSMOS.npz')[key]
Cc_posts_s2c = np.load(PATH_POSTS+f'Cumulative_S2COSMOS.npz')[key]


#######################
#### CONTOUR PLOTS ####
#######################

#set up a dictionary for the legend
by_labels = {}

#cycle through the axes
for i in range(len(ax)):
	#corresponding index for the y axis matching the current x axis
	iy = (i + 1) % len(xlabels)

	#label the axes
	ax[i].set_xlabel(xlabels[i])
	ax[i].set_ylabel(ylabels[i])

	print(gen.colour_string(f'Making contours for {ylabels[i]} vs {xlabels[i]}...', 'purple'))

	print(gen.colour_string(f'S2COSMOS', 'orange'))

	#retrieve the data to bin for this subplot
	X, Y = nc_posts_s2c.T[[i,iy]]

	#bin the walker postions in 2D
	z, xbins, ybins = np.histogram2d(X, Y, nbins)
	#get the bin centres in x and y
	xc = (xbins[1:] + xbins[:-1]) / 2.
	yc = (ybins[1:] + ybins[:-1]) / 2.
	#get the extent for plotting the contours
	ext = [min(xc), max(xc), min(yc), max(yc)]
	#normalise the histogram so that it represents the 2D PDF
	z /= z.sum()

	#create an array of steps at which the PDF will be integrated
	t = np.linspace(0, z.max(), 1000)
	#integrate the PDF
	integral = ((z >= t[:, None, None]) * z).sum(axis=(1,2))
	#interpolate t with respect to the integral
	f_int = interp1d(integral, t)
	#find t at the value of each contour
	t_contours = f_int(clevels)
	#plot the contours
	ax[i].contour(xc, yc, z.T, levels=t_contours, colors=ps.crimson, linestyles='--', extent=None)

	if show_data:
		#retrieve the best-fit Schechter parameters and uncertainties to plot on these axes
		(X_best, eX_best_lo, eX_best_hi), (Y_best, eY_best_lo, eY_best_hi) = nc_params_s2c.T[[i,iy]]
		#plot the parameters with uncertainties
		ax[i].plot(X_best, Y_best, marker='D', c=ps.crimson, linestyle='none', ms=10.)
		ax[i].errorbar(X_best, Y_best, xerr=[[eX_best_lo], [eX_best_hi]], yerr=[[eY_best_lo], [eY_best_hi]], ecolor=ps.crimson, fmt='none')

	#cycle through the different radii
	nclr = 0
	for j in range(N_radii):

		print(gen.colour_string(f'R = {gen.r_search_all[j]:.1f} arcmin', 'orange'))

		#select the colour and marker to use for this dataset
		if gen.r_search_all[j] == r_emph:
			c_now = 'k'
			alpha = 1.
			mkr_now = 'o'
			ms = 11.
			label = r'$\mathbf{R = %.1f^{\prime}}$'%gen.r_search_all[j]
		else:
			c_now = cycle_clr[nclr]
			alpha = 0.4
			label = r'$R = %.1f^{\prime}$'%gen.r_search_all[j]
			mkr_now = ps.cycle_mkr[j]
			ms = 8.
			nclr += 1

		#retrieve the data to bin for this subplot
		X, Y = nc_posts_all[j].T[[i,iy]]

		#bin the walker postions in 2D
		z, xbins, ybins = np.histogram2d(X, Y, nbins)
		#get the bin centres in x and y
		xc = (xbins[1:] + xbins[:-1]) / 2.
		yc = (ybins[1:] + ybins[:-1]) / 2.
		#get the extent for plotting the contours
		ext = [min(xc), max(xc), min(yc), max(yc)]

		#smooth the histogram with a 2D gaussian filter
		#z = gaussian_filter(z, (0.2, 0.2))
		#normalise the histogram so that it represents the 2D PDF
		z /= z.sum()

		#create an array of steps at which the PDF will be integrated
		t = np.linspace(0, z.max(), 1000)
		#integrate the PDF
		integral = ((z >= t[:, None, None]) * z).sum(axis=(1,2))
		#interpolate t with respect to the integral
		f_int = interp1d(integral, t)
		#find t at the value of each contour
		t_contours = f_int(clevels)
		#plot the contours
		ax[i].contour(xc, yc, z.T, levels=t_contours, colors=[c_now], extent=None, alpha=alpha)

		if show_data:
			#retrieve the best-fit Schechter parameters and uncertainties to plot on these axes
			(X_best, eX_best_lo, eX_best_hi), (Y_best, eY_best_lo, eY_best_hi) = nc_params_all[j].T[[i,iy]]
			#plot the parameters with uncertainties
			ax[i].plot(X_best, Y_best, marker=mkr_now, c=c_now, linestyle='none', alpha=alpha, ms=ms)
			ax[i].errorbar(X_best, Y_best, xerr=[[eX_best_lo], [eX_best_hi]], yerr=[[eY_best_lo], [eY_best_hi]], ecolor=c_now, fmt='none', alpha=alpha)

		#add entry to the legend
		if i == 0:
			by_labels[label] = Line2D([0], [0], color=c_now, marker=mkr_now, alpha=alpha, ms=ms)

	#add legend entry for S2COSMOS contours
	if i == 0:
		by_labels['S2COSMOS (Simpson+19)'] = Line2D([0], [0], color=ps.crimson, marker='D', ms=10., linestyle='--')
	
	#set the axis limits
	#ax[i].set_xlim(min(xbins), max(xbins))
	#ax[i].set_ylim(min(ybins), max(ybins))

ax[0].legend([by_labels[k] for k in by_labels.keys()], [k for k in by_labels.keys()])

##############################
#### FORMATTING OF FIGURE ####
##############################

#set axis limits
ax[0].set_xlim(1000., 11000.)
ax[0].set_ylim(1., 8.)
ax[1].set_xlim(1., 6.)
ax[1].set_ylim(-1., 3.)
ax[2].set_xlim(-1., 4.)
ax[2].set_ylim(1000., 11000.)

f.tight_layout()

figname = PATH_PLOTS + f'Differential_schechter_contours.png'
if show_data:
	figname = figname[:-4] + '_with_markers.png'
f.savefig(figname, bbox_inches='tight', dpi=300)




