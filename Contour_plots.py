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
nbins = 30

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
f, ax = plt.subplots(3, 2, figsize=(2.*ps.x_size, 2.5*ps.y_size))

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
nc_params_all = [np.load(PATH_PARAMS+f'Differential_{r:.1f}am_{gen.n_gal}{gen.gal_type}.npz')['ALL'] for r in gen.r_search_all]
cc_params_all = [np.load(PATH_PARAMS+f'Cumulative_{r:.1f}am_{gen.n_gal}{gen.gal_type}.npz')['ALL'] for r in gen.r_search_all]
#retrieve the posterior distributions from each MCMC fit
nc_posts_all = [np.load(PATH_POSTS+f'Differential_{r:.1f}am_{gen.n_gal}{gen.gal_type}.npz')['ALL'] for r in gen.r_search_all]
cc_posts_all = [np.load(PATH_POSTS+f'Cumulative_{r:.1f}am_{gen.n_gal}{gen.gal_type}.npz')['ALL'] for r in gen.r_search_all]

#blank field parameters (S2COSMOS)
nc_params_s2c = np.load(PATH_PARAMS+f'Differential_bf.npz')['S2COSMOS']
cc_params_s2c = np.load(PATH_PARAMS+f'Cumulative_bf.npz')['S2COSMOS']
#blank field posteriors (S2COSMOS)
nc_posts_s2c = np.load(PATH_POSTS+f'Differential_bf.npz')['S2COSMOS']
cc_posts_s2c = np.load(PATH_POSTS+f'Cumulative_bf.npz')['S2COSMOS']

#combine the differential and cumulative results into lists
params_all = [nc_params_all, cc_params_all]
posts_all = [nc_posts_all, cc_posts_all]
params_s2c = [nc_params_s2c, cc_params_s2c]
posts_s2c = [nc_posts_s2c, cc_posts_s2c]

#######################
#### CONTOUR PLOTS ####
#######################

for j in range(len(ax[0])):

	if j == 0:
		print(gen.colour_string('Differential number counts', 'blue'))
	else:
		print(gen.colour_string('Cumulative number counts', 'blue'))

	#set up a dictionary for the legend
	by_labels = {}

	#cycle through the axes
	for i in range(len(ax)):
		#corresponding index for the y axis matching the current x axis
		iy = (i + 1) % len(xlabels)

		#label the axes
		ax[i][j].set_xlabel(xlabels[i])
		ax[i][j].set_ylabel(ylabels[i])

		print(gen.colour_string(f'Making contours for {ylabels[i]} vs {xlabels[i]}...', 'purple'))

		print(gen.colour_string(f'S2COSMOS', 'orange'))

		#retrieve the data to bin for this subplot
		X, Y = posts_s2c[j].T[[i,iy]]

		#bin the walker postions in 2D
		z, xbins, ybins = np.histogram2d(X, Y, nbins)
		#get the bin centres in x and y
		xc = (xbins[1:] + xbins[:-1]) / 2.
		yc = (ybins[1:] + ybins[:-1]) / 2.
		#get the extent for plotting the contours
		ext = [min(xc), max(xc), min(yc), max(yc)]
		#normalise the histogram so that it represents the 2D PDF
		#z = gaussian_filter(z, (0.5, 0.5))
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
		ax[i][j].contour(xc, yc, z.T, levels=t_contours, colors=ps.crimson, linestyles='--', extent=None)

		if show_data:
			#retrieve the best-fit Schechter parameters and uncertainties to plot on these axes
			(X_best, eX_best_lo, eX_best_hi), (Y_best, eY_best_lo, eY_best_hi) = params_s2c[j].T[[i,iy]]
			#plot the parameters with uncertainties
			ax[i][j].plot(X_best, Y_best, marker='D', c=ps.crimson, linestyle='none', ms=10.)
			ax[i][j].errorbar(X_best, Y_best, xerr=[[eX_best_lo], [eX_best_hi]], yerr=[[eY_best_lo], [eY_best_hi]], ecolor=ps.crimson, fmt='none')

		#cycle through the different radii
		nclr = 0
		for k in range(N_radii):

			print(gen.colour_string(f'R = {gen.r_search_all[k]:.1f} arcmin', 'orange'))

			#select the colour and marker to use for this dataset
			if gen.r_search_all[k] == r_emph:
				c_now = 'k'
				alpha = 1.
				mkr_now = 'o'
				ms = 11.
				lw = 2.
				label = r'$\mathbf{R = %.1f^{\prime}}$'%gen.r_search_all[k]
			else:
				c_now = cycle_clr[nclr]
				alpha = 0.4
				label = r'$R = %.1f^{\prime}$'%gen.r_search_all[k]
				mkr_now = ps.cycle_mkr[k]
				ms = 8.
				lw = 1.5
				nclr += 1

			#retrieve the data to bin for this subplot
			X, Y = posts_all[j][k].T[[i,iy]]

			#bin the walker postions in 2D
			z, xbins, ybins = np.histogram2d(X, Y, nbins)
			#get the bin centres in x and y
			xc = (xbins[1:] + xbins[:-1]) / 2.
			yc = (ybins[1:] + ybins[:-1]) / 2.
			#get the extent for plotting the contours
			ext = [min(xc), max(xc), min(yc), max(yc)]

			#smooth the histogram with a 2D gaussian filter
			#z = gaussian_filter(z, (0.5, 0.5))
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
			ax[i][j].contour(xc, yc, z.T, levels=t_contours, colors=[c_now], linewidths=lw, extent=None, alpha=alpha)

			if show_data:
				#retrieve the best-fit Schechter parameters and uncertainties to plot on these axes
				(X_best, eX_best_lo, eX_best_hi), (Y_best, eY_best_lo, eY_best_hi) = params_all[j][k].T[[i,iy]]
				#plot the parameters with uncertainties
				ax[i][j].plot(X_best, Y_best, marker=mkr_now, c=c_now, linestyle='none', alpha=alpha, ms=ms)
				ax[i][j].errorbar(X_best, Y_best, xerr=[[eX_best_lo], [eX_best_hi]], yerr=[[eY_best_lo], [eY_best_hi]], ecolor=c_now, fmt='none', alpha=alpha)

			#add entry to the legend
			if i == 0:
				by_labels[label] = Line2D([0], [0], color=c_now, marker=mkr_now, lw=lw, alpha=alpha, ms=ms)

		#add legend entry for S2COSMOS contours
		if i == 0:
			by_labels['S2COSMOS (Simpson+19)'] = Line2D([0], [0], color=ps.crimson, marker='D', ms=10., linestyle='--')
		
		#set the axis limits
		#ax[i][j].set_xlim(min(xbins), max(xbins))
		#ax[i][j].set_ylim(min(ybins), max(ybins))

	ax[0][j].legend([by_labels[k] for k in by_labels.keys()], [k for k in by_labels.keys()])

##############################
#### FORMATTING OF FIGURE ####
##############################

#set axis limits
ax[0][0].set_xlim(1000., 11000.)
ax[0][0].set_ylim(1., 8.)
ax[1][0].set_xlim(1., 5.4)
ax[1][0].set_ylim(-1., 2.7)
ax[2][0].set_xlim(-1., 3.6)
ax[2][0].set_ylim(1000., 11000.)
ax[0][1].set_xlim(1000., 11000.)
ax[0][1].set_ylim(1., 8.)
ax[1][1].set_xlim(1., 5.4)
ax[1][1].set_ylim(-1., 2.7)
ax[2][1].set_xlim(-1., 3.6)
ax[2][1].set_ylim(1000., 11000.)
'''
ax[0][1].set_xlim(2500., 13900.)
ax[0][1].set_ylim(1.1, 6.4)
ax[1][1].set_xlim(1., 5.)
ax[1][1].set_ylim(-1., 1.8)
ax[2][1].set_xlim(-1., 3.)
ax[2][1].set_ylim(1000., 14000.)
'''

f.tight_layout()

figname = PATH_PLOTS + f'Schechter_contours_{gen.n_gal}{gen.gal_type}.png'
if show_data:
	figname = figname[:-4] + '_with_markers.png'
f.savefig(figname, bbox_inches='tight', dpi=300)




