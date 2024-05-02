############################################################################################################
# A script for plotting histograms of the significance of the overdensity in each environment, along with
# data points coloured by their radio luminosity.
############################################################################################################

#import modules/packages
import os, sys
import general as gen
import plotstyle as ps
import stats
from matplotlib import pyplot as plt
from astropy.table import Table
import numpy as np
import glob
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Rectangle
from scipy.special import erf


#######################################################
###############    START OF SCRIPT    #################
#######################################################

#relevant paths
PATH_CATS = gen.PATH_CATS
PATH_PLOTS = gen.PATH_PLOTS
PATH_SIMS = gen.PATH_SIMS
PATH_PARAMS = PATH_CATS + 'Schechter_params/'
PATH_COUNTS = PATH_CATS + 'Number_counts/'
#directory containing the simulated number counts (relevant only if plot_sims = True)
PATH_SIM_NC = PATH_CATS + 'Significance_tests/'
main_only = True


#catalogue containing delta values for each RQ and HLAGN/MLAGN analogue
data_rq = Table.read(PATH_CATS + 'RAGERS_COSMOS2020_matches_Mstar_z_rq_with_deltas.fits')
data_rl = Table.read(PATH_CATS + 'RAGERS_COSMOS2020_matches_Mstar_z_rl_with_deltas.fits')

data_rq['Lradio_60cm'][data_rq['Lradio_60cm'].mask] = -99.
#retrieve the radio luminosities (500 MHz) of the two samples
L_rq = data_rq['Lradio_60cm']#.filled(-99)
L_rl = data_rl['Lradio_60cm']
#sort the tables by radio luminosity
data_rq.sort('Lradio_60cm')
data_rl.sort('Lradio_60cm')
#calculate the minimum and maximum radio luminosity in across the RQ and HLAGN/MLAGN samples
L_all = np.concatenate([L_rq, L_rl])
#Lmin, Lmax = np.nanmin(L_all), np.nanmax(L_all)
Lmin, Lmax = 23., 27.


plt.style.use(ps.styledict)
'''
#set up a figure with 4 subplots (one for each radius used)
f, ax = plt.subplots(2, 2, figsize=(14.,12.))#, gridspec_kw={'width_ratios':[15,15,1]})#, constrained_layout=True)
#create a larger axis for axis labels
ax_big = f.add_subplot(111, frameon=False)
ax_big.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, which='both')
#label x and y axes
ax_big.set_xlabel(r'$\delta$', labelpad=20.)
ax_big.set_ylabel(r'$\delta / \sigma_{\delta}^{-}$', labelpad=60.)
'''
f = plt.figure(figsize=(14.,12.))
gs = f.add_gridspec(ncols=4, nrows=3, width_ratios=[1, 50, 50, 4], height_ratios=[100, 100, 1])
#add 4 sets of axes for the data
ax1 = f.add_subplot(gs[0,1])
ax2 = f.add_subplot(gs[0,2])
ax3 = f.add_subplot(gs[1,1])
ax4 = f.add_subplot(gs[1,2])
#add subplots without frames to label the x and y axes
ax_xlabel = f.add_subplot(gs[2,1:3], frameon=False)
ax_xlabel.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, which='both')
ax_xlabel.set_xlabel(r'$\delta / \sigma_{\delta}$', fontsize=30., labelpad=-20)
ax_ylabel = f.add_subplot(gs[0:2,0], frameon=False)
ax_ylabel.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, which='both')
ax_ylabel.set_ylabel('Noramlised counts', fontsize=30.)

#colourmap settings
cmap = mpl.cm.viridis
norm = plt.Normalize(Lmin, Lmax)
cmap.set_under(ps.grey)
cmap.set_over(ps.red)

#cycle through the radii used
for ax,r in zip([ax1, ax2, ax3, ax4],[1, 2, 4, 6]):
	#determine which axis to plot on
	#nrow = idx // 2
	#ncol = idx % 2

	#get the delta values and lower uncertainties for the current radius
	delta_rq = data_rq[f'delta_{r}']
	edelta_lo_rq = data_rq[f'edelta_lo_{r}']
	edelta_hi_rq = data_rq[f'edelta_hi_{r}']
	delta_rl = data_rl[f'delta_{r}']
	edelta_lo_rl = data_rl[f'edelta_lo_{r}']
	edelta_hi_rl = data_rl[f'edelta_hi_{r}']
	#mask for identifying underdensities
	ud_mask_rq = delta_rq < 0.
	ud_mask_rl = delta_rl < 0.
	
	#calculate the ratio of delta to the relevant uncertainty
	X_rq = delta_rq / edelta_lo_rq
	X_rl = delta_rl / edelta_lo_rl
	X_rq[ud_mask_rq] = delta_rq[ud_mask_rq] / edelta_hi_rq[ud_mask_rq]
	X_rl[ud_mask_rl] = delta_rl[ud_mask_rl] / edelta_hi_rl[ud_mask_rl]
	'''
	X_rq = edelta_lo_rq
	X_rl = edelta_lo_rl
	X_rq[ud_mask_rq] = edelta_hi_rq[ud_mask_rq]
	X_rl[ud_mask_rl] = edelta_hi_rl[ud_mask_rl]
	'''
	#plot the data on the correct axes
	#ax.scatter(delta_rq, X_rq, c=L_rq, cmap=cmap, marker='o', s=40., edgecolor='k', linewidth=0.5, vmin=Lmin, vmax=Lmax, label='RQ')
	#ax.scatter(delta_rl, X_rl, c=L_rl, cmap=cmap, marker='^', s=60., edgecolor='k', linewidth=0.5, vmin=Lmin, vmax=Lmax, label='HLAGN/MLAGN analogues')
	#ax.scatter(X_rq, delta_rq, c=L_rq, cmap=cmap, marker='o', s=40., edgecolor='k', linewidth=0.5, vmin=Lmin, vmax=Lmax, label='RQ analogues')
	#ax.scatter(X_rl, delta_rl, c=L_rl, cmap=cmap, marker='^', s=60., edgecolor='k', linewidth=0.5, vmin=Lmin, vmax=Lmax, label='HLAGN/MLAGN analogues')

	X_all = np.concatenate([X_rq, X_rl])
	bins = np.linspace(X_all.min(), X_all.max(), 11)
	#bin the data so that the maxmima can be retrieved prior to plotting
	hist_rq, _ = np.histogram(X_rq, bins=bins)
	hist_rl, _ = np.histogram(X_rl, bins=bins)

	#plot the histograms, normalised such that the maximum is 1
	ax.hist(X_rq, bins=bins, color=ps.magenta, histtype='step', density=True, label='RQ_analogues')
	ax.hist(X_rl, bins=bins, color=ps.dark_blue, histtype='step', linestyle=':', density=True, label='HLAGN/MLAGN analogues')



	print(gen.colour_string(f'R = {r} arcmin'))
	
	'''print(gen.colour_string('RQ', 'purple'))
	print(f'f(<-3) = {(X_rq < -3).sum() / len(X_rq)}')
	print(f'f(<-1) = {(X_rq < -1).sum() / len(X_rq)}')
	print(f'f(-1 -> 1) = {((X_rq >= -1) * (X_rq <= 1.)).sum() / len(X_rq)}')
	print(f'f(>1) = {(X_rq > 1).sum() / len(X_rq)}')
	print(f'f(>3) = {(X_rq > 3).sum() / len(X_rq)}')
	print(gen.colour_string('HLAGN/MLAGN', 'purple'))
	print(f'f(<-3) = {(X_rl < -3).sum() / len(X_rl)}')
	print(f'f(<-1) = {(X_rl < -1).sum() / len(X_rl)}')
	print(f'f(-1 -> 1) = {((X_rl >= -1) * (X_rl <= 1.)).sum() / len(X_rl)}')
	print(f'f(>1) = {(X_rl > 1).sum() / len(X_rl)}')
	print(f'f(>3) = {(X_rl > 3).sum() / len(X_rl)}')
	print(gen.colour_string('ALL', 'purple'))
	print(f'f(<-3) = {((X_rl < -3).sum() + (X_rq < -3).sum())/ (len(X_rl)+len(X_rq))}')
	print(f'f(<-1) = {((X_rl < -1).sum() + (X_rq < -1).sum())/ (len(X_rl)+len(X_rq))}')
	print(f'f(-1 -> 1) = {(((X_rl >= -1) * (X_rl <= 1.)).sum() + ((X_rq >= -1) * (X_rq <= 1.)).sum())/ (len(X_rl)+len(X_rq))}')
	print(f'f(>1) = {((X_rl > 1).sum() + (X_rq > 1.).sum())/ (len(X_rl)+len(X_rq))}')
	print(f'f(>3) = {((X_rl > 3).sum() + (X_rq > 3.).sum())/ (len(X_rl)+len(X_rq))}')'''

	'''print(gen.colour_string('RQ', 'purple'))
	print(f'N(<-3) = {(X_rq < -3).sum()}')
	print(f'N(<-1) = {(X_rq < -1).sum()}')
	print(f'N(-1 -> 1) = {((X_rq >= -1) * (X_rq <= 1.)).sum()}')
	print(f'N(>1) = {(X_rq > 1).sum()}')
	print(f'N(>3) = {(X_rq > 3).sum()}')
	print(gen.colour_string('HLAGN/MLAGN', 'purple'))
	print(f'N(<-3) = {(X_rl < -3).sum()}')
	print(f'N(<-1) = {(X_rl < -1).sum()}')
	print(f'N(-1 -> 1) = {((X_rl >= -1) * (X_rl <= 1.)).sum()}')
	print(f'N(>1) = {(X_rl > 1).sum()}')
	print(f'N(>3) = {(X_rl > 3).sum()}')
	print(gen.colour_string('ALL', 'purple'))
	print(f'N(<-3) = {((X_rl < -3).sum() + (X_rq < -3).sum())}')
	print(f'N(<-1) = {((X_rl < -1).sum() + (X_rq < -1).sum())}')
	print(f'N(-1 -> 1) = {(((X_rl >= -1) * (X_rl <= 1.)).sum() + ((X_rq >= -1) * (X_rq <= 1.)).sum())}')
	print(f'N(>1) = {((X_rl > 1).sum() + (X_rq > 1.).sum())}')
	print(f'N(>3) = {((X_rl > 3).sum() + (X_rq > 3.).sum())}')'''
	#add a horizontal line at 1
	xmin, xmax = ax.get_xlim()
	ymin, ymax = ax.get_ylim()
	#ax.axhline(1., c='k', linestyle='--', alpha=0.3)
	#ax.axhline(-1., c='k', linestyle='--', alpha=0.3)
	#ax.plot([xmin, xmax], [xmin, xmax], c='k', alpha=0.3)
	#ax.plot([xmin, xmax], [-xmin, -xmax], c='k', alpha=0.3)
	#ax.axhline(0., c='k', linestyle='--', alpha=0.3)
	#ax.axvline(0., c='k', linestyle='--', alpha=0.3)

	#ax.set_xlim(xmin, xmax)
	#ax.set_ylim(ymin, ymax)

	#add some space to the top of the subplot
	ymin, ymax = ax.set_ylim(ymin, ymax*1.6)
	ymax_hists = ymax * 0.67

	#add lines for the medians
	ax.plot([np.median(X_rq)]*2, [ymin, ymax_hists], c=ps.magenta)
	ax.plot([np.median(X_rl)]*2, [ymin, ymax_hists], c=ps.dark_blue, linestyle=':')
	#1sigma range
	#ax.fill_between(np.percentile(X_rq, q=[stats.p16, stats.p84]), y1=[ymax_hists]*2, color=ps.magenta,  hatch='\\', alpha=0.2)
	#ax.fill_between(np.percentile(X_rl, q=[stats.p16, stats.p84]), y1=[ymax_hists]*2, color=ps.dark_blue,  hatch='/', alpha=0.2)
	ax.fill_between([-1., 1.], y1=[ymax]*2, color='k', alpha=0.1, hatch='/')

	#plot the data points and colour according to radio luminosity; y-coordinates are randomly generated
	dy = ymax - ymin
	Y_rq = np.random.uniform(low=ymax-0.3*dy, high=ymax-0.05*dy, size=len(X_rq))
	Y_rl = np.random.uniform(low=ymax-0.3*dy, high=ymax-0.05*dy, size=len(X_rl))
	ax.scatter(X_rq, Y_rq, c=L_rq, cmap=cmap, marker='o', s=40., linewidth=0.5, vmin=Lmin, vmax=Lmax, alpha=0.8)
	ax.scatter(X_rl, Y_rl, c=L_rl, cmap=cmap, marker='^', s=60., linewidth=0.5, vmin=Lmin, vmax=Lmax, alpha=0.8)


	ax.axhline(ymax_hists, c='k')
	#ax.plot([-1., -1.], [ymin, ymax_hists], linestyle='--', c='k', alpha=0.4)
	#ax.plot([1., 1.], [ymin, ymax_hists], linestyle='--', c='k', alpha=0.4)

	#label the axes with the radius
	#ax.text(0.05, 0.95, r'$R = %i^{\prime}$'%r, va='top', ha='left', transform=ax.transAxes)
	ax.text(0.95, 0.58, r'$R = %i^{\prime}$'%r, va='center', ha='right', transform=ax.transAxes)

	#retrieve the x and y limits again
	xmin, xmax = ax.get_xlim()
	ymin, ymax = ax.get_ylim()
	#plot a Gaussian with width 1 centred at 0
	x_g = np.linspace(xmin, xmax, 1000)
	y_g = stats.gaussian(x_g, 0, 1) / np.sqrt(2. * np.pi)
	ax.plot(x_g, y_g, c='k', alpha=0.2, zorder=0)

	#reset the x and y limits
	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax)

	#fraction of normal distribution above 1 and 3 sigma
	f_g1 = 0.5 * (1 - erf(1/np.sqrt(2)))
	f_g3 = 0.5 * (1 - erf(3/np.sqrt(2)))
	N_exp_rq1 = f_g1 * len(X_rq)
	N_exp_rl1 = f_g1 * len(X_rl)
	N_exp_rq3 = f_g3 * len(X_rq)
	N_exp_rl3 = f_g3 * len(X_rl)
	N_exp_all1 = f_g1 * (len(X_rq) + len(X_rl))
	N_exp_all3 = f_g3 * (len(X_rl) + len(X_rq))

	print(gen.colour_string('RQ', 'purple'))
	print(f'N(<-3 & >G) = {np.round((X_rq < -3).sum() - N_exp_rq3)}')
	print(f'N(<-1 & >G) = {np.round((X_rq < -1).sum() - N_exp_rq1)}')
	print(f'N(>1 & >G) = {np.round((X_rq > 1).sum() - N_exp_rq1)}')
	print(f'N(>3 & G) = {np.round((X_rq > 3).sum() - N_exp_rq3)}')
	print(gen.colour_string('HLAGN/MLAGN', 'purple'))
	print(f'N(<-3 & >G) = {np.round((X_rl < -3).sum() - N_exp_rl3)}')
	print(f'N(<-1 & >G) = {np.round((X_rl < -1).sum() - N_exp_rl1)}')
	print(f'N(>1 & >G) = {np.round((X_rl > 1).sum() - N_exp_rl1)}')
	print(f'N(>3 & >G) = {np.round((X_rl > 3).sum() - N_exp_rl3)}')
	print(gen.colour_string('ALL', 'purple'))
	print(f'N(<-3 & >G) = {np.round(((X_rl < -3).sum() + (X_rq < -3).sum()) - N_exp_all3)}')
	print(f'N(<-1 & >G) = {np.round(((X_rl < -1).sum() + (X_rq < -1).sum()) - N_exp_all1)}')
	print(f'N(>1 & >G) = {np.round(((X_rl > 1).sum() + (X_rq > 1.).sum()) - N_exp_all1)}')
	print(f'N(>3 & >G) = {np.round(((X_rl > 3).sum() + (X_rq > 3.).sum()) - N_exp_all3)}')

'''
legend_elements = [
	Line2D([0], [0], color='w', mec='k', mfc='k', marker='o', ms=7., alpha=0.8, label='RQ'),
	Line2D([0], [0], color='w', mec='k', mfc='k', marker='^', ms=8., alpha=0.8, label='HLAGN/MLAGN')
	]
ax3.legend(handles=legend_elements, loc=6)
'''
#ax3.legend([(h_rq, s_rq), (h_rl, s_rl)], ['RQ', 'MLAGN/HLAGN'], loc=6, handler_map={tuple: HandlerTuple(ndivide=None)})
L1a = Rectangle((0,0), 1, 1, fc='none', ec=ps.magenta)
L1b = Line2D([0], [0], color=ps.grey, marker='o', ms=7., alpha=0.8, linestyle='none')
L2a = Rectangle((0,0), 1, 1, fc='none', ec=ps.dark_blue, linestyle=':')
L2b = Line2D([0], [0], color=ps.grey, marker='^', ms=8., alpha=0.8, linestyle='none')
handles = [(L1a, L1b), (L2a, L2b)]
labels = ['RQ', 'HLAGN/MLAGN']
ax3.legend(handles=handles, labels=labels, loc=6, handler_map={tuple: HandlerTuple(ndivide=None)})

sm = ScalarMappable(norm=norm, cmap=cmap)
cax = f.add_subplot(gs[0:2,3])
#cax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False, which='both')
cbar = f.colorbar(sm, cax=cax, location='right', extend='both')
cbar.set_label(r'$\log_{10}(L_{500~{\rm MHz}}$ / W~Hz$^{-1}$)', fontsize=30.)

#f.text(0.5, -0.05, r'$\delta$', ha='center')
#f.text(-0.05, 0.5, r'$\delta / \sigma_{\delta}^{-}$', va='center', rotation='vertical')

#plt.tight_layout()
plt.savefig(PATH_PLOTS + 'delta_significance_histograms.png', dpi=300, bbox_inches='tight')

