############################################################################################################
# A script for randomly generating datasets using the S2COSMOS catalogue.
############################################################################################################

#import modules/packages
import os, sys
import general as gen
import plotstyle as ps
import numcounts as nc
import stats
import numpy as np
from multiprocessing import Pool, cpu_count
from astropy.table import Table
from astropy.io import fits
from scipy.interpolate import LinearNDInterpolator

#put main script into a function so that multiprocessing doesn't try rerunning the whole thing
def main():
	##################
	#### SETTINGS ####
	##################

	#toggle `switches' for additional functionality
	rand_comp = True		#calculate completeness values for every random flux density value

	settings = [
		rand_comp
		]

	#print the chosen settings to the Terminal
	print(gen.colour_string('CHOSEN SETTINGS:', 'white'))
	settings_print = [
		'Calculate completeness values for every random flux density value: '
		]
	for i in range(len(settings_print)):
		if settings[i]:
			settings_print[i] += 'y'
		else:
			settings_print[i] += 'n'

	nsim = gen.nsim			#number of iterations to use if randomising the flux densities/completeness
	settings_print.append(f'Number of datasets required: {nsim}')
	print(gen.colour_string('\n'.join(settings_print), 'white'))


	#######################################################
	###############    START OF SCRIPT    #################
	#######################################################

	#relevant paths
	PATH_CATS = gen.PATH_CATS
	PATH_DATA = gen.PATH_DATA

	#S2COSMOS catalogue
	S19_cat = gen.S19_cat
	data_submm = Table.read(S19_cat, format='fits')
	#retrieve the deboosted flux densities and uncertainties, and RMS noise
	S850, eS850_lo, eS850_hi, RMS,  *_ = gen.get_relevant_cols_S19(data_submm, main_only=gen.main_only)

	#see if a file exists with the reconstructed S2COSMOS completeness grid (should have been created
	#in previous step of pipeline)
	compgrid_file = PATH_DATA + 'Completeness_at_S850_and_rms.fits'
	if os.path.exists(compgrid_file):
		print(gen.colour_string(f'Loading and interpolating completeness grid...', 'purple'))
		CG_HDU = fits.open(compgrid_file)[0]
		zgrid = CG_HDU.data.T
		hdr = CG_HDU.header
		#get the min, max and step along the S850 and RMS axes
		xmin = hdr['CRVAL1']
		xstep = hdr['CDELT1']
		xmax = xmin + (hdr['NAXIS1'] - 1) * xstep
		ymin = hdr['CRVAL2']
		ystep = hdr['CDELT2']
		ymax = ymin + (hdr['NAXIS2'] - 1) * ystep
		#create a grid in flux density-RMS space
		xgrid, ygrid = np.mgrid[xmin:xmax+xstep:xstep, ymin:ymax+ystep:ystep]
		#interpolate the completeness values from the file w.r.t. S850 and RMS
		points = np.array([xgrid.flatten(), ygrid.flatten()])
		values = zgrid.flatten()
		comp_interp = LinearNDInterpolator(points.T, values)
	#otherwise, print error message
	else:
		gen.error_message(
			sys.argv[0],
			'''Completeness grid does not exist. Run step 1 of pipeline to create it.\n
			Script will proceed but only generating flux density values.
			''')

	#set up empty lists for randomly generated flux densities and completeness values
	S850_rand = np.zeros((nsim,len(S850)))
	comp_rand = np.zeros((nsim,len(S850)))		#only relevant if rand_comp = True
	#number of datasets generated
	N_done = 0

	#create a dictionary containing the complete array of datasets
	dict_rand = {'S850_rand':S850_rand}

	#see if file exists containing randomised flux densities already
	npz_filename = PATH_CATS + 'S2COSMOS_randomised_S850.npz'
	if os.path.exists(npz_filename):
		rand_data = np.load(npz_filename)
		if 'S850_rand' in rand_data.files:
			S850_rand_old = rand_data['S850_rand']
			N_done += len(S850_rand_old)
			S850_rand[:N_done] = S850_rand_old
			#update the dictionary
			dict_rand['S850_rand'] = S850_rand
			if 'comp_rand' in rand_data.files:
				comp_rand_old = rand_data['comp_rand']
			else:
				comp_rand_old = []
			N_comp_done = len(comp_rand_old)
			#if the length of comp_rand is not the same as S850_rand, recalculate the completeness values
			if N_comp_done != N_done:
				if 'comp_interp' in locals():
					print(gen.colour_string(f'Recalculating completeness for existing S850 values...', 'purple'))
					#set up a multiprocessing Pool using all but one CPU
					with Pool(cpu_count()-1) as pool:
						#calculate the completeness for the randomly generated flux densities
						comp_rand[:N_done] = np.array(pool.starmap(comp_interp, [[S850_rand_old[i], RMS] for i in range(N_done)]))
			else:
				comp_rand[:N_done] = comp_rand_old
			dict_rand['comp_rand'] = comp_rand

	#calculate how many more datasets need to be generated to reach the required number
	N_todo = nsim - N_done

	#if required, generate more datasets
	if N_todo > 0:
		print(gen.colour_string(f'Generating new random S850 values...', 'purple'))
		S850_rand[N_done:] = np.array([stats.random_asymmetric_gaussian(S850[i], eS850_lo[i], eS850_hi[i], N_todo) for i in range(len(S850))]).T
		#create a dictionary containing the complete array of datasets
		dict_rand['S850_rand'] = S850_rand

		if 'comp_interp' in locals():
			if rand_comp:
				#calculate completeness for the new flux density datasets
				print(gen.colour_string(f'Calculating completeness for new S850_values...', 'purple'))
				#set up a multiprocessing Pool using all but one CPU
				with Pool(cpu_count()-1) as pool:
					#calculate the completeness for the randomly generated flux densities
					comp_rand[N_done:] = np.array(pool.starmap(comp_interp, [[S850_rand[N_done+i], RMS] for i in range(N_todo)]))
				#add the result to the dictionary
				dict_rand['comp_rand'] = comp_rand

		#save the randomised flux densities (and completenesses if generated) to a compressed numpy archive
		np.savez_compressed(npz_filename, **dict_rand)


if __name__ == '__main__':
	main()



