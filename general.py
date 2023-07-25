############################################################################################################
# Module containing general functions and variables to be used in the analysis of the environments of 
# radio-quiet massive galaxies for RAGERS.
############################################################################################################

import colorsys
import matplotlib as mpl
from astropy.io import fits
import scipy.optimize as opt
import numpy as np
import platform

####################################
#### RELEVANT PATHS & VARIABLES ####
####################################

#get the platform on which this pipeline is being run
pf = platform.platform().split('-')[0]

#relevant directories
if pf == 'macOS':
	PATH_RAGERS = '/Users/thomascornish/Desktop/PhD/RAGERS/'
	PATH_SIMS = PATH_RAGERS + 'Simulated_data/'
elif pf == 'Linux':
	PATH_RAGERS = '/home/cornisht/RAGERS/'
	PATH_SIMS = '/media/cornisht/DATA/RAGERS/Simulated_data/'
else:
	PATH_RAGERS = './'
	PATH_SIMS = PATH_RAGERS + 'Simulated_data/'
PATH_SCRIPTS = PATH_RAGERS + 'Scripts/Analysis_repo/'
PATH_CATS = PATH_RAGERS + 'Catalogues/'
PATH_PLOTS = PATH_RAGERS + 'Plots/'
PATH_TABLES = PATH_RAGERS + 'Tables/'
PATH_DATA = PATH_RAGERS + 'Data/'


#other settings
r_search = 6.	#default search radius to use when finding submm counterparts
#r_search_all = [0.5, 1.5, 2.5, 3., 3.5, 5., 7., 8.]		#list of different radii to try
r_search_all = [1., 2., 4., 6.]		#list of different radii to try
bin_edges = np.array([2., 3., 5., 7., 9., 12., 15., 18., 22.])
'''
bin_edges_all = [
	np.array([2., 3., 5., 8., 12., 17., 22.]),
	np.array([2., 3., 4.5, 6.5, 9., 13., 17.5, 22.]),
	np.array([2., 3., 4., 5., 6.5, 8., 9.5, 11., 14., 18., 22.]),
	np.array([2., 3., 3.6, 4.2, 4.9, 5.7, 6.6, 7.7, 9., 10.4, 12.1, 14.1, 16.3, 19., 22. ])
	]
'''
gal_type = 'rq'	#type of counterpart to be analysed (rq or rl)
n_gal = 1		#minimum number of counterparts required per RL galaxy
sens_lim = 1.3	#SCUBA-2 sensitivity limit (mJy) for searching for submm companions 
nsim = 10000	#number of simulated datasets to generate when constructing number counts
Smin = 3.		#minimum flux density below which bins will be excluded for Schechter fits

#blank-field data
S19_cat = PATH_CATS + 'Simpson+19_S2COSMOS_source_cat.fits' # alternatively 'S2COSMOS_sourcecat850_Simpson18.fits'
main_only = True	#whether or not to only use the MAIN S2COSMOS catalogue for blank-field calculations
S19_results_file = PATH_CATS + 'Simpson+19_number_counts_tab.txt'	#number counts from Simpson+19
if main_only:
	s2c_key = 'MAIN'
	A_s2c = 1.6		#sq. deg
else:
	s2c_key = 'ALL'
	A_s2c = 2.6		#sq. deg
#bin edges used in Simpson+19
S19_bin_edges = np.array([2., 2.3, 2.7, 3.1, 3.6, 4.2, 4.9, 5.7, 6.6, 7.7, 9., 10.4, 12.1, 14.1, 16.3, 19., 22. ])


###################
#### FUNCTIONS ####
###################

def colour_string(s, c='red'):
	'''
	Reformats a string so that it can be printed to the Terminal in colour (against a black background).
		s: The string to be printed in coloiur.
		c: The desired colour (must be one of the seven available choices; see below).
	'''

	#list of possible colour symbols
	colours = ['red', 'green', 'orange', 'blue', 'purple', 'cyan', 'white']
	#corresponding codes
	codes = ['0;31;40', '0;32;40', '0;33;40', '0;34;40', '0;35;40', '0;36;40', '0;37;40']
	#identify the code corresponding to the colour selected in the argument
	try:
		code_sel = codes[colours.index(c)]
		#use the relevant code to ensure the string is printed to the terminal in the chosen colour
		s_new = '\x1b[%sm%s\x1b[0m'%(code_sel, s)
		#return the new string
		return s_new
	except ValueError:
		#if the use did not select an available colour, print an explanation and return the original string
		print('colour_string: Selected colour not available. Available colours are:\nred\ngreen\norange\nblue\npurple\nolive\nwhite\n')
		return s


def string_important(s):
	'''
	Prints the provided string, using large numbers of '#'s to make it easy to spot when running the code.
		s: String to print.
	'''
	N_str = len(s)
	N_pad = 8
	N_total = N_str + 2 * (N_pad + 1) 
	pad_newline = '#' * N_total
	pad_textline = '#' * N_pad
	textline = ' '.join([pad_textline, s, pad_textline])	#line containing the important text with padding
	return '\n'.join([pad_newline, textline, pad_newline])


def scale_RGB_colour(rgb, scale_l=1., scale_s=1.):
	'''
	Takes any RGB code and scales its 'lightness' and saturation (according to the hls colour 
	model) by factors of scale_l and scale_s.
		hex: The RGB colour specifications.
		scale_l: The factor by which to scale the lightness.
		scale_s: The factor by which to scale the saturation.
	'''
	#convert the rgb to hls (hue, lightness, saturation)
	h, l, s = colorsys.rgb_to_hls(*rgb)
	#scale the lightness ad saturation and ensure the results are between 0 and 1
	l_new = max(0, min(1, l * scale_l))
	s_new = max(0, min(1, s * scale_s))
	#convert back to rgb and return the result
	return colorsys.hls_to_rgb(h, l_new, s_new)


def scale_HEX_colour(c_hex, scale_l=1., scale_s=1.):
	'''
	Takes any colour HEX code and scales its 'lightness' and saturation (according to the hls colour 
	model) by factors of scale_l and scale_s.
		c_hex: The HEX colour code.
		scale_l: The factor by which to scale the lightness.
		scale_s: The factor by which to scale the saturation.
	'''
	#convert the colour HEX code to RGB format
	rgb = mpl.colors.ColorConverter.to_rgb(c_hex)
	#convert back to rgb and return the result
	return scale_RGB_colour(rgb, scale_l, scale_s)


def make_latex_table(data, titles, filename='table.tex', full_pagewidth=False, caption='', include_footnotes=False, footnotes=[], label='tab:tab', alignment='c', padding='0em'):
	'''
	Takes data and formats it into a LaTeX table, saving it to a file for importing directly into a .tex file.
		data: List or array where each entry contains data for one row of the table (provide all as strings).
		titles: List of names for each column in the table.
		filename: Filename to be given to the table file.
		full_pagewidth: (Bool) Whether or not this table should span the full page width. 
		caption: Caption for the table.
		include_footnotes: (Bool) Whether or not the table will have footnotes.
		footnotes: List of footnotes.
		label: Label to be given to the table in the main .tex file.
		alignment: Horizontal alignment of each entry (assumes same alignment across the table).
		padding: Space to leave between data lines in the table.
	'''
	#begin constructing a list of lines to write to the file
	if full_pagewidth:
		lines = [r'\begin{table*}']
	else:
		lines = [r'\begin{table}']
	#add more table pramble, but idnent for ease of interpretation
	add_lines = [r'\centering', r'\caption{%s}'%caption, r'\label{%s}'%label]
	add_lines = ['\t' + s for s in add_lines]
	lines.extend(add_lines)
	#if footnotes are to be included, this needs to be a 'threeparttable'
	if include_footnotes:
		lines.append('\t' + r'\begin{threeparttable}')
	#begin tabular environment
	align_all = ''.join([alignment]*len(titles))
	lines.append('\t' + r'\begin{tabular}{%s}'%align_all)

	#column titles
	title_lines = [r'\hline']
	title_lines.append(r' & '.join(titles) + r'\\')
	title_lines.append(r'\hline')
	title_lines.append(r'\hline')
	title_lines = ['\t\t' + s for s in title_lines]
	#append these to the list of lines
	lines.extend(title_lines)

	#data
	data_lines = ['\t\t' + r' & '.join(l) + r'\\' + '\n\t\t'+ r'\vspace{%s}'%padding for l in data]
	lines.extend(data_lines)
	lines.append('\t\t' + r'\hline')

	#end tabular environment
	lines.append('\t' + r'\end{tabular}')

	#if footnotes included, begin tablenotes environment
	if include_footnotes:
		lines.append('\t' + r'\begin{tablenotes}')
		footnote_lines = ['\t\t' + r'\item[%i] %s'%(i+1,footnotes[i]) for i in range(len(footnotes))]
		lines.extend(footnote_lines)
		#end tablenotes environment
		lines.append('\t' + r'\end{tablenotes}')
		#end the threeparttable environment as well
		lines.append('\t' + r'\end{threeparttable}')

	#end the table environment
	if full_pagewidth:
		lines.append(r'\end{table*}')
	else:
		lines.append(r'\end{table}')

	#open the file and begin write the table to it
	with open(filename, 'w') as f:
		f.write('\n'.join(lines))


def round_sigfigs(num, sf):
	'''
	Rounds a number to a given number of significant figures
		num: The number to be rounded
		sf: The number of significant figures to which num will be rounded
	'''
	if num != 0.:
		i = -int(np.floor(np.log10(abs(num))) - (sf - 1))		#the number of decimal places to round to
		num_rounded = round(num, i)
	else:
		num_rounded = 0.
	return num_rounded


def array_to_fits(data, filename, CTYPE1='RA', CTYPE2='DEC', CRPIX=[1,1], CRVAL=[0,0], CDELT=[1,1]):
	'''
	Takes an array and details describing a coordinate system and creates a FITS file.
		data: The array containing the data.
		filename: Output file name.
		CTYPE1: Name of the variable along axis 1.
		CTYPE2: Name of the variable along axis 2.
		CRPIX: Reference pixels in [X,Y].
		CRVAL: Reference pixel values in [X,Y].
		CDELT: Pixel scales in [X,Y].
	'''
	#create a PrimaryHDU from the chi^2 grid
	hdu = fits.PrimaryHDU(data)
	#update the relevant parameters in the header
	hdu.header['CTYPE1'] = CTYPE1
	hdu.header['CTYPE2'] = CTYPE2
	hdu.header['CRPIX1'] = CRPIX[0]
	hdu.header['CRPIX2'] = CRPIX[1]
	hdu.header['CRVAL1'] = CRVAL[0]
	hdu.header['CRVAL2'] = CRVAL[1]
	hdu.header['CDELT1'] = CDELT[0]
	hdu.header['CDELT2'] = CDELT[1]
	#create an HDUList
	hdul = fits.HDUList([hdu])
	#write this to the file
	hdul.writeto(filename, overwrite=True)


def error_message(module, message):
	'''
	Prints a nicely formatted error message. For use within other modules as a means of identifying
	issues. 
		module: The name of the module being debugged.
		message: The body of the error message to print.
	'''
	err_str = [
		colour_string(f'{module}\n', 'cyan'),
		colour_string('Error: '),
		colour_string(message, 'white')]
	print(''.join(err_str))


def get_ndim(x):
	'''
	Takes a variable and determines its the number of dimensions, e.g. 0 for single number (scalar), 
	1 for a list or 1d array, 2 for a 2d array, etc. Note: only really effective with int, float, 
	str, list, tuple, and other array-like types.
		x: The variable for which the rank is to be determined.
	'''
	xtype = type(x)

	#if the variable is a single integer, float or string, set ndim = 0
	if xtype in [int, float, str, np.float32, np.float64, np.int32, np.int64]:
		ndim = 0
	#otherwise, attempt to convert to an array then find the number of dimensions
	else:
		x_array = np.array(x)
		ndim = len(x_array.shape)
		#if the length is still 0, return None instead and print an error message
		if ndim == 0:
			ndim = None
			error_message('general.get_ndim', f'could not find ndim for data type {xtype.__name__}')

	return ndim



def area_of_intersection(r_a, r_b, d):
	'''
	Calculates the intersecting area between two overlappig circles.
		r_a: Radius of the first circle.
		r_b: Radius of the second circle.
		d: Distance between the centres of the two circles.
	'''
	#see which circle has the smaller radius (r1 = bigger circle, r2 = smaller circle)
	if r_a >= r_b:
		r1 = r_a
		r2 = r_b
	else:
		r1 = r_b
		r2 = r_a
	#check if the distance is greater than the sum of the radii - if so then there is no intersection
	if d > (r1 + r2):
		return 0.
	#if the distance is less than the difference between the two radii, the area is equal to that of the smaller circle
	if d <= (r1 - r2):
		return np.pi * (r2 ** 2.)

	#otherwise, the area of intersection is non-trivial and must be calculated differently; see
	#https://diego.assencio.com/?index=8d6ca3d82151bad815f78addf9b5c1c6 for derivation and notation
	d1 = ((r1 ** 2.) - (r2 ** 2.) + (d ** 2.)) / (2. * d)
	d2 = d - d1
	A1 = (r1 ** 2.) * np.arccos(d1 / r1) - d1 * np.sqrt((r1 ** 2.) - (d1 ** 2.))
	A2 = (r2 ** 2.) * np.arccos(d2 / r2) - d2 * np.sqrt((r2 ** 2.) - (d2 ** 2.))
	return A1 + A2


def get_relevant_cols_S19(data, main_only=False):
	'''
	Retrieves the columns containing the positions, deboosted S850 flux densities and uncertainties
	of the submillimetre sources in the S2COSMOS catalogue.

	Parameters
	----------
	data: astropy.table.Table
		Table form of the S2COSMOS catalogue.

	main_only: bool
		Retrieves data for only the 'MAIN' sample if True.

	Returns
	-------
	S850, eS850_lo, eS850_hi, RMS, RA, DEC: Columns
		The relevant columns from the catalogue.
	'''

	#two possible catalogues exist, therefore one of two possibilities for some column names;
	#make a dictionary for each set of column names
	cols1 = {
		'S850' : 'S850-deb',
		'eS850_lo' : 'e_S850-deb',
		'eS850_hi' : 'E_S850-deb',
		'RMS' : 'e_S850-obs',
		'Sample' : 'Sample',
		'RA' : 'RA_deg',
		'DEC' : 'DEC_deg'
		}
	cols2 = {
		'S850' : 'S_deboost',
		'eS850_lo' : 'S_deboost_errlo',
		'eS850_hi' : 'S_deboost_errhi',
		'RMS' : 'RMS',
		'Sample' : 'CATTYPE',
		'RA' : 'RA_deg',
		'DEC' : 'DEC_deg'
		}

	#see which column name for flux density is in the data
	if cols1['S850'] in data.colnames:
		cols = cols1
	else:
		cols = cols2

	#trim catalogue if told to use only the MAIN sample
	if main_only:
		data = data[data[cols['Sample']] == 'MAIN']

	#retrieve the relevant columns
	get_keys = ['S850', 'eS850_lo', 'eS850_hi', 'RMS', 'RA', 'DEC']
	S850, eS850_lo, eS850_hi, RMS, RA, DEC = [data[cols[c]] for c in get_keys]

	#see if a completeness column also exists
	if 'Completeness' in data.colnames:
		#return it along with everything else
		comp = data['Completeness']
		return S850, eS850_lo, eS850_hi, RMS, RA, DEC, comp		
	else:
		return S850, eS850_lo, eS850_hi, RMS, RA, DEC


