############################################################################################################
# Module containing general functions and variables to be used in the analysis of the environments of 
# radio-quiet massive galaxies for RAGERS.
############################################################################################################

import astrometry as astrom
import colorsys
import matplotlib as mpl
from astropy.io import fits
import scipy.optimize as opt
import numpy as np

####################################
#### RELEVANT PATHS & VARIABLES ####
####################################

#relevant directories
PATH_RAGERS = '/Users/thomascornish/Desktop/PhD/RAGERS/'
#PATH_RAGERS = '/home/cornisht/RAGERS/'
PATH_SCRIPTS = PATH_RAGERS + '/Scripts/Analysis_repo/'
PATH_CATS = PATH_RAGERS + '/Catalogues/'
PATH_PLOTS = PATH_RAGERS + '/Plots/'
PATH_DATA = PATH_RAGERS + '/Data/'

#search radius to use when finding submm counterparts
r_search = 6.

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



def table_to_DS9_regions(T, RAcol, DECcol, convert_to_sexagesimal=True, output_name='targets.reg', labels=False, labelcol='ID', color='green', radius='1"', dashlist='8 3', width='1', font='helvetica 10 normal roman', select='1', highlite='1', dash='0', fixed='0', edit='1', move='1', delete='1', include='1', source='1', coords='fk5'):
	'''
	Takes a table of information about sources and converts the RAs and DECs into a generic file that can be used
	as input by SAOImage DS9 (NOTE: this should not be used if the user wants specific settings for each 
	region that is drawn by DS9; it is purely for drawing generic, uniform regions onto an image).
		T: The table of data.
		RAcol, DECcol: The column names containing the RAs and DECs of the sources in the table. 
		convert_to_sexagesimal: Boolean; True if RA and Dec need to be converted from decimal degrees to sexagesimal.
		output_name: The filename of the output to be used by DS9.
		labels: A boolean for specifying whether or not each region should be labelled. Labels must be included in the table.
		labelcol: The name of the column containing the source labels.
		color: The desired colour of the regions.
		radius: The desired radius of the regions (needs to include '' if wanted in arcseconds).
		Everything else: Global settings to be applied to the regions drawn in DS9.
	'''

	#retrieve the RAs and Decs from the table
	RAs = T[RAcol]
	DECs = T[DECcol]
	
	#make a new file for the DS9 settings if it doesn't exist; if it does exist, then the global settings are not written to the file and everything else is appended to it
	try:
		with open(output_name, 'x') as f:
			#write a header line to the file, containing all of the gloal settings
			f.write('global dashlist=%s width=%s font="%s" select=%s highlite=%s dash=%s fixed=%s edit=%s move=%s delete=%s include=%s source=%s\n'%(dashlist,width,font,select,highlite,dash,fixed,edit,move,delete,include,source))
			#specify the coordinate system in the next line
			f.write('%s\n'%coords)
	except FileExistsError:
		pass

	#open the text file in which the DS9 input will be written
	with open(output_name, 'a+') as f:
		#cycle through each coordinate
		for i in range(len(RAs)):
			#if the coordinates are given in decimal degrees, then they need to be converted to sexagesimal
			if (convert_to_sexagesimal == True):
				RA_hms = astrom.deg_to_hms(RAs[i])
				DEC_dms = astrom.deg_to_dms(DECs[i])
				#now convert to a usable string
				RA = astrom.sexagesimal_to_string(RA_hms)
				DEC = astrom.sexagesimal_to_string(DEC_dms)
			#if the coordinates were already in sexagesimal format, then the coordinates can be kept as is
			elif (convert_to_sexagesimal == False):
				RA = RAs[i]
				DEC = DECs[i]
			#account for the possibility that convert_to_sexagesimal was entered as neither True nor False
			else:
				raise TypeError('table_to_DS9_regions argument \'convert_to_sexagesimal\' must be either True or False')

			#now need to format the regions; first, the case where labels are to be added
			if (labels == True):
				#retrieve the labels from the table
				LAB = T[labelcol][i]
				#write the specifications to the output file
				f.write('circle(%s, %s, %s) # color=%s text={%s}\n'%(RA, DEC, radius, color, LAB))

			#now for the case where labels aren't being added
			elif (labels == False):
				#write the specifications to the output file (no label)
				f.write('circle(%s, %s, %s) # color=%s \n'%(RA, DEC, radius, color))

			#finally, account for the possibility that labels was entered as neither True nor False
			else:
				raise TypeError('table_to_DS9_regions argument \'labels\' must be either True or False')


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
	if xtype in [int, float, str]:
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

