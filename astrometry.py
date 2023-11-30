############################################################################################################
# Module containing functions relating to astrometry.
###########################################################################################################

import numpy as np
from astropy.table import Table, hstack, Column, vstack
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy.io import fits
import general as gen
from matplotlib import pyplot as plt


def hms_to_deg(h, m, s):
	'''
	Converts coordinates from h:m:s format to decimal degrees.
		h: The number of hours from the origin (int).
		m: The number of minutes after the hour (int).
		s: The number of seconds after the minute (float).
	'''
	d = 15. * (h + m/60. + s/3600.)
	return d 


def dms_to_deg(d, m, s):
	'''
	Converts coordinates from d:m:s format to decimal degrees.
		d: The number of degrees from the origin.
		m: The number of minutes after the hour.
		s: The number of seconds after the minute.
	'''
	d_dec = abs(d) + m/60. + s/3600.
	if (d >= 0.):
		return d_dec
	else:
		return -d_dec 


def deg_to_hms(d):
	'''
	Converts coordinates from decimal degrees to h:m:s format.
		d: The coordinate in decimal degrees.
	'''
	#first calculate the number of whole hours that make up the coordinate
	h = np.floor(d/15.)
	#next calculate the number of whole minutes
	m = np.floor(60. * (d/15. - h))
	#and finally the number of seconds left over
	s = 3600. * (d/15. - h) - (60. * m)
	return h, m, s 



def deg_to_dms(dd):
	'''
	Converts coordinates from decimal degrees to d:m:s format.
		dd: The coordinate in decimal degrees.
	'''
	#if the coordinate is negative, needs to be coverted to a positive number temporarily
	dd_abs = abs(dd)
	#first calculate the number of whole degrees that make up the coordinate
	d = np.floor(dd_abs)
	#next calculate the number of whole minutes
	m = np.floor(60. * (dd_abs - d))
	#and finally the number of seconds left over
	s = 3600. * (dd_abs - d) - (60. * m)
	if (dd >= 0):
		return d, m, s
	else:
		return -d, m, s


def sexagesimal_to_string(c):
	'''
	Takes a coordinate in h:m:s or d:m:s format and converts them to one string.
		c: A tuple containing the individual (or lists of) hms or dms components of the coordinate.
	'''
	_str = '%i:%i:%.4f'%(c[0], c[1], c[2])
	return _str


def find_coords_at_pixel(f, X, Y):
	'''
	Finds the RA and Dec. at a specific pixel position in an image.
		f: The filename of the image.
		X: The x-coordinate of the pixel of interest.
		Y: The y-coordinate of the pixel of interest.
	'''
	#retrieve the header of the image
	hdr = fits.getheader(f)
	#create a WCS object from the information in the header
	w = wcs.WCS(hdr)
	#retrieve the RA and Dec of the pixel
	RA_c, Dec_c = w.wcs_pix2world(X, Y, 1)
	return RA_c, Dec_c



def find_central_coords(f):
	'''
	Finds the central RA and Dec. of an image by using information from the header.
		f: The filename of the image.
	'''
	#retrieve the header of the image
	hdr = fits.getheader(f)
	#retrieve the number of pixels in X and Y from the header
	Nx = hdr['NAXIS1']
	Ny = hdr['NAXIS2']
	#create a WCS object from the information in the header
	w = wcs.WCS(hdr)
	#retrieve the RA and Dec of the central pixel
	RA_c, Dec_c = w.wcs_pix2world(Nx/2., Ny/2., 1)
	return RA_c, Dec_c


def find_pixscale(f, hdu=0, update_header=False):
	'''
	Uses the WCS information from an image to determine the pixel scale in ''/pix.
		f: The filename of the image.
		hdu: The index of the HDU in which the image data are kept.
		update_header: Whether or not to write the pixel scale to the header of the image.
		open the image
	'''
	with fits.open(f) as i:
		#retrieve the header of the image
		hdr = i[hdu].header
		#check if the pixel scale is in the image header under the keyword PIXSCALE
		try:
			ps = hdr['PIXSCALE']
		except KeyError:
			#create a WCS object from the information in the header
			w = wcs.WCS(hdr)
			#find the pixel scale (in arcsec/pix) in each dimension
			ps_all = wcs.utils.proj_plane_pixel_scales(w) * 3600.
			#use the average across all dimensions to estimate the scalar pixel scale of the image
			ps = np.mean(ps_all)
			#if the header is to be updated, then do so
			if update_header == True:
				hdr['PIXSCALE'] = (round_sigfigs(ps,5), 'Pixel scale in arcsec/pix')
				i.writeto(f, overwrite=True)
	return ps



def cross_match_catalogues(
	t1, 
	t2, 
	RAcol1, 
	DECcol1, 
	RAcol2, 
	DECcol2, 
	tol=np.inf, 
	utol=u.arcsec, 
	match='best', 
	join='1and2'
):
	'''
	Takes two tables and cross-matches them within a set tolerance for separation. Returns a merged table
	containing data for either all sources or just those with matches. Can include just the nearest match
	for each source, or every match within the tolerance.
	
	Parameters
	----------
	t1: astropy.table.Table 
		Data from the first catalogue.

	t2: astropy.table.Table
		Data from the second catalogue.

	RAcol1: str
		The name of the RA column in the first catalogue.

	DECcol1: str
		The name of the DEC column in the first catalogue.

	RAcol2: str
		The name of the RA column in the second catalogue.

	DECcol2: str
		The name of the DEC column in the second catalogue.

	tol: float 
		The tolerance within which matches will be considered valid.

	utol: astropy.units
		The units of the specified tolerance.

	join: str ('best', 'all')
		The type of joining used to create the cross-matched table.

	match: str ('1and2', '1or2', 'all1', 'all2')
		Whether the matching should only include nearest or all matches.


	Returns
	----------
	t_matched: astropy.table.Table
		Table containing the matched sources.
	'''

	#determine which of the two tables is shortest
	if (len(t1) <= len(t2)):
		ts0, RAcol_s, DECcol_s = t1, RAcol1, DECcol1
		tl0, RAcol_l, DECcol_l = t2, RAcol2, DECcol2
	else:
		tl0, RAcol_l, DECcol_l = t1, RAcol1, DECcol1
		ts0, RAcol_s, DECcol_s = t2, RAcol2, DECcol2
	#select only sources with RA and DEC data available
	ts = ts0[(~np.isnan(ts0[RAcol_s])) * (~np.isnan(ts0[DECcol_s]))]
	tl = tl0[(~np.isnan(tl0[RAcol_l])) * (~np.isnan(tl0[DECcol_l]))]
	#also make tables containing just the sources without these data available
	tsn = ts0[(np.isnan(ts0[RAcol_s])) * (np.isnan(ts0[DECcol_s]))]
	tln = tl0[(np.isnan(tl0[RAcol_l])) * (np.isnan(tl0[DECcol_l]))]
	#create a list of the RA and DEC columns from each table
	ast_cols_list = [ts[RAcol_s], ts[DECcol_s], tl[RAcol_l], tl[DECcol_l]]
	#ensure each column is in units of degrees; if it has no units, assume the values are in degrees
	for col in ast_cols_list:
		if col.unit == None:
			col.unit = u.degree
		else:
			col = col.to(u.degree)
	#retrieve the updated columns from the list and store them as separate Columns
	RA_s, DEC_s, RA_l, DEC_l = ast_cols_list
	#create SkyCoord objects using the RA and DEC coordinates
	coords_s = SkyCoord(ra=RA_s, dec=DEC_s)
	coords_l = SkyCoord(ra=RA_l, dec=DEC_l)
	#if only nearest matches are required, use astropy's match_to_catalog_sky to find them
	if match == 'best':
		#find best counterparts in the longer table for each source in the shorter table, and the separations between matched sources
		idx_ml, d_ml, _ = coords_s.match_to_catalog_sky(coords_l)
		#select matched sources from the longer table that are also within the separation tolerance
		idx_ml = idx_ml[d_ml <= tol * utol]
		matches_l = tl[idx_ml]
		#some sources in the longer table may have more than one match in the shorter;
		#account for this by reversing the matching process, but with no constraint on the separation
		coords_ml = coords_l[idx_ml]
		idx_ms, d_ms, _ = coords_ml.match_to_catalog_sky(coords_s)
		matches_s = ts[idx_ms]
		#now each source should be matched to only one other, however there may be some matches that
		#appear twice; remove any duplicates based on repeated pairs of RA and DEC coordinates
		_, idx_u = np.unique(matches_s[[RAcol_s, DECcol_s]], axis=0, return_index=True)
		matches_s = matches_s[np.sort(idx_u)]
		matches_l = matches_l[np.sort(idx_u)]	
	#if all matches are required, use astropy's search_around_sky instead
	elif match == 'all':
		#find the indices of all counterparts for each source
		idx_ml, idx_ms, *_ = coords_s.search_around_sky(coords_l, tol * utol)
		#trim the catalogues so only the matched sources remain, duplicated as necessary depending on number of matches
		matches_s = ts[idx_ms]
		matches_l = tl[idx_ml]
	#if a non-string was entered as an argument, raise an error
	elif type(match) != str:
		raise TypeError('match must be either "best" or "all"')
	#if a string was entered but it isn't any of the available options, raise an error
	else:
		raise ValueError('match must be either "best" or "all"')

	#now merge the tables of matched sources; keep the order in which the tables were provided
	if (len(t1) <= len(t2)):
		t_matched = hstack([matches_s, matches_l], metadata_conflicts='silent')
		#list of column names in t_matched for columns coming from the short and long table, respectively
		cols_ms = t_matched.colnames[:len(ts.colnames)]
		cols_ml = t_matched.colnames[len(ts.colnames):]
	else:
		t_matched = hstack([matches_l, matches_s], metadata_conflicts='silent')
		#list of column names in t_matched for columns coming from the short and long table, respectively
		cols_ml = t_matched.colnames[:len(tl.colnames)]
		cols_ms = t_matched.colnames[len(tl.colnames):]

	#if only the matched sources are required, return the merged table as is
	if join == '1and2':
		return t_matched
	#if all sources are required, add all unmatched sources at the bottom of the table	
	elif join == '1or2':
		#find the sources from each catalogue that went unmatched
		ts.remove_rows(idx_ms)
		tl.remove_rows(idx_ml)
		#account for the possibility of column names changing during the merging process
		for cm,cs in zip(cols_ms, ts.colnames):
			#rename the columns in the unmatched and RA/DEC-lacking tables
			ts[cs].name = cm
			tsn[cs].name = cm
		for cm,cl in zip(cols_ml, tl.colnames):
			#rename the columns in the unmatched and RA/DEC-lacking tables
			tl[cl].name = cm
			tln[cl].name = cm
		#add these sources to the end of the matched catalogue, as well as the sources that had no RA and/or DEC data
		t_matched = vstack([t_matched, tl, tln, ts, tsn], metadata_conflicts='silent')
		#return the stacked table
		return t_matched
	#if all sources are required from the first table, then add the unmatcehd sources from that table to the bottom
	elif join == 'all1':
		#identify which table to include unmatched sources from (i.e. the first catalogue)
		if (len(t1) <= len(t2)):
			#find the sources from the first catalogue that went unmatched
			ts.remove_rows(idx_ms)
			#account for the possibility of column names changing during the merging process
			for cm,cs in zip(cols_ms, ts.colnames):
				#rename the columns in the unmatched and RA/DEC-lacking tables
				ts[cs].name = cm
				tsn[cs].name = cm
			#add these sources to the end of the matched catalogue, as well as the sources that had no RA and/or DEC data
			t_matched = vstack([t_matched, ts, tsn], metadata_conflicts='silent')
		else:
			#find the sources from the first catalogue that went unmatched
			tl.remove_rows(idx_ml)
			#account for the possibility of column names changing during the merging process
			for cm,cl in zip(cols_ml, tl.colnames):
				#rename the columns in the unmatched and RA/DEC-lacking tables
				tl[cl].name = cm
				tln[cl].name = cm
			#add these sources to the end of the matched catalogue, as well as the sources that had no RA and/or DEC data
			t_matched = vstack([t_matched, tl, tln], metadata_conflicts='silent')
		return t_matched
	#if all sources are required from the first table, then add the unmatcehd sources from that table to the bottom
	elif join == 'all2':
		#identify which table to include unmatched sources from (i.e. the second catalogue)
		if (len(t1) <= len(t2)):
			#find the sources from the first catalogue that went unmatched
			tl.remove_rows(idx_ml)
			#account for the possibility of column names changing during the merging process
			for cm,cl in zip(cols_ml, tl.colnames):
				#rename the columns in the unmatched and RA/DEC-lacking tables
				tl[cl].name = cm
				tln[cl].name = cm
			#add these sources to the end of the matched catalogue, as well as the sources that had no RA and/or DEC data
			t_matched = vstack([t_matched, tl, tln], metadata_conflicts='silent')
		else:
			#find the sources from the first catalogue that went unmatched
			ts.remove_rows(idx_ms)
			#account for the possibility of column names changing during the merging process
			for cm,cs in zip(cols_ms, ts.colnames):
				#rename the columns in the unmatched and RA/DEC-lacking tables
				ts[cs].name = cm
				tsn[cs].name = cm
			#add these sources to the end of the matched catalogue, as well as the sources that had no RA and/or DEC data
			t_matched = vstack([t_matched, ts, tsn], metadata_conflicts='silent')
		#return the stacked table
		return t_matched
	#if a non-string was entered as an argument, raise an error
	elif type(join) != str:
		raise TypeError('join must be either "1and2", "1or2", "all1", or "all2"')
	#if a string was entered but it isn't any of the available options, raise an error
	else:
		raise ValueError('join must be either "1and2", "1or2", "all1", or "all2"')


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
				RA_hms = deg_to_hms(RAs[i])
				DEC_dms = deg_to_dms(DECs[i])
				#now convert to a usable string
				RA = sexagesimal_to_string(RA_hms)
				DEC = sexagesimal_to_string(DEC_dms)
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


def aperture_mask(RA_c, DEC_c, r, RAgrid, DECgrid):
	'''
	Creates a circular aperture mask to be placed on a grid of (RA, Dec.) coordinates.

	Parameters
	----------
	RA_c, DEC_c: floats
		The central coordinates (in degrees) of the aperture.
	
	r: float
		The radius of the aperture (in degrees).

	RAgrid, DECgrid: 2D arrays
		Grids containing the central RA and Dec. of each pixel in a grid. Can be created using
		e.g. numpy.mgrid.


	Returns
	-------
	MASK: 2D array (boolean)
		Array with the same dimensions as RAgrid and DECgrid, equals True at pixels covered by the
		aperture.
	'''

	#circular aperture is actually elliptical in RA-Dec. due to the celestial sphere; define ellipse
	A = ((RAgrid - RA_c) * np.cos(DEC_c * np.pi / 180.)) ** 2.
	B = (DECgrid-DEC_c) ** 2.
	MASK = (A + B) <= (r ** 2.)

	return MASK


def apertures_area(coords, r, RAstep=0.001, DECstep=0.001, use_nsteps=False, nsteps=1000, save_fig=False, **plot_kwargs):
	'''
	Estimates the area covered by circular apertures by placing them on a grid of RA-Dec.
	coordinates.
	
	Parameters
	----------
	coords: array-like or SkyCoord
		List/array or a SkyCoord object of RA-Dec. coordinate pairs for	each aperture.

	r: float or array-like
		Radius of the apertures in degrees.

	RAstep, DECstep: floats
		Dimensions of each pixel in the grid (in degrees).

	use_nsteps: bool
		Whether to define the dimensions of each pixel as a fraction of the dimensions of the grid.

	nsteps: int or tuple
		Number of pixels to use along each dimension.

	save_fig: bool
		Whether to save a figure showing the positions of the apertures.

	**plot_kwargs
		Any remaining keyword arguments will be used to format the plot (if generated).

	Returns
	-------
	A: float
		Area covered by the apertures (in deg^2).
	'''

	#see if coordinates have been provided as SkyCoord objects
	if type(coords) == SkyCoord:
		#convert to a numpy array
		coords = np.array([coords.ra.value, coords.dec.value]).T

	#check to see if only one coordinate pair was provided
	if gen.get_ndim(coords) == 1:
		#convert to a 2D array, containing the coordinate pair as a single entry
		coords = np.array([coords])

	#see if only one radius was provided (in which case all apertures will have the same radius)
	if gen.get_ndim(r) == 0:
		r = np.full(len(coords), r)
	else:
		if type(r) == list:
			r = np.array(r)

	#define boundaries of the RA-Dec grid using the aperture coordinates, including padding
	padding = r.max() * 1.2
	RAmin = coords[:,0].min() - padding
	RAmax = coords[:,0].max() + padding
	DECmin = coords[:,1].min() - padding
	DECmax = coords[:,1].max() + padding

	#see if dimensions of each pixel are to be a fraction of the total grid dimensions
	if use_nsteps:
		#define the dimensions of each pixel
		if gen.get_ndim(nsteps) == 0:
			RAstep = (RAmax - RAmin) / nsteps
			DECstep = (DECmax - DECmin) / nsteps
		else:
			RAstep = (RAmax - RAmin) / nsteps[0]
			DECstep = (DECmax - DECmin) / nsteps[1]
	#else:
	RA_axis = np.arange(RAmin, RAmax+RAstep, RAstep)
	DEC_axis = np.arange(DECmin, DECmax+DECstep, DECstep)

	#create a grid of RA and Dec. coordinates
	RAgrid, DECgrid = np.mgrid[RAmin:RAmax+RAstep:RAstep, DECmin:DECmax+DECstep:DECstep]

	#create an array of zeros with the same dimensions as the grid
	covered = np.zeros(RAgrid.shape)

	#create a grid containing the areas of each pixel
	areas = DECstep * RAstep * np.cos(DEC_axis * np.pi / 180.)
	areas = np.tile(areas, (len(RA_axis), 1))

	#cycle through the apertures
	for i in range(len(coords)):
		#for the pixels covered by the aperture, replace the zeros with the area of the pixel
		apermask = aperture_mask(coords[i][0], coords[i][1], r[i], RAgrid, DECgrid)
		covered[apermask] = areas[apermask]

	#calculate the total area covered by the apertures
	A = covered.sum()

	#see if told to save a figure
	if save_fig:
		#see if a colourmap was provided in kwargs
		if 'cmap' in plot_kwargs:
			cmap = plot_kwargs['cmap']
		else:
			cmap = plt.cm.viridis

		#see if a set of axes has already been provided for the plot
		if 'ax' in plot_kwargs:
			ax = plot_kwargs['ax']
			#plot the positions of the apertures on the existing axes
			ax.imshow(covered.T, extent=[RAmin,RAmax,DECmin,DECmax], origin='lower', cmap=cmap)
		#otherwise, make a fresh one
		else:
			#see if a plotstyle dictionary or file has been provided
			if 'plotstyle' in plot_kwargs:
				plt.style.use(plot_kwargs['plotstyle'])

			#make the figure
			f, ax = plt.subplots()
			#label the axes
			ax.set_xlabel('RA (deg.)')
			ax.set_ylabel('Dec. (deg.)')

			#plot the positions of the apertures
			ax.imshow(covered.T, extent=[RAmin,RAmax,DECmin,DECmax], origin='lower', cmap=cmap)
			ax.invert_xaxis()

			#minimise unnecessary whitespace
			f.tight_layout()

			#see if a figure filename was provided
			if 'figname' in plot_kwargs:
				figname = plot_kwargs['figname']
			else:
				figname = 'Aperture_positions.png'
			f.savefig(figname, bbox_inches='tight', dpi=300)


	return A



def area_within_sensitivity_limit(smap_file, sens_lim, to_plot=False):
	'''
	Using a sensitivity map, calculates area of the survey (in sq. deg.) with sensitivity below
	a given limit (in mJy).

	Parameters
	----------
	smap_file: str
		Full path of the file containing the sensitivity map.

	sens_lim: float
		Sensitivity limit in mJy.

	to_plot: bool
		If True, will additionally return the (a) the 2D array indicating all pixels below the 
		sensitivity limit, and (b) coordinates defining the corners of the sensitivity map as a tuple: 
		(RA_max, RA_min, DEC_min, DEC_max). Convenient if using plt.imshow to display the desired area.

	Returns
	-------
	A: float
		Area covered by the apertures (in deg^2).



	extent: tuple
		If to_plot=True, additionally returns this tuple of coordinates defining the corners of
		the sensitivity map: (RA_max, RA_min, DEC_min, DEC_max).
	'''

	smap = fits.open(smap_file)
	#retrieve the data and header (NOTE: the main HDU has 3 axes but the third has one value labelled as 'wavelength' - not actually useful)
	smap_data = smap[0].data
	smap_hdr = smap[0].header
	#create a wcs object for just the first 2 axes
	w = wcs.WCS(smap_hdr, naxis=2)

	#create a copy of the smap_data where everything below the sensitivity limit = 1 and everything above = 0
	smap_sel = np.zeros(smap_data[0].shape)
	smap_sel[smap_data[0] <= sens_lim] = 1.

	#calculate the area of each pixel in sq. deg.
	dRA = smap_hdr['CDELT1']
	dDEC = smap_hdr['CDELT2']
	A_pix = np.abs(dRA * dDEC)
	#calculate the area of the survey that is below the sensitivity limit
	A = smap_sel.sum() * A_pix

	if to_plot:
		#number of pixels in each dimension
		NAXIS1 = smap_hdr['NAXIS1']
		NAXIS2 = smap_hdr['NAXIS2']
		#get the coordinates at the lower left and upper right corners of the S2COSMOS field
		RA_max, DEC_min = w.wcs_pix2world(0.5, 0.5, 1)
		RA_min, DEC_max = w.wcs_pix2world(NAXIS1+0.5, NAXIS2+0.5, 1)
		#define the `extent' within which the data from the sensitivity map can be plotted with imshow
		extent = (RA_max, RA_min, DEC_min, DEC_max)

		return A, smap_sel, extent

	else:
		return A


