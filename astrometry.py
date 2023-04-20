############################################################################################################
# Module containing functions relating to astrometry.
###########################################################################################################

import numpy as np
from astropy.table import Table, hstack, Column, vstack
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy.io import fits

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

