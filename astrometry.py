############################################################################################################
# Module containing functions relating to astrometry.
###########################################################################################################


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