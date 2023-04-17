#dictionary containing custom formatting for plots
styledict = {
	'figure.figsize' : (8., 6.),
	'legend.fontsize' : 14,
	'legend.shadow' : False,
	'legend.framealpha' : 0.9,
	'xtick.labelsize' : 22,
	'ytick.labelsize' : 22,
	'axes.labelsize' : 24,
	'axes.linewidth' : 2.,
	'image.origin' : 'lower',
	'xtick.minor.visible' : True,
	'xtick.major.size' : 7,
	'xtick.minor.size' : 4,
	'xtick.major.width' : 2.,
	'xtick.minor.width' : 1.5,
	'xtick.top' : True,
	'xtick.major.top' : True,
	'xtick.minor.top' : True,
	'xtick.direction' : 'in',
	'ytick.minor.visible' : True,
	'ytick.major.size' : 7,
	'ytick.minor.size' : 4,
	'ytick.major.width' : 2.,
	'ytick.minor.width' : 1.5,
	'ytick.right' : True,
	'ytick.major.right' : True,
	'ytick.minor.right' : True,
	#font.family: serif,
	#font.serif: Computer Modern Roman,
	'font.size' : 22,
	'font.weight' : 'bold',
	'ytick.direction' : 'in',
	'text.usetex' : True,				#enables the use of LaTeX style fonts and symbols
	'mathtext.fontset' : 'stix',
	'font.family' : 'STIXGeneral',
	'axes.formatter.useoffset' : False,
}

#colours
red = '#eb0505'
dark_red = '#ab0000'
ruby = '#C90058'
crimson = '#AF0404'
coral = '#FF4848'
magenta = '#C3027D'
orange = '#ED5A01'
green = '#0A8600'
light_green = '#11C503'
teal = '#00A091'
cyan = '#00d0f0'
blue = '#0066ff'
light_blue = '#00C2F2'
dark_blue = '#004ab8'
purple = '#6D04C4'
lilac = '#EB89FF'
plum = '#862388'
pink = '#E40ACA'
baby_pink = '#FF89FD'
fuchsia = '#E102B5'
grey = '#969696'

#obtain the figure size in inches
x_size, y_size = 8., 8.
#formatting for any arrows to be added to the plot for representing upper/lower limits
ax_frac = 1/40.				#the fraction of the y axis that the total length of a vertical arrow should occupy
al = ax_frac * y_size		#the length of each arrow in inches (ew, but sadly metric isn't allowed)
scale = 1./al				#'scale' parameter used for defining the length of each arrow in a quiver
aw = 0.0175 * al				#the width of each arrow shaft in inches
hw = 4.						#width of the arrowheads in units of shaft width
hl = 3.						#length of the arrowheads in units of shaft width
hal = 2.5					#length of the arrowheads at the point where they intersect the shaft 
							#(e.g. hal = hl gives a triangular head, hal < hl gives a more pointed head)