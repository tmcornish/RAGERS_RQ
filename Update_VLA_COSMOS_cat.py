############################################################################################################
# A script for updating the VLA-COSMOS catalogue with estimates of L_500MHz.
############################################################################################################

from astropy.table import Table
import numpy as np
import general as gen

#######################################################
###############    START OF SCRIPT    #################
#######################################################


#load the catalogue
cat_in = gen.PATH_CATS + 'VLA_3GHz_counterpart_array_20170210_paper_smolcic_et_al.fits'
t = Table.read(cat_in)

#get the 3 GHz (10 cm) and 1.4 GHz (21 cm) logged radio luminosities
L1 = 10. ** t['Lradio_10cm']
L2 = 10. ** t['Lradio_21cm']
#frequencies corresponding to these wavelengths (in GHz)
nu1 = 3.
nu2 = 1.4

#calculate the spectral index using L propto nu^alpha
alpha = np.log10(L1/L2) / np.log10(nu1/nu2)
#add this to the catalogue
t['spec_index'] = alpha

#use the spectral index to estimate log(L) at 500 MHz (60 cm)
nu3 = 0.5
logL3 = np.log10(L1) + alpha * np.log10(nu3/nu1)
#add this to the catalogue
t['Lradio_60cm'] = logL3

#save the catalogue
cat_out = cat_in[:-5] + '_with_L500MHz.fits'
t.write(cat_out, format='fits', overwrite=True)
