############################################################################################################
# A script for formatting the best-fit results from the previous step into a LaTeX-stle table.
############################################################################################################

#import modules/packages
import os, sys
import general as gen
import numpy as np


#######################################################
###############    START OF SCRIPT    #################
#######################################################

#padding to use between lines of data
padding = '0.5em'

#filename to give to table
tab_name = gen.PATH_TABLES + 'Schechter_params.tex'
#see if PATH_TABLES already exists, and make it if it doesn't
if not os.path.exists(gen.PATH_TABLES):
	os.system(f'mkdir -p {gen.PATH_TABLES}')

#set up the table preamble
tab_lines = [r'\begin{table}']
#decide the caption and label
caption = '''
	Best-fit Schechter parameters for the differential and cumulative number counts, obtained from 
	MCMC fitting.
	'''
label = 'tab:params'
#add more table pramble, but idnent for ease of interpretation
add_lines = [r'\centering', r'\caption{%s}'%caption, r'\label{%s}'%label]
add_lines = ['\t' + s for s in add_lines]
tab_lines.extend(add_lines)
#begin tabular environment
alignment = 'c'
align_all = ''.join([alignment]*7)
tab_lines.append('\t' + r'\begin{tabular}{%s}'%align_all)

#column titles
title_lines = [r'\hline']
titles = ['', r'Radius [$\arcmin$]', r'$N_{0}$', r'$S_{0}$', r'$\gamma$']
#join the titles together
title_lines.append(r' & '.join(titles) + r'\\')
title_lines.append(r'\hline')
title_lines.append(r'\hline')
title_lines = ['\t\t' + s for s in title_lines]
#append these to the list of lines
tab_lines.extend(title_lines)

#set up a list for the lines of data in the table
data_lines = []

#data containing the best-fit parameters for the various radii
nc_data = [np.load(gen.PATH_CATS + 'Schechter_params/' + f'Differential_{r:.1f}am.npz')['ALL'] for r in gen.r_search_all]
cc_data = [np.load(gen.PATH_CATS + 'Schechter_params/' + f'Cumulative_{r:.1f}am.npz')['ALL'] for r in gen.r_search_all]

#get the number of different radii used
N_radii = len(gen.r_search_all)
#cycle through differential vs cumulative results
for counttype, data in zip(['Differential', 'Cumulative'], [nc_data, cc_data]):
	for j in range(N_radii):
		#get the radius used for the current datasets
		r = gen.r_search_all[j]
		#if the first radius used, include a multirow entry stating the type of number counts
		if r == gen.r_search_all[0]:
			params_all = ['\t\t' + r'\multirow{%i}{*}{%s}'%(N_radii,counttype), r'%g'%r]
		#otherwise keep the first entry empty
		else:
			params_all = ['\t\t' + r' & %g'%r]

		for i in range(len(data[j][0])):
			#parameter and errors
			pe = data[j][:,i]
			#figure out how many significant figures the parameter and uncertainties should be quoted to
			order_p = int(np.floor(np.log10(pe[0])))
			order_err = int(np.floor(np.log10(min(pe[1:]))))
			if order_p == order_err:
				Nsf_p = Nsf_err = 1
			else:
				Nsf_p = order_p - order_err + 1
				Nsf_err = 1
			pe = [gen.round_sigfigs(x, n) for x,n in zip(pe, [Nsf_p, Nsf_err, Nsf_err])]
			params_all.append(r'$%g_{-%g}^{+%g}$'%tuple(pe))
		'''
		if r == gen.r_search_all[-1]:
			data_lines.append(' & '.join(params_all) + r'\\' + '\n\t\t' + r'\hline' + '\n\t\t' + r'\vspace{%s}'%padding)
		else:
			data_lines.append(' & '.join(params_all) + r'\\' + '\n\t\t' + r'\vspace{%s}'%padding)
		'''
		data_lines.append(' & '.join(params_all) + r'\\' + '\n\t\t' + r'\vspace{%s}'%padding)

	#retrieve the best-fit parameters for the blank field
	bf_data = np.load(gen.PATH_CATS + 'Schechter_params/' + f'{counttype}_S2COSMOS.npz')[gen.s2c_key]
	#cycle through each parameter
	params_all = ['\t\t' + 'S2COSMOS', '--']
	for i in range(len(bf_data[0])):
		#parameter and errors
		pe = bf_data[:,i]
		#figure out how many significant figures the parameter and uncertainties should be quoted to
		order_p = int(np.floor(np.log10(pe[0])))
		order_err = int(np.floor(np.log10(min(pe[1:]))))
		if order_p == order_err:
			Nsf_p = Nsf_err = 1
		else:
			Nsf_p = order_p - order_err + 1
			Nsf_err = 1
		pe = [gen.round_sigfigs(x, n) for x,n in zip(pe, [Nsf_p, Nsf_err, Nsf_err])]
		params_all.append(r'$%g_{-%g}^{+%g}$'%tuple(pe))
	data_lines.append(' & '.join(params_all) + r'\\' + '\n\t\t' + r'\hline' + '\n\t\t'+ r'\vspace{%s}'%padding)


tab_lines.extend(data_lines)
#tab_lines.append('\t\t' + r'\hline')

#end tabular environment
tab_lines.append('\t' + r'\end{tabular}')
tab_lines.append(r'\end{table}')

#write to a the file
with open(tab_name, 'w') as f:
	f.write('\n'.join(tab_lines))



