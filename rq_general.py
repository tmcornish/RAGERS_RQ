############################################################################################################
# Module containing general functions and variables to be used in the analysis of the environments of 
# radio-quiet massive galaxies for RAGERS.
############################################################################################################

####################################
#### RELEVANT PATHS & VARIABLES ####
####################################

#relevant directories
PATH_RAGERS = '/home/cornisht/RAGERS/'
PATH_SCRIPTS = PATH_RAGERS + '/Scripts/Analysis_repo/'
PATH_CATS = PATH_RAGERS + '/Catalogues/'
PATH_PLOTS = PATH_RAGERS + '/Plots/'
PATH_DATA = PATH_RAGERS + '/Data/'


#################################
#### FORMATTING OUTPUT ##########
#################################


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