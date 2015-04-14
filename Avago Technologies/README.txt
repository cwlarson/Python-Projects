Project: Data Plot - Avago Technologies
Author: Casey W. Larson

REQUIREMENTS:
1.) System must have NumPy, SciPy, and MatPlotLib installed to run this program
2.) You must have the following files to run this program:
	a.) A dielist (CSV formatted) containing the X- and Y-coordinates, ID's, and Labels of the die in question in the format provided
	b.) Data files (CSV formatted) for each of the data sets that you wish to plot in the format provided
	c.) A [SET_NAME].gti file, where
		- [SET_NAME] will become the title of the plots and the name of the directory for the plots
		- The first line reads "f0: [VALUE OF f0]"
		- The second line reads "dielist: [PATHNAME OF DIELIST]"
		- The third through nth lines are filenames of the data sets to be plotted

HOW TO RUN:
1.) In Terminal, change to the directory containing dataPlot.py
2.) Run the command:	python dataPlot.py
3.) You will be prompted for the pathname for a .gti file.  The .gti file extension was created to denote the instructions for how to run the dataPlot.py script for the appropriate data files.

After providing the .gti pathname, the program will run and produce a folder containing the plots, graphs, and CSV files that were generated.


NOTE: Sample input files (dielist.csv, Init_2033.gti, and SJ_EG2001_wafer_probe_data/*) and output files (Init_2033_plts/*) have been provided here for demonstration.