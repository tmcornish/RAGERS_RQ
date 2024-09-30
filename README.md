This repo contains the pipeline used to conduct the analysis for [Cornish et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024MNRAS.533.1032C/abstract), which searches for overdensities of submillimetre galaxies (SMGs) in the environments of massive, radio-quiet galaxies in the COSMOS field.

# Prerequisites
Three public catalogues are required for the analysis:
1. COSMOS2020 ([Weaver et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJS..258...11W/abstract))
2. VLA-COSMOS ([Smolčić et al. 2017](https://ui.adsabs.harvard.edu/abs/2017A%26A...602A...2S/abstract))
3. S2COSMOS ([Simpson et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJ...880...43S/abstract)).
These should be downloaded and stored in the same directory.

Additionally, certain Python packages are required for the pipeline to run. These are listed in `requirements.txt`.

# Pipeline structure
Each stage of the pipeline has its own script. These can be run individually, or one can run multiple steps sequentially by toggling the `True/False` settings in MASTER.py and running this instead.

## Configuration
Global settings for the pipeline can be set in `general.py`. (Note that this script also contains various convenience functions for the pipeline - these are all located below the settings and can be left untouched.)

Individual stages often also have their own settings, which are located near the start of the corresponding script.

## Stages
The following briefly describes each stage of the pipeline. Many stages require previous steps be run in order to work, so the stages are listed here in the intended order of execution.

- Update the VLA-COSMOS catalogue with estimates of the rest-frame 500 MHz luminosities for each galaxy \[`Update_VLA_COSMOS_cat.py`\]
- Approximately recreate the results of the S2COSMOS completeness calculations carried out in Simpson et al. (2019) \[`Recreate_S19_completeness.py`\]
- Generate many versions of the S2COSMOS catalogue with perturbed flux densities \[`Generate_random_datasets.py`\]
- Recreate and plot the S2COSMOS number counts from Simpson et al. (2019) \[`Recreate_S19_number_counts.py`\]
- Select the sample of massive, radio-quiet galaxies used for the analysis from the COSMOS2020 catalogue (as well as the secondary sample of MLAGN/HLAGN flagged by Smolčić et al. 2017) and plot the sample in stellar mass-redshift space \[`Select_radio_quiet_sample.py`\]
- Calculate the submillimetre number counts in the environments of the massive, radio-quiet galaxies (or of the MLAGN/HLAGN) \[`Calculate_numcounts.py`\]
- \[Optional\] Create plots showing the areas probed in the search for submillimetre companions for each radio-quiet galaxy (or HLAGN/MLAGN) \[`Plot_search_areas.py`\]
- Fit Schechter functions to the calculated number counts \[`Fit_schechter_funcs.py`\]
- Fit Schechter functions but with $S_0$ and $\gamma$ fixed to the blank-field values from S2COSMOS \[`Fit_N0_only.py`\]
- \[Optional\] Create a LaTeX-formatted table of the best-fit Schechter parameters \[`Make_schechter_param_tables.py`\]
- Create contour plots showing the posterior distributions for each Schechter parameter  \[`Contour_plots.py`\]
- Determine the minimum density of SMGs required to detect an overdensity \[`Significance_test.py`\]
- Measure the density of peaks in the S2COSMOS SNR map with different SNR thresholds, plot histograms and perform a Kolmogorov-Smirnoff test to determine if the distribution differs significantly from the blank field \[`Density_histograms.py`\]
- Calculate the SMG overdensity parameter ($\delta$) and its uncertainty ($\sigma_\delta$) for the environments of each galaxy individually (rather than summary statistics for the whole sample) \[`Individual_deltas.py`\]
- Plot histograms showing the distributions of $\delta/\sigma_\delta$ for individual galaxy environments \[`Delta_significance_histograms.py`\]
- Backup the results of all major calculations from previous steps of the pipeline by making copies and saving to a new directory \[`Backup_results.py`\]