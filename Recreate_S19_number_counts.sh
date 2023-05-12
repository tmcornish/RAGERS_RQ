#!/bin/bash

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh 
conda activate mcmc
python Recreate_S19_number_counts.py
conda deactivate