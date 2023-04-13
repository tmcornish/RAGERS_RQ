#!/bin/bash

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh 
conda activate photom
python Select_radio_quiet_sample_v3.py $1 $2 $3 $4
conda deactivate