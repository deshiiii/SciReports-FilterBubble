#!/bin/bash

# Set the path to the directory containing your module
module_directory="./divAtScale"

# Add the module directory to the PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$module_directory"

## 1. intra-s
# python3 ./divAtScale/src/exp1-intra-sess/run_spatial_expr.py
# python3 ./divAtScale/src/exp1-intra-sess/run_novelty_expr.py
#
# ## 2. inter-s
# python3 ./divAtScale/src/exp2-inter-sess/run_spatial_expr.py
# python3 ./divAtScale/src/exp2-inter-sess/run_novelty_expr_bicm.py 2

## 3. inter-aff.
python3 ./divAtScale/src/exp3-inter-aff/run_spatial_expr.py
python3 ./divAtScale/src/exp3-inter-aff/run_novelty_expr_bicm.py

echo "All experiments complete"
