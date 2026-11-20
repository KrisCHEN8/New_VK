#!/bin/bash
set -e
export FDD_DATA=${FDD_DATA:-../07-D4-01_data/dfs.pkl}
python run_fit.py         # fit the SCM (mechs.pkl)
python run_graph.py       # edge checks
python run_eval.py        # one-step and 24-h simulation accuracy
python run_quant.py       # heat balance decomposition + figure
python run_quant2.py      # effect sizes with bootstrap, linear cross-check, strengths
python run_comp.py        # open- vs closed-loop responses (control compensation)
python run_validate.py 20 # threshold calibration + detection/naming validation (calibration.pkl)
python run_validate2.py   # attribution in kelvin vs true fault effects
python run_detect.py      # two-layer diagnosis of the real year
python run_figs_final.py  # figures
