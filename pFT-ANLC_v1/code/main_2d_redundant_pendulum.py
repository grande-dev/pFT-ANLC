#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 19:44:12 2022

@authors: Davide Grande
          Andrea Peruffo

The main file to run a simulation campaign for the training of a 
2-dimensional system redundant inverted pendulum system.

    This scripts executes recursive callbacks to the ANLC training file and
    saves the statistics at the end.
    - An incremental seed is passed to the ANLC file (seed_);

"""

import timeit
import os
import sys
import dreal
import matplotlib
import numpy as np
import torch.onnx
import logging

from cegis_ftc import cegis
import utilities.saving_log as saving_log
import utilities.saving_stat as saving_stat
from utilities.nn import Net
from utilities.models import RedundantPendulum as UsedModel
import configuration.config_2d_redundant_pendulum as config_file
import closed_loop_testing.cl_2d_redundant_pendulum as cl
import systems_dynamics.dynamics_2d_redundant_pendulum as dynamic_sys

logging.basicConfig(level=logging.INFO)

# Libraries configuration
torch.set_default_dtype(torch.float64)  # setting default tensor type
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# dReal configuration
config = dreal.Config()
config.use_polytope_in_forall = True
config.precision = 1e-6  # delta-precision

# Setting all the parameters defined in the configuration file
parameters, dyn_sys_params = config_file.set_params()

# Initialising directories (this prevents overwriting old results)
# Generating result (top) directory
try:
    folder_results = "results"
    current_dir = os.getcwd()
    final_dir = current_dir + "/" + folder_results
    os.mkdir(final_dir)

except OSError:
    logging.info(f"\nOverall result directory: \n{final_dir} already existing!\n")
else:
    logging.info(f"\nOverall result directory sucessfully created as: \n{final_dir}\n")

# Generating campaign directory
try:
    folder_results_campaign = "campaign_" + str(parameters['campaign_run'])
    current_dir = os.getcwd()
    final_dir_campaign = final_dir + "/" + folder_results_campaign
    os.mkdir(final_dir_campaign)
except OSError:
    print(f"\nRun result directory: \n{final_dir_campaign} already existing!\n")
    logging.error("\n\nERROR: Simulation campaign not started as your old results would be overwritten.")
    print("\nPlease update the value of 'parameters['campaign_run']' first (in the configuration file)!")
    sys.exit()
else:
    logging.info(f"\nRun result directory sucessfully created as: \n{final_dir_campaign}\n")


'''
Main loop
'''
seed_ = parameters['init_seed'] - 1
time_elapse_hist = np.zeros(parameters['tot_runs'])
iterations_hist = np.zeros(parameters['tot_runs'])
conv_hist = np.zeros(parameters['tot_runs'])  # convergence history
to_learner_hist = np.zeros(parameters['tot_runs'])  # learner TO history
to_fals_hist = np.zeros(parameters['tot_runs'])  # falsifier TO history
to_fals_check = np.zeros(parameters['tot_runs'])  # falsifier TO history check
start_stat = timeit.default_timer()  # initialise timer
tot_falsifier_to = 0  # total number of falsifier time out
tot_learner_to = 0  # total number of learner time out
count_conv = 0  # total number of coverged tests


for i_loop in range(parameters['tot_runs']):
    seed_ += 1  # incrementing seed over each run

    # callback to the training file
    parameters, exit_info = cegis(parameters, seed_,
                                  Net,
                                  UsedModel().vars_, UsedModel.f_torch, UsedModel.f_symb,
                                  config,
                                  final_dir_campaign, i_loop, saving_log,
                                  cl, dynamic_sys)

    print(f"\n{i_loop + 1}/{parameters['tot_runs']} training run terminated.\n\n")

    # Saving convergence information for final statistics
    time_elapse_hist[i_loop] = (exit_info['stop'] - exit_info['start'])
    iterations_hist[i_loop] = (exit_info['tot_iters'])
    conv_hist[i_loop] = exit_info['found_lyap_f']
    to_learner_hist[i_loop] = exit_info['to_learner']
    to_fals_hist[i_loop] = exit_info['to_fals']
    if exit_info['found_lyap_f']:
        count_conv += 1
    if exit_info['to_fals']:
        to_fals_check[i_loop] = 1

'''
Final statistics
'''
stop_stat = timeit.default_timer()
print(f"Average running time = {time_elapse_hist.mean()} ['']")
print(f"Average iteration per run = {iterations_hist.mean()}")

# Postpro clock
minutes_elapsed_postp = (stop_stat - start_stat) / 60

saving_stat.gen_stat(parameters,  
                     seed_, minutes_elapsed_postp, time_elapse_hist, 
                     iterations_hist, conv_hist, to_learner_hist, to_fals_hist, 
                     to_fals_check, count_conv, final_dir_campaign)


# Saving dynamic parameters
f = open(final_dir_campaign + "/dyn_system_params.txt",'w')
f.write(str(dyn_sys_params))
f.close()
