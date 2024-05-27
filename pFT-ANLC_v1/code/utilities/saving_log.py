#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:53:31 2023

@authors: Davide Grande
          Andrea Peruffo

A function to save the log file of a training run.

"""

import numpy as np
from datetime import datetime
import timeit
import time

def gen_log(found_lyap_f, 
            parameters, x, config, model, seed_,
            to_fals, to_learner, seconds_elapsed, minutes_elapsed, hours_elapsed,
            out_iters, i_epoch, start, init_date, end_date, falsifier_elapsed,
            final_dir_run):

    print("Saving logs ...\n")
        
    dt_string_init = init_date.strftime("%d/%m/%Y %H:%M:%S")  # date init training
    dt_string_end = end_date.strftime("%d/%m/%Y %H:%M:%S")  # date end training
    
    # Postpro clock
    stop_postp = timeit.default_timer()
    minutes_elapsed_postp = (stop_postp - start) / 60
    
    result_report = [f"Run of the {parameters['system_name']} system.\n" + 
                     f"Convergence reached = {found_lyap_f}\n\n" +
                     "TRAINING PARAMS: \n" + 
                     f"Seed = {seed_}\n" +
                     f"max_loop_number = {parameters['max_loop_number']}\n" +
                     f"max_iters = {parameters['max_iters']}\n" +
                     f"Initial dataset dimension = {parameters['N']}\n" +
                     f"Final dataset dimension = {len(x)}\n" + 
                     f"Using a sliding window = {parameters['sliding_window']}\n" +
                     f"Maximum dataset dimension (if using sliding wind) = {parameters['N_max']}\n" +
                     f"Equilibrium (x_star) = {parameters['x_star']}\n" +
                     "\n\n" +
                     "LYAPUNOV ANN: \n" + 
                     f"layers size = {parameters['size_layers']}\n" +
                     f"bias = {parameters['lyap_bias']}\n" +
                     f"lyap_activations = {parameters['lyap_activations']}\n" +
                     f"beta_sfpl = {parameters['beta_sfpl']}\n" +
                     f"Clipping Lyapunov weights = {parameters['clipping_V']}\n" +
                     "\n\n" +
                     "CONTROL ANN: \n" +
                     f"Use linear control = {parameters['use_lin_ctr']}\n" + 
                     f"Linear control has bias = {parameters['lin_contr_bias']}\n" +
                     f"Linear control initialised = {parameters['control_initialised']}\n" +
                     f"use_saturation = {parameters['use_saturation']}\n" +
                     f"ctrl_sat = {parameters['ctrl_sat']}\n" +
                     "If nonlinear control law is used, then:\n" +
                     f"layers size = {parameters['size_ctrl_layers']}\n" +
                     f"bias = {parameters['ctrl_bias']}\n" +
                     f"ctrl_activations = {parameters['ctrl_activations']}\n" +
                     "\n\n" +
                     "LEARNER: \n" + 
                     f"Learning rate Lyap. = {parameters['learning_rate']}\n" +
                     f"Learning rate control = {parameters['learning_rate_c']}\n" + 
                     f"use l.r. scheduler = {parameters['use_scheduler']}\n" +
                     f"scheduler period = {parameters['sched_T']}\n" +
                     f"Enforce CLF shape = {parameters['enforce_CLF_shape']}\n" +
                     f"tau_overbar = {parameters['tau_overbar']}\n" +
                     "\nLYAPUNOV RISK:\n" +
                     f"alpha_1 (Weight V) = {parameters['alpha_1']}\n" +
                     f"alpha_2 (Weight V_dot) = {parameters['alpha_2']}\n" +
                     f"alpha_3 (Weight V0) = {parameters['alpha_3']}\n" +
                     f"alpha_4 (Weight V tuning) = {parameters['alpha_4']}\n" +
                     f"alpha_5(overall weight) = {parameters['alpha_5']}\n" +
                     f"alpha_roa (ROA tuning) = {parameters['alpha_roa']}\n" +
                     f"offset V_dot = {parameters['off_Lie']}\n" +
                     "\n\n" +
                     "FALSIFIER (SMT): \n" +
                     f"Falsifier domain = {parameters['epsilon']} --- {parameters['gamma_overbar']}\n" +
                     f"config.precision = {config.precision}\n"
                     f"zeta_SMT (SMT CE point cloud) = {parameters['zeta_SMT']}\n" +
                     "\n\n" + 
                     "DISCRETE FALSIFIER (DF): \n" +  
                     f"zeta_D (DF CEs added at each callback) = {parameters['zeta_D']} \n" +
                     f"grid_points = {parameters['grid_points']}" +
                     "\n\n" +
                     "RESULTS: \n" +
                     f"Falsifier Time Out = {to_fals}\n" +
                     f"Learner Time Out = {to_learner}\n" +
                     "Time elapsed:\n" +
                     f"seconds = {seconds_elapsed}\n" +
                     f"minutes = {minutes_elapsed}\n" +
                     f"hours = {hours_elapsed}\n" +
                     f"Falsifier time [']: {falsifier_elapsed}\n" +
                     f"Falsifier time [%]: {falsifier_elapsed/minutes_elapsed*100}\n" +
                     f"Postprocessing time ['] = {minutes_elapsed_postp}\n" +
                     f"Training iterations completed = {(out_iters)}\n" +
                     f"Training epochs (last iteration) = {i_epoch}\n\n" + 
                     f"Training start = {dt_string_init}\n" +
                     f"Training end = {dt_string_end}"          
                     ]

    np.savetxt(final_dir_run + "/logs.txt", result_report, fmt="%s")
