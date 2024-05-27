#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:39:58 2023

@authors: Davide Grande
          Andrea Peruffo

A function collecting the parameters of the training.

"""

import numpy as np
import torch


def set_params():

    # Synthesis campaign parameters
    campaign_params = {
        'init_seed': 1,        # initial campaign seed.
        'campaign_run': 100,   # number of the run:
                               # the results will be saved in /results/campaign_'campaign_run'.
        'tot_runs': 3,        # total number of runs in the campaign (each run will be initialised with a different seed).
        'max_loop_number': 1,  # number of loops per run (default = 1, >1 means that the weights will be re-initialised).
                               # default value = 1.
        'max_iters': 2000,     # number of maximum learning iterations per run.
        'system_name': "2d_auv_faulty_efficiency_loss",  # name of the systems to be controlled (only used in the statistics file, pick any name).
        'x_star': torch.tensor([0.5, 0.0]),  # x*: target equilibrium point.
    }        

    # Parameters for the learner
    learner_params = {
        'N': 500,                  # initial dataset size.
        'N_max': 1000,             # maximum dataset size (if using a sliding window).
        'sliding_window': True,    # use sliding window (default = True).
        'learning_rate': 0.01,     # learning rate Lyapunov ANN (lambda_c).
        'learning_rate_c': 0.1,    # learning rate control ANN (lambda).
        'use_scheduler': True,     # use learning rate scheduler to enable a dynamic learning rate (default = True).
        'sched_T': 500,            # cosine annealing scheduler period.
        'print_interval': 200,     # interval of loss function printouts.
        'enforce_CLF_shape': True, # require the CLF to approximate a desired shape, i.e. enforce --> L_ELR < tau_overbar.
        'tau_overbar': 1.0,        # maximum error on L_ELR.
    }

    # Parameters for the Lyapunov ANN
    lyap_params = {
        'n_input': 2,                 # input dimension (n = n-dimensional system).
        'beta_sfpl': 2,               # the higher, the steeper the Softplus, the better approx. sfpl(0) ~= 0. If unsure, do not modify it.
        'clipping_V': True,           # clip weight of Lyapunov ANN to positive values only.
        'size_layers': [10, 10, 1],   # CAVEAT: the last entry needs to be = 1 (this ANN outputs a scalar)!
        'lyap_activations': ['pow2', 'linear', 'linear'],
        'lyap_bias': [False, False, False],
    }

    # Parameters for the control ANN
    control_params = {
        'use_lin_ctr': False,            # use linear or nonlinear control law.
        'lin_contr_bias': True,          # use bias on linear control law, namely use: u = Kx + b.
        'control_initialised': False,    # initialised the linear control ANN with pre-computed state-feedback law.
        'init_control': torch.tensor([[-23.58639732, -5.31421063]]),  # initial the control solution if a linear control law is selected.
        'size_ctrl_layers': [30, 3],   # CAVEAT: the last entry is the number of control actions! In this case the system has 3 actuators.
        'ctrl_bias': [True, False],
        'ctrl_activations': ['tanh', 'linear'],
        'use_saturation': False,         # use saturations in the control law.
        'ctrl_sat': [1.0, 20., 20.],     # actuator saturation values:
                                         # this vector needs to be as long as 'size_ctrl_layers[-1]' (same size as the control vector).
    }

    falsifier_params = {
        # a) SMT parameters
        'epsilon': 0.025,         # CLF domain lower boundary.
        'gamma_overbar': 1.0,     # CLF domain upper boundary.
        'zeta_SMT': 200,          # how many points are added to the dataset after a CE box
                                  # is returned by the SMT Falsifier.                   
        # b) Discrete Falsifier parameters
        'grid_points': 60,        # sampling size grid.
        'zeta_D': 200,            # how many CE points are added at each DF callback.
    }


    loss_function = {
        # Loss function tuning
        'alpha_1': 1.0,     # weight V.
        'alpha_2': 1.0,     # weight V_dot.
        'alpha_3': 1.0,     # weight V0.
        'alpha_4': 1.0,   # weight tuning term V.
        'alpha_roa': 1.0*falsifier_params['gamma_overbar'],  # CLF steepness.
        'alpha_5': 1.0,     # general scaling factor.
        'off_Lie': 0.0,     # additional penalisation of the Lie derivative.
    }

    # Parameters specific to the dynamical system
    dyn_sys_params = {
        'm' : 500.0,   # AUV mass
        'Jz' : 300.0,  # inertia around z-axis
        'Xu' : 6.106,  # linear drag coefficient - surge
        'Xuu' : 5.0,   # quadratic drag coefficient - surge
        'Nr' : 210.0,  # linear drag coefficient - angular velocity around z-axis
        'Nrr' : 3.0,   # quadratic drag coefficient - angular velocity around z-axis
        'l1x' : -1.01, 
        'l1y' : -0.353, 
        'alpha1' : np.deg2rad(110.0),
        'l2x' : -1.01,  
        'l2y' : 0.353,
        'alpha2' : np.deg2rad(70.0),
        'l3x' : 0.75,
        'l3y' : 0.0, 
        'alpha3' : np.deg2rad(180.0),
        'h1': 1.,  # nominal health status of actuator 1
        'h2': 1.,  # nominal health status of actuator 2
        'h3': 1.,  # nominal health status of actuator 3
    }

    # Faulty cases
    params_faults = {
        'phi_i': [[1, 1, 1]],
    }

    params_efficiency = {
        'n_input_eff': 1, # how many loss of efficiency cases
        'eff_low': 0,  # lowest expected efficiency to tolerate
        'eff_high': 1,  # highest expected efficiency to tolerate
        'grid_points_eff': falsifier_params['grid_points'],  # sampling size grid for Discrete Falsifier
    }

    # Postprocessing parameters
    postproc_params = {
        'execute_postprocessing': True,    # True: generate the plots, as per the options selected below.
        'verbose_info': True,              # print info with high verbosity.
        'dpi_': 300,                       # DPI number for plots.
        'plot_V': True,
        'plot_Vdot': False, # CAVEAT: This needs to be = False in the case of FTC with loss of efficiency
        'plot_u': True,
        'plot_4D_': True,                  # plot 4D Lyapunov f, Lie derivative and control function.
        'n_points_4D': 500,
        'n_points_3D': 100,
        'plot_ctr_weights': False,
        'plot_V_weights': False,
        'plot_dataset': True,
        'save_pdf': True,
    }

    # Closed-loop system testing parameters
    closed_loop_params = {
        'test_closed_loop_dynamics': True,
        'end_time': 100.0,  # [s] time span of closed-loop tests.
        'Dt': 0.01,         # [s] sampling time for Forward Euler integration.
        'skip_closed_loop_unconverged': True,  # skip closed-loop tests if the training does not converge (default= True:
                                               # setting this to False might lead to numerical errors).
    }

    # Joining all the parameters in a single dictionary
    parameters = {**campaign_params,
                  **learner_params,
                  **lyap_params,
                  **control_params,
                  **falsifier_params,
                  **loss_function,
                  **dyn_sys_params,
            	  **params_faults,
                  **params_efficiency,
                  **postproc_params,
                  **closed_loop_params}

    return parameters, dyn_sys_params

