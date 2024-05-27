#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13:57:27 2024

@authors: Andrea Peruffo
          Davide Grande
          

This script contains initial sanity checks. 
If the tests are passed, the training run can be performed, otherwise the
training is halted before starting.

"""


def initial_checks(parameters):

    warn = 0

    assert len(parameters['size_layers']) == len(parameters['lyap_activations']), "ERROR in your config file setup: 'size_layers' dimension not consistent with 'lyap_activations'." 
    assert len(parameters['size_layers']) == len(parameters['lyap_bias']), "ERROR in your config file setup: 'size_layers' dimension not consistent with 'lyap_bias'." 
    assert parameters['size_layers'][-1] == 1, "ERROR in your config file setup (issue with your Lyapunov ANN): you are attempting to generate a non-scalar Lyapunov Function! (tip: the output layer must have dimension 1, namely the last element of 'size_layers' must be 1.)." 
    assert len(parameters['x_star']) == parameters['n_input'], "ERROR in your config file setup: 'x_star' dimension not consistent with 'n_input'." 
    if parameters['use_lin_ctr'] == False:
        # if nonlinear control law is used
        assert len(parameters['size_ctrl_layers']) == len(parameters['ctrl_bias']), "ERROR in your config file setup (issue with your nonlinear control ANN): 'size_ctrl_layers' dimension not consistent with 'ctrl_bias'." 
        assert len(parameters['size_ctrl_layers']) == len(parameters['ctrl_activations']), "ERROR in your config file setup (issue with your nonlinear control ANN): 'size_ctrl_layers' dimension not consistent with 'ctrl_activations'."       

    xstar_position = sum(abs(parameters['x_star'])).item() 
    if xstar_position != 0.0:
        # the equilibrium is not in the origin
        if parameters['use_lin_ctr'] == True:
            # if linear control law is used
            assert parameters['lin_contr_bias'] == True, "ERROR in your config file setup: you are trying to stabilise an equilibrium not in the origin without using a control bias (tip: set 'lin_contr_bias' = True)."
        else: 
            # if nonlinear control law is used
            ctr_bias = False
            for iC in range(len(parameters['size_ctrl_layers'])):
                ctr_bias = ctr_bias or parameters['ctrl_bias'][iC]
            assert ctr_bias == True, "ERROR in your config file setup: you are trying to stabilise an equilibrium not in the origin without using a control bias (tip: set at least one element of 'ctrl_bias' to True)."

    if parameters['use_saturation'] == True:
        assert parameters['size_ctrl_layers'][-1] == len(parameters['ctrl_sat']), "ERROR in your config file setup (issue with your nonlinear control ANN): 'size_ctrl_layers' dimension not consistent with 'ctrl_sat'." 
    
    # warning on number of points 
    no_DF_points = pow(parameters['grid_points'], parameters['n_input'])
    if no_DF_points > 1.e5 and no_DF_points <= 1.e8:
        print("\n\nWARNING: your Discrete Falsifier is using a very high number of points. The training will be slow. Consider reducing the value of 'grid_points'.\n\n")
        warn += 1
    if no_DF_points > 1.e8:
        print("\n\nWARNING: your Discrete Falsifier is using a STRATOSPHERIC number of points. The training will be SEVERLY SLOWED DOWN. Consider reducing the value of 'grid_points'.\n\n")
        warn += 1

    return warn


def check_ANN_model(model, vars_):
    # Check the ANN is correctly instantiated

    warn = 0

    assert len(model.size_layers) == len(model.lyap_bias), "ERROR in your config file setup (issue with your Lyapunov ANN): 'size_layers' dimension not consistent with 'lyap_bias'."
    assert len(model.size_layers) == len(model.activs), "ERROR in your config file setup (issue with your Lyapunov ANN): 'size_layers' dimension not consistent with 'lyap_activations'."
    assert len(model.size_ctrl_layers) == len(model.ctrl_bias), "ERROR in your config file setup (issue with your control ANN): 'size_ctrl_layers' dimension not consistent with 'ctrl_bias'." 
    assert len(model.size_ctrl_layers) == len(model.ctrl_activs), "ERROR in your config file setup: 'size_ctrl_layers' dimension not consistent with 'ctrl_activations'." 
    assert model.n_input == len(vars_), "ERROR in your config file setup: 'n_input' dimension not consistent with the size of 'vars_' in your model class (tip: in your model class, check the 'def __init__(self)' method)." 
    assert model.size_layers[-1] == 1, "ERROR in your config file setup (issue with your Lyapunov ANN): you are attempting to generate a non-scalar Lyapunov Function!." 

    return warn
