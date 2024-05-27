#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:02:40 2023

@authors: Andrea Peruffo
          Davide Grande
"""

import dreal
import torch
import numpy as np

def translator(
            parameters,
            vars_,
            model, f_symb):
    """
    translates the torch network into a symbolic expression for dReal
    :param parameters:
    :param vars_:
    :param model:
    :return:
    """

    # Lyapunov ANN
    z = np.atleast_2d(np.array(vars_)).T
    for idx in range(len(model.layers)):

        # linear pass
        if model.lyap_bias[idx]:
            zhat = model.layers[idx].weight.detach().numpy() @ z \
                   + np.atleast_2d(model.layers[idx].bias.detach().numpy()).T
        else:
            zhat = model.layers[idx].weight.detach().numpy() @ z

        # dimension check
        assert zhat.shape[0] == 1 or zhat.shape[1] == 1

        # activation
        if model.activs[idx] == 'tanh':
            z = np.atleast_2d([dreal.tanh(v[0]) for v in zhat]).T
        elif model.activs[idx] == 'pow2':
            z = np.atleast_2d([dreal.pow(v[0], 2) for v in zhat]).T
        elif model.activs[idx] == 'pow3':
            z = np.atleast_2d([dreal.pow(v[0], 3) for v in zhat]).T 
        elif model.activs[idx] == 'sfpl':
            z = np.atleast_2d([
                1. / parameters['beta_sfpl'] * dreal.log(1. + dreal.exp(parameters['beta_sfpl'] * v[0]))
                for v in zhat
            ]).T
        elif model.activs[idx] == 'linear':
            z = zhat
        else:
            raise ValueError(f'Not Implemented Activation Function {model.activs[idx]}.')

    # V is contained in element [0,0] of the np.array
    V = z[0, 0]

    # Candidate control function
    if parameters['use_lin_ctr']:

        # linear control network
        c = np.atleast_2d(np.array(vars_)).T
        if parameters['lin_contr_bias']:
            chat = model.control.weight.detach().numpy() @ c \
                   + np.atleast_2d(model.control.bias.detach().numpy()).T
        else:
            chat = model.control.weight.detach().numpy() @ c
        c = chat
        
        # Implementing saturation
        if parameters['use_saturation']:
            c2 = np.atleast_2d([dreal.tanh(v[0]) for v in chat]).T
            for jC in range(chat.size):
                c[jC] = parameters['ctrl_sat'][jC]*c2[jC]
        else:
            c = c

    else:
        # nonlinear control network
        c = np.atleast_2d(np.array(vars_)).T
        for idx in range(len(model.ctrl_layers)):
            # linear pass
            if model.ctrl_bias[idx]:
                chat = model.ctrl_layers[idx].weight.detach().numpy() @ c \
                       + np.atleast_2d(model.ctrl_layers[idx].bias.detach().numpy()).T
            else:
                chat = model.ctrl_layers[idx].weight.detach().numpy() @ c

            # dimension check
            assert chat.shape[0] == 1 or chat.shape[1] == 1

            # activation
            if model.ctrl_activs[idx] == 'tanh':
                c = np.atleast_2d([dreal.tanh(v[0]) for v in chat]).T
            elif model.ctrl_activs[idx] == 'pow2':
                c = np.atleast_2d([dreal.pow(v[0], 2) for v in chat]).T
            elif model.ctrl_activs[idx] == 'sfpl':
                c = np.atleast_2d([
                    1. / parameters['beta_sfpl'] * dreal.log(1. + dreal.exp(parameters['beta_sfpl'] * v[0]))
                    for v in chat
                ]).T
            elif model.ctrl_activs[idx] == 'linear':
                c = chat
            else:
                raise ValueError(f'Not Implemented Control Activation Function {model.ctrl_activs[idx]}.')

        # Implementing saturation
        if parameters['use_saturation']:
            c2 = np.atleast_2d([dreal.tanh(v[0]) for v in chat]).T
            for jC in range(chat.size):
                c[jC] = parameters['ctrl_sat'][jC]*c2[jC]
        else:
            c = c

    # define control actions as a list for compatibility with f_symb symbolic function
    u = [v[0] for v in c]

    # Computing symbolic dynamics
    f_out_sym = f_symb(vars_, u, parameters)

    return u, V, f_out_sym



def translator_efficiency_loss(
            parameters,
            vars_, eff_,
            model, f_symb):
    """
    translates the torch network into a symbolic expression for dReal
    :param parameters:
    :param vars_:
    :param model:
    :return:
    """

    # Lyapunov ANN
    z = np.atleast_2d(np.array(vars_)).T
    for idx in range(len(model.layers)):

        # linear pass
        if model.lyap_bias[idx]:
            zhat = model.layers[idx].weight.detach().numpy() @ z \
                   + np.atleast_2d(model.layers[idx].bias.detach().numpy()).T
        else:
            zhat = model.layers[idx].weight.detach().numpy() @ z

        # dimension check
        assert zhat.shape[0] == 1 or zhat.shape[1] == 1

        # activation
        if model.activs[idx] == 'tanh':
            z = np.atleast_2d([dreal.tanh(v[0]) for v in zhat]).T
        elif model.activs[idx] == 'pow2':
            z = np.atleast_2d([dreal.pow(v[0], 2) for v in zhat]).T
        elif model.activs[idx] == 'sfpl':
            z = np.atleast_2d([
                1. / parameters['beta_sfpl'] * dreal.log(1. + dreal.exp(parameters['beta_sfpl'] * v[0]))
                for v in zhat
            ]).T
        elif model.activs[idx] == 'linear':
            z = zhat
        else:
            raise ValueError(f'Not Implemented Activation Function {model.activs[idx]}.')

    # V is contained in element [0,0] of the np.array
    V = z[0, 0]

    # Candidate control function
    if parameters['use_lin_ctr']:

        # linear control network
        c = np.atleast_2d(np.array(vars_)).T
        if parameters['lin_contr_bias']:
            chat = model.control.weight.detach().numpy() @ c \
                   + np.atleast_2d(model.control.bias.detach().numpy()).T
        else:
            chat = model.control.weight.detach().numpy() @ c
        c = chat
        # raise ValueError('Not Implemented translator for linear control')


    else:
        # nonlinear control network
        c = np.atleast_2d(np.array(vars_)).T
        for idx in range(len(model.ctrl_layers)):
            # linear pass
            if model.ctrl_bias[idx]:
                chat = model.ctrl_layers[idx].weight.detach().numpy() @ c \
                       + np.atleast_2d(model.ctrl_layers[idx].bias.detach().numpy()).T
            else:
                chat = model.ctrl_layers[idx].weight.detach().numpy() @ c

            # dimension check
            assert chat.shape[0] == 1 or chat.shape[1] == 1

            # activation
            if model.ctrl_activs[idx] == 'tanh':
                c = np.atleast_2d([dreal.tanh(v[0]) for v in chat]).T
            elif model.ctrl_activs[idx] == 'pow2':
                c = np.atleast_2d([dreal.pow(v[0], 2) for v in chat]).T
            elif model.ctrl_activs[idx] == 'sfpl':
                c = np.atleast_2d([
                    1. / parameters['beta_sfpl'] * dreal.log(1. + dreal.exp(parameters['beta_sfpl'] * v[0]))
                    for v in chat
                ]).T
            elif model.ctrl_activs[idx] == 'linear':
                c = chat
            else:
                raise ValueError(f'Not Implemented Control Activation Function {model.ctrl_activs[idx]}.')

    # define control actions as a list for compatibility with f_symb symbolic function
    u = [v[0] for v in c]

    # Computing symbolic dynamics
    f_out_sym = f_symb(vars_, eff_, u, parameters)

    return u, V, f_out_sym
