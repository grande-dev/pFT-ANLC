#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:59:18 2022

@authors: Davide Grande
          Andrea Peruffo
    
A script collecting functions to run the closed-loop tests upon training 
completion.

    
"""

import torch 
import matplotlib.pyplot as plt
import numpy as np
import math


#
# Closed-loop dynamics
#
def closed_loop_system(samples_number, model, 
                       des_x1, des_x2,
                       control_active_test, 
                       init_x1, init_x2,
                       parameters,
                       final_dir_,
                       dynamic_sys, 
                       gamma_overbar, epsilon):

    Dt = parameters['Dt']
    end_time = parameters['end_time']
    dpi_ = parameters['dpi_']
    D_in = parameters['n_input']
    
    # Reference values
    reference = torch.zeros(1, D_in)
    reference[0, 0] = des_x1
    reference[0, 1] = des_x2

    x_test_hist = []
    u_test_hist = []
    V_test_hist = []
    error_hist = []
    h1_hist = []

    # saving initial efficiecy 1 value 
    e1_init = parameters['h1']

    for iiter in range(samples_number):
        if (iiter == 0):
            # initial condition (x1, ...)
            x_0_test = torch.zeros(1, D_in)
            err_ref = torch.zeros(1, D_in)
            h1_hist = torch.zeros(1)

            x_0_test[0, 0] = init_x1
            x_0_test[0, 1] = init_x2
            h1_hist[0] = parameters['h1']

            x_test = x_0_test  # needed for next iteration step
            err_ref = x_0_test - reference
            V_test, outU = model.use_in_control_loop(err_ref)

            u_test = outU * control_active_test
            u_test[0, 0] = u_test[0, 0]*parameters['h1']

            x_test_hist = np.append(x_test_hist, x_0_test)
            V_test_hist = np.append(V_test_hist, V_test.detach())
            u_test_hist = np.append(u_test_hist, u_test.detach())
            error_hist = np.append(error_hist, err_ref)

        else:

            V_test, outU = model.use_in_control_loop(err_ref)
            u_test = outU * control_active_test
            u_test[0, 0] = u_test[0, 0]*parameters['h1']

            x_test_hist = np.vstack([x_test_hist, x_test.numpy()])
            V_test_hist = np.vstack([V_test_hist, V_test.detach()])
            u_test_hist = np.vstack([u_test_hist, u_test.detach()])
            error_hist = np.vstack([error_hist, err_ref])
            h1_hist = np.vstack([h1_hist, parameters['h1']]) 

        if (iiter/samples_number*100 < 20.0):
            parameters['h1'] = 1.0
        elif (iiter/samples_number*100 < 40.0):
            parameters['h1'] = 0.2
        elif (iiter/samples_number*100 < 60.0):
            parameters['h1'] = 0.0  
        else:
            parameters['h1'] = 0.6

        # forward dynamics iteration
        f_next = dynamic_sys.dyn(x_test.detach()[0], u_test.detach()[0], 
                                 Dt, parameters)  

        x_test = torch.zeros(1, D_in)
        for jIn in range(D_in):
            x_test[0,jIn] = f_next[0, jIn]

        err_ref = x_test - reference


    # restoring efficiency 1 value for the next test
    parameters['h1'] = e1_init

    # producing x-axis scale vector
    x_axis_scale = np.linspace(0, end_time, samples_number)

    size_ctrl_layers = parameters['size_ctrl_layers']

    title_fig_c = "Control_input_forces.png"
    fig = plt.figure()
    for iC in range(size_ctrl_layers[-1]):
        plt.plot(x_axis_scale, u_test_hist[:, iC], label=f'$u_{iC}$')
    plt.xlabel("Time [s]")
    plt.ylabel("Control effort")
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_c, dpi=dpi_)
    plt.close(fig)

    title_fig_v = "h1_value.png"
    fig = plt.figure()
    plt.plot(x_axis_scale, h1_hist*100)
    plt.xlabel("Time [s]")
    plt.ylabel("%")
    plt.grid()
    plt.savefig(final_dir_ + title_fig_v, dpi=dpi_)
    plt.close(fig)

    title_fig_v = "Lyapunov_value.png"
    fig = plt.figure()
    plt.plot(x_axis_scale, V_test_hist)
    plt.xlabel("Time [s]")
    plt.ylabel(None)
    plt.grid()
    plt.savefig(final_dir_ + title_fig_v, dpi=dpi_)
    plt.close(fig)


    title_fig_e = "State_error.png"
    fig = plt.figure()
    plt.plot(x_axis_scale, error_hist[:, 0], 'b', label='$err_{x_1}$')
    plt.plot(x_axis_scale, error_hist[:, 1], 'g', label='$err_{x_2}$')
    plt.xlabel("Time [s]")
    plt.ylabel(None)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_e, dpi=dpi_)
    plt.close(fig)


    title_fig_ref = "Reference_dynamics_x1.png"    
    fig = plt.figure()
    plt.plot(x_axis_scale, x_test_hist[:, 0], 'b', label='$x_1$')
    plt.plot(x_axis_scale, x_test_hist[:, 0]*0 + des_x1, '--r', label='$x_{1_{REF}}$')
    plt.fill_between(x_axis_scale, x_test_hist[:, 0]*0 + des_x1+epsilon, 
                        x_test_hist[:, 0]*0 + des_x1-epsilon, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')
    plt.xlabel("Time [s]")
    plt.ylabel(None)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)

    title_fig_ref = "Reference_dynamics_x2.png"    
    fig = plt.figure()
    plt.plot(x_axis_scale, x_test_hist[:, 1], 'b', label='$x_2$')
    plt.plot(x_axis_scale, x_test_hist[:, 1]*0 + des_x2, '--r', label='$x_{2_{REF}}$')
    plt.fill_between(x_axis_scale, x_test_hist[:, 1]*0 + des_x2+epsilon, 
                        x_test_hist[:, 1]*0 + des_x2-epsilon, facecolor='green', alpha=0.2, interpolate=True, label='$\epsilon$-stability bound')

    plt.xlabel("Time [s]")
    plt.ylabel(None)
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)
    

    title_fig_ref = "2d_vel_trajetory.png"    
    fig = plt.figure()
    plt.plot(x_test_hist[:, 0], x_test_hist[:, 1])
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.grid()
    plt.savefig(final_dir_ + title_fig_ref, dpi=dpi_)
    plt.close(fig)

