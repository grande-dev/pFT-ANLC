#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 19:02:40 2023

@authors: Andrea Peruffo
          Davide Grande
          
A script containing the postprocessing functions to be executed following 
each training run completion.
"""

import datetime
import torch
import numpy as np
import timeit
import os
import torch.nn.functional as F
import postprocessing.plot_and_save as plot_and_save
import matplotlib.pyplot as plt
import postprocessing.plot_2D as plot_2D
import postprocessing.plot_3D as plot_3D
import postprocessing.plot_4D as plot_4D
import postprocessing.plot_dataset_fun as plot_dataset_fun
from matplotlib import cm
import logging
import re
from mpl_toolkits.mplot3d import Axes3D
from utilities.falsifier import compute_symbolic
import utilities.Functions as Functions
import utilities.from_dreal_to_np as from_dreal_to_np
import dreal as dreal


def Plot3D(X, Y, V, r):
    # Plot 3D Lyapunov functions
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X,Y,V, rstride=5, cstride=5, alpha=0.5, cmap=cm.winter)
    ax.contour(X,Y,V,10, zdir='z', offset=0, cmap=cm.winter)

    cb = plt.colorbar(surf, pad=0.2)

    return ax


def init_history_arrays(parameters, max_iters):
    start_ = timeit.default_timer()
    # instantiating variables
    dim1 = max_iters
    x_0 = torch.zeros([1, parameters['n_input']])
    size_ctrl_layers = parameters['size_ctrl_layers']
    size_layers = parameters['size_layers']


    wc1_hist = np.empty((dim1, size_ctrl_layers[0], parameters['n_input']))  # 1st control layer weight
    q_hist = np.empty((dim1, size_ctrl_layers[-1], parameters['n_input']))
    w1_hist = np.empty((dim1, size_layers[0], parameters['n_input']))

    Lyap_risk_ELR_hist = np.empty(dim1)
    Lyap_risk_SLR_hist = np.empty(dim1)

    V_dot_hist = np.empty(dim1)
    V_hist = np.empty(dim1)
    V0_hist = np.empty(dim1)

    V_tuning_hist = np.empty(dim1)

    learning_rate_history = np.empty(dim1)
    learning_rate_c_history = np.empty(dim1)

    ce_found = np.empty((dim1, parameters['n_input']))
    disc_viol_found = np.empty((dim1, parameters['n_input']))

    # Assigning vectors values to NaN
    wc1_hist[:] = np.NaN
    q_hist[:] = np.NaN
    w1_hist[:] = np.NaN
    Lyap_risk_ELR_hist[:] = np.NaN
    Lyap_risk_SLR_hist[:] = np.NaN
    V_dot_hist[:] = np.NaN
    V_hist[:] = np.NaN
    V0_hist[:] = np.NaN
    V_tuning_hist[:] = np.NaN
    ce_found[:] = np.NaN
    learning_rate_history[:] = np.NaN
    learning_rate_c_history[:] = np.NaN
    disc_viol_found[:] = np.NaN

    start = timeit.default_timer()
    t_falsifier = 0.

    # campaign-specific variables
    tot_iters = 0  # total number of iterations
    to_fals = False  # time out falsifier
    to_learner = False  # time out learner
    stop_ = timeit.default_timer()

    init_time = stop_ - start_

    history_arrays = {'Lyap_risk_SLR_hist': Lyap_risk_SLR_hist, 
                      'Lyap_risk_ELR_hist': Lyap_risk_ELR_hist,
                      'V_hist': V_hist,
                      'V_dot_hist': V_dot_hist,
                      'V0_hist': V0_hist,
                      'V_tuning_hist': V_tuning_hist,
                      'ce_found': ce_found,
                      'learning_rate_history': learning_rate_history,
                      'learning_rate_c_history': learning_rate_c_history,
                      'disc_viol_found': disc_viol_found,
                      'w1_hist': w1_hist,
                      'wc1_hist': wc1_hist,
                      'q_hist': q_hist}

    return history_arrays


def save_cost_funct(history_arrays, 
                    parameters,
                    out_iters, max_iters, i_epoch,
                    Lyap_risk_SLR, Lyap_risk_ELR, 
                    V_candidate, Lie_V, V0, Circle_Tuning, model):
   
    # reading dictionary values
    Lyap_risk_ELR_hist = history_arrays['Lyap_risk_ELR_hist']
    Lyap_risk_SLR_hist = history_arrays['Lyap_risk_SLR_hist']
    V_hist = history_arrays['V_hist']
    V_dot_hist = history_arrays['V_dot_hist']
    V0_hist = history_arrays['V0_hist']
    V_tuning_hist = history_arrays['V_tuning_hist']
    
    Lyap_risk_ELR_hist[out_iters * max_iters + i_epoch] = \
        Lyap_risk_ELR.item()
    Lyap_risk_SLR_hist[out_iters * max_iters + i_epoch] = \
        Lyap_risk_SLR.item()
    V_hist[out_iters * max_iters + i_epoch] = \
        parameters['alpha_1'] * F.relu(-V_candidate).sum()
    V_dot_hist[out_iters * max_iters + i_epoch] = \
        parameters['alpha_2'] * F.relu(Lie_V).sum()
    V0_hist[out_iters * max_iters + i_epoch] = \
        parameters['alpha_3'] * (V0).pow(2)
    V_tuning_hist[out_iters * max_iters + i_epoch] = \
        parameters['alpha_4'] * ((Circle_Tuning - parameters['alpha_roa'] * (V_candidate)).pow(2)).mean()

    # updating dictionary
    history_arrays['Lyap_risk_ELR_hist'] = Lyap_risk_ELR_hist
    history_arrays['Lyap_risk_SLR_hist'] = Lyap_risk_SLR_hist
    history_arrays['V_hist'] = V_hist
    history_arrays['V_dot_hist'] = V_dot_hist
    history_arrays['V0_hist'] = V0_hist
    history_arrays['V_tuning_hist'] = V_tuning_hist

    ## Bonus: saving ANN weight evolution
    w1_hist = history_arrays['w1_hist']
    wc1_hist = history_arrays['wc1_hist']
    q_hist = history_arrays['q_hist']

    w1_hist[out_iters * max_iters + i_epoch] = model.layers[0].weight.data.numpy()
    wc1_hist[out_iters * max_iters + i_epoch] = model.ctrl_layers[0].weight.data.numpy()
    q_hist[out_iters * max_iters + i_epoch] = model.control.weight.data.numpy()

    # updating dictionary
    history_arrays['w1_hist'] = w1_hist
    history_arrays['wc1_hist'] = wc1_hist
    history_arrays['q_hist'] = q_hist

    return history_arrays


def save_lr_values(history_arrays, optimizer, out_iters, max_iters, i_epoch):
    
    # reading dictionary values
    learning_rate_history = history_arrays['learning_rate_history']
    learning_rate_c_history = history_arrays['learning_rate_c_history']
    
    learning_rate_history[out_iters * max_iters + i_epoch] = \
        optimizer.param_groups[0]["lr"]
    learning_rate_c_history[out_iters * max_iters + i_epoch] = \
        optimizer.param_groups[1]["lr"]

    # updating dictionary
    history_arrays['learning_rate_history'] = learning_rate_history
    history_arrays['learning_rate_c_history'] = learning_rate_c_history
    
    return history_arrays


def save_model_params(model, parameters, final_dir_run):

    # Saving model parameters
    # Generating ANN result folder
    try:
        folder_results_ann = final_dir_run + "/ANN_params/"
        current_dir = os.getcwd()
        final_dir_ann = current_dir + "/" + folder_results_ann
        os.mkdir(folder_results_ann)
    
    except OSError:
        logging.error(f"\nCreation of the ANN params directory: \n{folder_results_ann} \nFAILED!!\n")
    else:
        print(f"\nANN params directory successfully created as: \n{folder_results_ann}\n")


    for j in range(len(model.layers)):
        wj = model.layers[j].weight.data   
        weight_j = "L_w" + str(j+1) + ".txt"
        np.savetxt(folder_results_ann + weight_j, wj, fmt="%s")
    
        if parameters['lyap_bias'][j]:
            bj = model.layers[j].bias.data   
            bias_j = "L_b" + str(j+1) + ".txt"
            np.savetxt(folder_results_ann + bias_j, bj, fmt="%s")
    
    for j in range(len(model.ctrl_layers)):
        wj = model.ctrl_layers[j].weight.data   
        weight_j = "NLC_w" + str(j+1) + ".txt"
        np.savetxt(folder_results_ann + weight_j, wj, fmt="%s")
    
        if parameters['ctrl_bias'][j]:
            bj = model.ctrl_layers[j].bias.data   
            bias_j = "NLC_b" + str(j+1) + ".txt"
            np.savetxt(folder_results_ann + bias_j, bj, fmt="%s")

    wj = model.control.weight.data  
    weight_j = "LC_w.txt"
    np.savetxt(folder_results_ann + weight_j, wj, fmt="%s")
    if parameters['lin_contr_bias']:
        bj = model.control.bias.data   
        bias_j = "LC_b.txt"
        np.savetxt(folder_results_ann + bias_j, bj, fmt="%s")
        
        
def plot_loss_function(final_dir_plots, parameters, history_arrays):
    # Plot loss function
    try:
        folder_loss = final_dir_plots + "/loss_function/"
        os.mkdir(folder_loss)
    except OSError:
        logging.error("Creation of the 'Loss function result' directory %s failed" % folder_loss)
    else:
        print("'Loss function result' directory successfully created as: \n %s \n" % folder_loss)
    
    Lyap_risk_ELR_hist = history_arrays['Lyap_risk_ELR_hist']
    Lyap_risk_SLR_hist = history_arrays['Lyap_risk_SLR_hist']
    
    plot_and_save.plot_and_save(data=Lyap_risk_ELR_hist,
                                save_figures=True,
                                save_fig_folder=folder_loss,
                                save_fig_file="lyapunov_risk_ELR.png",
                                title=None, 
                                xlabel='Epochs',
                                ylabel=None,
                                caption=None)
    
    plot_and_save.plot_and_save(data=Lyap_risk_SLR_hist,
                                save_figures=True,
                                save_fig_folder=folder_loss,
                                save_fig_file="lyapunov_risk_SLR.png",
                                title=None, 
                                xlabel='Epochs',
                                ylabel=None,
                                caption=None)
    
    
    # Plot loss functions comparison
    plt.figure()
    plt.plot(Lyap_risk_ELR_hist, label='$L_{ELR}$')
    plt.plot(Lyap_risk_SLR_hist, label='$L_{SLR}$')
    plt.xlabel('Epochs')
    plt.ylabel(None)
    save_fig_file="Lyapunov_risk_comparison.png"
    save_fig_folder=folder_loss
    plt.grid()
    plt.legend(loc='best')
    plt.savefig(save_fig_folder + save_fig_file, dpi=parameters['dpi_'])
    plt.close()


    # Plot loss functions comparison -- logscale
    plt.figure()
    plt.yscale("log")
    plt.plot(Lyap_risk_ELR_hist, label='log($L_{ELR})$')
    plt.plot(Lyap_risk_SLR_hist, label='log($L_{SLR})$')
    if parameters['enforce_CLF_shape']:
         plt.plot(Lyap_risk_SLR_hist*0 + parameters['tau_overbar'], label=r'log($\overline{\tau}$)')
    plt.xlabel('Epochs')
    plt.ylabel(None)
    save_fig_file="Lyapunov_risk_comparison_log.png"
    save_fig_folder=folder_loss
    plt.grid()
    plt.legend(loc='best')
    plt.savefig(save_fig_folder + save_fig_file, dpi=parameters['dpi_'])
    plt.close()


def plot_learning_rate(final_dir_plots, parameters, history_arrays):
    # Plot learning rates
    try:
        folder_loss = final_dir_plots + "/learning_rate/"
        os.mkdir(folder_loss)
    except OSError:
        logging.error("Creation of the 'learning rate plot' directory %s failed" % folder_loss)
    else:
        print("'learning rate plot' directory successfully created as: \n %s \n" % folder_loss)
        
    learning_rate_history = history_arrays['learning_rate_history']
    learning_rate_c_history = history_arrays['learning_rate_c_history']
    
    plot_and_save.plot_and_save(data=learning_rate_history,
                                save_figures=True,
                                save_fig_folder=folder_loss,
                                save_fig_file="learning_rate_lyapunov.png",
                                title=None, 
                                xlabel='Epochs',
                                ylabel=None,
                                caption=None)
    
    plot_and_save.plot_and_save(data=learning_rate_c_history,
                                save_figures=True,
                                save_fig_folder=folder_loss,
                                save_fig_file="learning_rate_control.png",
                                title=None, 
                                xlabel='Epochs',
                                ylabel=None,
                                caption=None)

def find_max_V(parameters, V_learn):
        ## Max V
        # inputs
        x1 = - parameters['gamma_overbar']
        x2 = parameters['gamma_overbar']

        expr_str = V_learn.to_string() 
        expr_sub = from_dreal_to_np.sub(expr_str)  # substitute dreal functions
        expr_eval = eval(expr_sub) 

        max_V = np.max(expr_eval)

        return max_V


def find_V_init(parameters, V_learn):
        
        # set the points exactly at epsilon distance from the origin
        x1 = parameters['epsilon']
        x2 = 0.0

        expr_str = V_learn.to_string() 
        expr_sub = from_dreal_to_np.sub(expr_str)  # substitute dreal functions
        V_at_epsilon = eval(expr_sub) 

        return V_at_epsilon


def plot_3D_functions(final_dir_plots, parameters, name_folder, name_function, sym_function,
                      gamma_overbar):

    try:
        folder_loss = final_dir_plots + f"/{name_folder}/"
        os.mkdir(folder_loss)
    except OSError:
        logging.error(f"Creation of the '{name_folder}' plot directory %s failed" % folder_loss)
    else:
        print(f"'{name_folder}' directory successfully created as: \n %s \n" % folder_loss)

    if (parameters['n_input'] == 2):

        title = f"{name_function}"  
        plot_2D.plot(sym_function, parameters['n_points_3D'], gamma_overbar, 
                     title, False,
                     folder_loss, parameters['dpi_'], parameters, Plot3D) 


    if (parameters['n_input'] == 3):

        title = f"{name_function}"  
        plot_3D.plot(sym_function, parameters['n_points_3D'], gamma_overbar, 
                     title, False,
                     folder_loss, parameters['dpi_'], parameters, Plot3D) 


    if (parameters['n_input'] == 4):

        title = f"{name_function}"  
        plot_4D.plot(sym_function, title, False, folder_loss, parameters, Plot3D) 


def plot_ann_weight(final_dir_plots, parameters, history_arrays):

    print("Plotting ANN weights...")

    # 3) Plotting ANN weights
    # Generating control ANN weight and bias result folder
    try:
        folder_loss = final_dir_plots + "/ANN_weight/"
        os.mkdir(folder_loss)
    except OSError:
        logging.error("Creation of the 'ANN plot' directory %s failed" % folder_loss)
    else:
        print("'ANN plot' directory successfully created as: \n %s \n" % folder_loss)
    
    
    w1_hist = history_arrays['w1_hist']
    wc1_hist = history_arrays['wc1_hist']
    q_hist = history_arrays['q_hist']
    
    if parameters['plot_ctr_weights']:
        if parameters['use_lin_ctr']:
            for i_p in range(q_hist.shape[1]):
              for j_p in range(q_hist.shape[2]):
                  plt.figure()
                  plt.plot(q_hist[:, i_p, j_p])
                  plt.xlabel('Epochs')
                  plt.grid()
                  name_fig = '/weight_lin_control_neuron_' + str(i_p) + '_to_' +\
                      str(j_p) + '.png'
                  plt.savefig(folder_loss + name_fig, dpi=parameters['dpi_'])
                  plt.close()
    
        else:
    
            for i_p in range(wc1_hist.shape[1]):
                for j_p in range(wc1_hist.shape[2]):
                    plt.figure()
                    plt.plot(wc1_hist[:, i_p, j_p])
                    plt.xlabel('Epochs')
                    plt.grid()
                    name_fig = '/weight_nonlin_control_layer_1_neuron_' + str(i_p) + '_to_' +\
                        str(j_p) + '.png'
                    plt.savefig(folder_loss + name_fig, dpi=parameters['dpi_'])
                    plt.close()
     
    if parameters['plot_V_weights']:
        for i_p in range(w1_hist.shape[1]):
            for j_p in range(w1_hist.shape[2]):
               plt.figure()
               plt.plot(w1_hist[:, i_p, j_p])
               plt.xlabel('Epochs')
               plt.grid()
               name_fig = '/weight_lyapunov_layer_1_neuron_' + str(i_p) + '_to_' +\
                   str(j_p) + '.png'
               plt.savefig(folder_loss + name_fig, dpi=parameters['dpi_'])
               plt.close()    

def plot_dataset(parameters, folder_results_plots, x_dataset_init, x, 
                 gamma_overbar):    #ce_found


    # Generating CEs result folder
    try:
        folder_ds_plots = folder_results_plots + "/dataset_plots"
        os.mkdir(folder_ds_plots)
    except OSError:
        logging.error("Creation of the dataset plot result directory %s failed" % folder_ds_plots)
    else:
        print("Plot dataset directory successfully created as: \n %s \n" % folder_ds_plots)
    
    if parameters['n_input']==2:
        plot_dataset_fun.plot_dataset_dim2(x_dataset_init, x, folder_ds_plots, parameters)

    elif parameters['n_input']==3:
        plot_dataset_fun.plot_dataset_dim3(x_dataset_init, x, folder_ds_plots, parameters)

    elif parameters['n_input']==4:
        plot_dataset_fun.plot_dataset_dim4(x_dataset_init, x, folder_ds_plots, parameters)

    else:
        raise ValueError(f'Not implemented plotting of dataset function.')

def save_expressions(final_dir_run, V_learn, lie_derivative_of_V, u_learn, f_out_sym,
                     parameters):
    # Saving ANN symbolic expressions
    try:
        folder_expr = final_dir_run + "/expressions"
        os.mkdir(folder_expr)
    except OSError:
        logging.error("Creation of the 'Expressions' directory %s failed" % folder_expr)
    else:
        print("'Expressions' directory successfully created as: \n %s \n" % folder_expr)

    np.savetxt(folder_expr + "/Expression_V.txt", [V_learn.Expand().to_string()], fmt="%s")
    np.savetxt(folder_expr + "/Expression_Vdot.txt", [lie_derivative_of_V.Expand().to_string()], fmt="%s")

    for iS in range(parameters['n_input']):
        np.savetxt(folder_expr + "/Expression_f" + str(iS) + ".txt", [f_out_sym[iS].Expand().to_string()], fmt="%s")

    contr_out = parameters['size_ctrl_layers'][-1]
    for iC in range(contr_out):
        # retrieving control variables
        np.savetxt(folder_expr + "/Expression_u" + str(iC) + ".txt", [u_learn[iC].Expand().to_string()], fmt="%s")

def run_closed_loop_tests(parameters, x_star, final_dir_plots, 
                          cl, model, dynamic_sys, 
                          gamma_overbar, epsilon):


    samples_number = int(parameters['end_time'] / parameters['Dt'])

    if parameters['n_input'] == 2:

        # # Setting up three closed-loop tests
        initial_x1 = [gamma_overbar,     gamma_overbar/8, x_star[0].item(),             0.05, x_star[0].item(),              0.1,              0.0,          epsilon*2]
        initial_x2 = [gamma_overbar,     gamma_overbar/8, x_star[1].item(),            -0.05, x_star[1].item(),              0.0,              0.0,         -epsilon*2]
        desired_x1 = [x_star[0].item(), x_star[0].item(), x_star[0].item(), x_star[0].item(), x_star[0].item(), x_star[0].item(), x_star[0].item(),   x_star[0].item()]
        desired_x2 = [x_star[1].item(), x_star[1].item(), x_star[1].item(), x_star[1].item(), x_star[1].item(), x_star[1].item(), x_star[1].item(),   x_star[1].item()]
        control_active = [1,                           1,                1,                1,                0,                1,                1,                  1]

        # Creating closed-loop folder
        final_dir_test = final_dir_plots + "/closed_loop_tests/"
        os.mkdir(final_dir_test)

        for iTest in range(len(initial_x1)):
            message = "Closed-loop " + str(iTest + 1) + "/" + str(len(initial_x1))
            print(message)

            final_dir_ = final_dir_test + "test_#" + str(iTest) + "/"
            os.mkdir(final_dir_)

            des_x1 = desired_x1[iTest]
            des_x2 = desired_x2[iTest]
            init_x1 = initial_x1[iTest]
            init_x2 = initial_x2[iTest]
            control_active_test = control_active[iTest]

            # closed-loop test callback
            cl.closed_loop_system(samples_number, model,
                                  des_x1, des_x2,
                                  control_active_test,
                                  init_x1, init_x2,
                                  parameters,
                                  final_dir_,
                                  dynamic_sys,
                                  gamma_overbar, epsilon)

            # saving test report
            result_report = [f"Initial x1 = {init_x1} [m/s]\
                             \nInitial x2 = {init_x2} [m/s]\
                             \nDesired x1 = {des_x1} [m/s]\
                             \nDesired x2 = {des_x2} [m/s]\
                             \nIs control active = {control_active_test}\
                             "]
            np.savetxt(final_dir_ + "test_report.txt", result_report, fmt="%s")

    elif parameters['n_input'] == 3:

        # Setting up three closed-loop tests
        initial_x1 = [gamma_overbar, 0.0, -gamma_overbar / 2, gamma_overbar]
        initial_x2 = [gamma_overbar, 0.0, -gamma_overbar / 2, gamma_overbar]
        initial_x3 = [gamma_overbar, 0.0, -gamma_overbar / 2, gamma_overbar]
        desired_x1 = [x_star[0].item(), x_star[0].item(), x_star[0].item(), x_star[0].item()]
        desired_x2 = [x_star[1].item(), x_star[1].item(), x_star[1].item(), x_star[1].item()]
        desired_x3 = [x_star[2].item(), x_star[2].item(), x_star[2].item(), x_star[2].item()]
        control_active = [1, 1, 1, 0]

        # Creating closed-loop folder
        final_dir_test = final_dir_plots + "/closed_loop_tests/"
        os.mkdir(final_dir_test)

        for iTest in range(len(initial_x1)):
            message = "Closed-loop " + str(iTest + 1) + "/" + str(len(initial_x1))
            print(message)

            final_dir_ = final_dir_test + "test_#" + str(iTest) + "/"
            os.mkdir(final_dir_)

            des_x1 = desired_x1[iTest]
            des_x2 = desired_x2[iTest]
            des_x3 = desired_x3[iTest]
            init_x1 = initial_x1[iTest]
            init_x2 = initial_x2[iTest]
            init_x3 = initial_x3[iTest]
            control_active_test = control_active[iTest]

            # closed-loop test callback
            cl.closed_loop_system(samples_number, model,
                                  des_x1, des_x2, des_x3,
                                  control_active_test,
                                  init_x1, init_x2, init_x3,
                                  parameters,
                                  final_dir_,
                                  dynamic_sys,
                                  gamma_overbar, epsilon)

            # saving test report
            result_report = [f"Initial x1 = {init_x1}\
                             \nInitial x2 = {init_x2}\
                             \nInitial x3 = {init_x3}\
                             \nDesired x1 = {des_x1}\
                             \nDesired x2 = {des_x2}\
                             \nDesired x3 = {des_x3}\
                             \nIs control active = {control_active_test}\
                             "]
            np.savetxt(final_dir_ + "test_report.txt", result_report, fmt="%s")


    elif parameters['n_input'] == 4:
    
        # Setting up the closed-loop tests
        initial_x1 = [0.2, 0.2]
        initial_x2 = [np.deg2rad(-35.0), np.deg2rad(-35.0)]
        initial_x3 = [np.deg2rad(1.0), np.deg2rad(1.0)]
        initial_x4 = [0., 0.]
        desired_x1 = [x_star[0].item(), x_star[0].item()]
        desired_x2 = [x_star[1].item(), x_star[1].item()]
        desired_x3 = [x_star[2].item(), x_star[2].item()]
        desired_x4 = [x_star[3].item(), x_star[3].item()]
        control_active = [0, 1]

        # Creating closed-loop folder
        final_dir_test = final_dir_plots + "/closed_loop_tests/"
        os.mkdir(final_dir_test)
    
        for iTest in range(len(initial_x1)):
            message = "Closed-loop " + str(iTest + 1) + "/" + str(len(initial_x1))
            print(message)
    
            final_dir_ = final_dir_test + "test_#" + str(iTest) + "/"
            os.mkdir(final_dir_)
    
            des_x1 = desired_x1[iTest]
            des_x2 = desired_x2[iTest]
            des_x3 = desired_x3[iTest]
            des_x4 = desired_x4[iTest]
            init_x1 = initial_x1[iTest]
            init_x2 = initial_x2[iTest]
            init_x3 = initial_x3[iTest]
            init_x4 = initial_x4[iTest]
            control_active_test = control_active[iTest]
    
            # closed-loop test callback
            cl.closed_loop_system(samples_number, model,
                                  des_x1, des_x2, des_x3, des_x4,
                                  control_active_test,
                                  init_x1, init_x2, init_x3, init_x4,
                                  parameters,
                                  final_dir_,
                                  dynamic_sys,
                                  gamma_overbar, epsilon)
    
            # saving test report
            result_report = [f"Initial x1 = {init_x1}\
                             \nInitial x2 = {init_x2}\
                             \nInitial x3 = {init_x3}\
                             \nInitial x4 = {init_x4}\
                             \nDesired x1 = {des_x1}\
                             \nDesired x2 = {des_x2}\
                             \nDesired x3 = {des_x3}\
                             \nDesired x4 = {des_x4}\
                             \nIs control active = {control_active_test}\
                             "]
            np.savetxt(final_dir_ + "test_report.txt", result_report, fmt="%s")


    else:       
        sys_dim = parameters['n_input']
        raise ValueError(f'Not implemented plotting of closed-loop function for system of dimension {sys_dim}.')


def postprocessing(final_dir_campaign, i_loop, parameters, x,
                   model, cl, dynamic_sys,
                   history_arrays,
                   gamma_overbar, epsilon,
                   u_learn, V_learn, lie_derivative_of_V, f_out_sym,
                   u_inter_hist, V_inter_hist, lie_der_inter_hist, lie_der_inter_hist_tot,
                   x_dataset_init, found_lyap_f, 
                   vars_, f_symb, config):

    # 0) Generating result folder
    try:
        final_dir_run = final_dir_campaign + "/" + str(i_loop)
        os.mkdir(final_dir_run)
    except OSError:
        logging.error("Creation of the result directory %s failed" % final_dir_run)
    else:
        print("Result directory successfully created as: \n %s \n" % final_dir_run)

    # Generating figures result folder
    try:
        final_dir_plots = final_dir_run + "/figures"
        os.mkdir(final_dir_plots)
    except OSError:
        logging.error("Creation of the plot result directory %s failed" % final_dir_plots)
    else:
        print("Plot result directory successfully created as: \n %s \n" % final_dir_plots)


    # Saving symbolic expressions
    save_expressions(final_dir_run, V_learn, lie_derivative_of_V, u_learn, f_out_sym,
                     parameters)

    # Generating plots
    save_model_params(model, parameters, final_dir_run)

    plot_loss_function(final_dir_plots, parameters, history_arrays)

    plot_learning_rate(final_dir_plots, parameters, history_arrays)


    if parameters['plot_ctr_weights'] or parameters['plot_V_weights']:
        plot_ann_weight(final_dir_plots, parameters, history_arrays)

    if parameters['plot_Vdot']:
        
        try:
            # System with faults                       
            # this plot the last value of the Lie der
            for j_fault in range(len(parameters['phi_i'])):
                # retrieving current values of actuators health status
                for k_ctr in range(parameters['size_ctrl_layers'][-1]):
                    # automatic definition of the variables
                    str_h = "h" + str(k_ctr+1)
                    parameters[str_h] = parameters['phi_i'][j_fault][k_ctr]

                u_, V_, lie_derivative_  = compute_symbolic(parameters, vars_, model, f_symb)

                plot_3D_functions(final_dir_plots, parameters, 'Lie_der_function'+str(j_fault), 
                                '$\dot{V}$', lie_derivative_,
                                gamma_overbar)

                hh1 = parameters['h1']
                hh2 = parameters['h2']
                result_report = [f"h1 = {hh1}\
                                \nh2 = {hh2}\
                                "]
                np.savetxt(final_dir_plots + "/Lie_der_function"+str(j_fault) + "/test_report.txt", result_report, fmt="%s")

                np.savetxt(final_dir_plots + "/Lie_der_function"+str(j_fault) + "/Expression_Vdot.txt", 
                            [lie_derivative_.Expand().to_string()], fmt="%s")

        except:        
            plot_3D_functions(final_dir_plots, parameters, 'Lie_der_function', 
                            '$\dot{V}$', lie_derivative_of_V,
                            gamma_overbar)
            
    if parameters['plot_V']:
        plot_3D_functions(final_dir_plots, parameters, 'V_function', 'V', V_learn,
                        gamma_overbar)

    if parameters['plot_u']:
        print("Plotting control function ... ")    

        contr_out = parameters['size_ctrl_layers'][-1]
        for iC in range(contr_out):
            plot_3D_functions(final_dir_plots, parameters, f'Control function {iC}', 
                              f'$u_{iC}$', u_learn[iC],
                              gamma_overbar)


    if parameters['plot_dataset']:

        plot_dataset(parameters, final_dir_plots, 
                     x_dataset_init, x,
                     gamma_overbar) #ce_found, 


    # Perform closed loop (convergence) tests
    if (parameters['test_closed_loop_dynamics']):

        if found_lyap_f or (not found_lyap_f and not parameters['skip_closed_loop_unconverged']):
            print("Run closed-loop test.")
            run_closed_loop_tests(parameters, parameters['x_star'], final_dir_plots, 
                                cl, model, dynamic_sys, 
                                parameters['gamma_overbar'], parameters['epsilon'])

    return final_dir_run

