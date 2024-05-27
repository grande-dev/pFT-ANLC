#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:20:47 2023

@authors: Davide Grande
          Andrea Peruffo

Given a function f(x1, x2, x3) --> R, this function
iteratively sets one of the three variables to zero, and returns the 3D plot
of the remaining two. 

"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import re
import random
import utilities.from_dreal_to_np as from_dreal_to_np
import postprocessing.plot_3D as plot_3D


def plot(sym_expr, title_str, plot_title, 
         folder_results_plots, parameters, Plot3D):
    
    if(parameters['n_points_3D']>1000):
        print("CAVEAT: the plot is being generated, please be patient ...")
    

    # inputs
    x_span = np.linspace(-parameters['gamma_overbar'], parameters['gamma_overbar'], parameters['n_points_3D']) 
    y_span = np.linspace(-parameters['gamma_overbar'], parameters['gamma_overbar'], parameters['n_points_3D'])
    z_span = np.linspace(-parameters['gamma_overbar'], parameters['gamma_overbar'], parameters['n_points_3D'])
    w_span = np.linspace(-parameters['gamma_overbar'], parameters['gamma_overbar'], parameters['n_points_3D'])

    expr_str = sym_expr.to_string() 
    expr_sub = from_dreal_to_np.sub(expr_str)  # substitute dreal functions

    # 1) removing x3, x4
    x1, x2 = np.meshgrid(x_span, y_span)
    try:
        out_str_x3 = re.sub(r'x3' , r'0', expr_sub)
        out_str_x34 = re.sub(r'x4' , r'0', out_str_x3)
    except:
        print("No x3, x4 variables")
    else:
        in_str_x3 = out_str_x34

    try:
        expr_eval_x3 = eval(out_str_x34)
        plot_3D.plot_dimG2(expr_eval_x3, x1, x2, parameters['n_points_3D'], title_str, plot_title, 'x3', 'x4',
                           folder_results_plots, parameters['dpi_'], parameters, Plot3D)
    except: 
        print(f"{title_str} not plotted due to numerical errors.")


    # 1) removing x1, x2
    x3, x4 = np.meshgrid(z_span, w_span)
    try:
        out_str_x3 = re.sub(r'x1' , r'0', expr_sub)
        out_str_x34 = re.sub(r'x2' , r'0', out_str_x3)
    except:
        print("No x1, x2 variables")
    else:
        in_str_x3 = out_str_x34

    try:
        expr_eval_x3 = eval(out_str_x34)

        plot_3D.plot_dimG2(expr_eval_x3, x3, x4, parameters['n_points_3D'], title_str, plot_title, 'x1', 'x2',
                        folder_results_plots, parameters['dpi_'], parameters, Plot3D)
    except: 
        print(f"{title_str} not plotted due to numerical errors.")
