#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:05:22 2023

@authors: Davide Grande
          Andrea Peruffo

Plotting and saving a (generic) dataset.
"""
import matplotlib.pyplot as plt

def plot_dataset_dim2(x_dataset_init, x, folder_ds_plots, parameters):
    # Plot dataset for a 3-dimensional system
    # The plots are produced for the touples (x1, x2, x3)
        
    gamma_overbar = parameters['gamma_overbar']

    # plot initial dataset
    plt.figure()
    plt.scatter(x_dataset_init[:,0], x_dataset_init[:,1], c='limegreen', s=15) 
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid()
    plt.xlim(-gamma_overbar, gamma_overbar)
    plt.ylim(-gamma_overbar, gamma_overbar)
    name_fig = "dataset_init"
    plt.savefig(folder_ds_plots + '/' +name_fig+".png", dpi=parameters['dpi_'])
    if parameters['save_pdf']:
        plt.savefig(folder_ds_plots + '/' +name_fig+".pdf", format='pdf')
    plt.close()
    
    
    # plot final dataset
    plt.figure()
    plt.scatter(x[:,0], x[:,1], s=15) 
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid()
    name_fig = "dataset_final"
    plt.savefig(folder_ds_plots + '/' +name_fig+".png", dpi=parameters['dpi_'])
    if parameters['save_pdf']:
        plt.savefig(folder_ds_plots + '/' +name_fig+".pdf", format='pdf')
    plt.close()
    
    # plot final dataset (sum of CE and initial)
    plt.figure()
    ax = plt.subplot(111)
    #plt.scatter(ce_found[:,0], ce_found[:,1], s=15, label='$CE_{SMT}$')  
    plt.scatter(x[:,0], x[:,1], s=15, label='$CEs$') 
    plt.scatter(x_dataset_init[:,0], x_dataset_init[:,1], c='limegreen', s=15, label='Intial dataset') 
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid()
    plt.legend(bbox_to_anchor=(-0.02, 1.02), loc="lower left", ncol=3)
    plt.xlim(-gamma_overbar, gamma_overbar)
    plt.ylim(-gamma_overbar, gamma_overbar)
    name_fig = "dataset_init_plus_CEs"
    plt.savefig(folder_ds_plots + '/' +name_fig+".png", dpi=parameters['dpi_'])
    if parameters['save_pdf']:
        plt.savefig(folder_ds_plots + '/' +name_fig+".pdf", format='pdf')
    plt.close()


def plot_dataset_dim3(x_dataset_init, x, folder_ds_plots, parameters):
    # Plot dataset for a 3-dimensional system
    # The plots are produced for the touples (x1, x2, x3)

    gamma_overbar = parameters['gamma_overbar']

    # Plot initial dataset
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    #plt.title('Counter examples')
    ax.scatter3D(x_dataset_init[:,0], x_dataset_init[:,1], x_dataset_init[:,2], c='limegreen', s=15)
    name_fig = "dataset_init"
    plt.savefig(folder_ds_plots + '/' + name_fig + ".png", dpi=parameters['dpi_'])
    if parameters['save_pdf']:
        plt.savefig(folder_ds_plots + '/' + name_fig + ".pdf", format='pdf')
    plt.close()    


    # Plot final dataset (sum of CE and initial)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.set_xlim(-gamma_overbar, gamma_overbar)
    ax.set_ylim(-gamma_overbar, gamma_overbar)
    ax.set_zlim(-gamma_overbar, gamma_overbar)
    ax.scatter3D(x[:,0], x[:,1], x[:,2], s=15, label='$CEs$')
    ax.scatter3D(x_dataset_init[:, 0], x_dataset_init[:, 1], x_dataset_init[:, 2], 
                    c='limegreen', s=15, label='Intial points')
    ax.legend()
    name_fig = "dataset_init_plus_CEs"
    plt.savefig(folder_ds_plots + '/' + name_fig + ".png", dpi=parameters['dpi_'])
    if parameters['save_pdf']:
        plt.savefig(folder_ds_plots + '/' + name_fig + ".pdf", format='pdf')
    plt.close()


def plot_dataset_dim4(x_dataset_init, x, folder_ds_plots, parameters):
    # Plot dataset for a 4-dimensional system
    # The plots are produced for the touples (x1, x2, x3) and (x2, x3, x4), as this function serves mostly 
    # for debugging purposes.

    gamma_overbar = parameters['gamma_overbar']

    # Plot initial dataset (x1, x2, x3)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    #plt.title('Counter examples')
    ax.scatter3D(x_dataset_init[:,0], x_dataset_init[:,1], x_dataset_init[:,2], c='limegreen', s=15)
    name_fig = "dataset_init1"
    plt.savefig(folder_ds_plots + '/' + name_fig + ".png", dpi=parameters['dpi_'])
    if parameters['save_pdf']:
        plt.savefig(folder_ds_plots + '/' + name_fig + ".pdf", format='pdf')
    plt.close()    


    # Plot final dataset (sum of CE and initial) (x1, x2, x3)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.set_xlim(-gamma_overbar, gamma_overbar)
    ax.set_ylim(-gamma_overbar, gamma_overbar)
    ax.set_zlim(-gamma_overbar, gamma_overbar)
    ax.scatter3D(x[:,0], x[:,1], x[:,2], s=15, label='$CEs$')
    ax.scatter3D(x_dataset_init[:, 0], x_dataset_init[:, 1], x_dataset_init[:, 2], 
                    c='limegreen', s=15, label='Intial points')
    ax.legend()
    name_fig = "dataset_init_plus_CEs1"
    plt.savefig(folder_ds_plots + '/' + name_fig + ".png", dpi=parameters['dpi_'])
    if parameters['save_pdf']:
        plt.savefig(folder_ds_plots + '/' + name_fig + ".pdf", format='pdf')
    plt.close()


    # Plot initial dataset (x2, x3, x4)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('$x_2$')
    ax.set_ylabel('$x_3$')
    ax.set_zlabel('$x_4$')
    #plt.title('Counter examples')
    ax.scatter3D(x_dataset_init[:,1], x_dataset_init[:,2], x_dataset_init[:,3], c='limegreen', s=15)
    name_fig = "dataset_init2"
    plt.savefig(folder_ds_plots + '/' + name_fig + ".png", dpi=parameters['dpi_'])
    if parameters['save_pdf']:
        plt.savefig(folder_ds_plots + '/' + name_fig + ".pdf", format='pdf')
    plt.close()    


    # Plot final dataset (sum of CE and initial) (x2, x3, x4)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('$x_2$')
    ax.set_ylabel('$x_3$')
    ax.set_zlabel('$x_4$')
    ax.set_xlim(-gamma_overbar, gamma_overbar)
    ax.set_ylim(-gamma_overbar, gamma_overbar)
    ax.set_zlim(-gamma_overbar, gamma_overbar)
    ax.scatter3D(x[:,1], x[:,2], x[:,3], s=15, label='$CEs$')
    ax.scatter3D(x_dataset_init[:, 1], x_dataset_init[:, 2], x_dataset_init[:, 3], 
                    c='limegreen', s=15, label='Intial points')
    ax.legend()
    name_fig = "dataset_init_plus_CEs2"
    plt.savefig(folder_ds_plots + '/' + name_fig + ".png", dpi=parameters['dpi_'])
    if parameters['save_pdf']:
        plt.savefig(folder_ds_plots + '/' + name_fig + ".pdf", format='pdf')
    plt.close()

