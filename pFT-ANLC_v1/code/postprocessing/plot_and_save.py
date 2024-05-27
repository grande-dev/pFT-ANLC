#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 11:16:50 2021

@authors: Davide Grande
          Andrea Peruffo

Plotting and saving a (generic) dataset.
"""
import matplotlib.pyplot as plt

def plot_and_save(data, save_figures, save_fig_folder, save_fig_file, title, xlabel, ylabel, caption):

    plt.figure()
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if save_figures:
        plt.savefig(save_fig_folder + save_fig_file, dpi=300)
    # plt.show()
    plt.close()
