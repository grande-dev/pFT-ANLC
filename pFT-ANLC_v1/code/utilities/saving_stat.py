#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:30:41 2023

@authors: Davide Grande
          Andrea Peruffo
      
A function that saves the statistics of the training campaign.
      
"""

import numpy as np

def gen_stat(parameters,  
             seed_, minutes_elapsed_postp, time_elapse_hist, 
             iterations_hist, conv_hist, to_learner_hist, to_fals_hist, 
             to_fals_check, count_conv, final_dir_campaign):

    print("Saving campaign statistics ...\n")
    

    # Saving statistic
    result_stat = [f"Run of the controlled '{parameters['system_name']}' system campaign number {parameters['campaign_run']}.\n" +
                   f"The control weights were intialised = {parameters['control_initialised']}\n" +
                   f"\n{parameters['tot_runs']} tests were run\n" +
                   f"The seeds were cycled from {parameters['init_seed']} to {seed_}.\n" +
                   "\nThe overall time for the statistic generation was: " +
                   f"{minutes_elapsed_postp} [']" +
                   f"\n\nRun time (per test) (mu+-3sigma) = " +
                   f"{time_elapse_hist.mean()}+-{3 * time_elapse_hist.std()} ['']" +
                   f"\nNumber of iterations (per test) (mu+-3sigma) = " +
                   f"{iterations_hist.mean()}+-{3 * iterations_hist.std()}" +
                   f"\nConvergence history = {conv_hist}" +
                   f"\nTO learner history = {to_learner_hist}" +
                   f"\nTO falsifier history = {to_fals_hist}" +
                   f"\nTO falsifier check [s] = {to_fals_check}" +
                   f"\n\nConverged tests = {count_conv}/{parameters['tot_runs']}" +
                   f"\n\nElapsed time history = {time_elapse_hist} ['']" +
                   f"\nIteration history = {iterations_hist}" +
                   f"\n\nTotal Learner not-converged = {int(to_learner_hist.sum())}" +
                   f"\nTotal Falsifier not-converged = {int(to_fals_hist.sum())}"
                   ]
    
    np.savetxt(final_dir_campaign + "/statistics.txt", result_stat, fmt="%s")


def info_faults(parameters, final_dir_campaign):
    # Saving the fault configurations employed during the training
    
    result_report = [f"Actuators health stati configurations: \n{parameters['phi_i']}\n"]

    np.savetxt(final_dir_campaign + "/faults_info.txt", result_report, fmt="%s")