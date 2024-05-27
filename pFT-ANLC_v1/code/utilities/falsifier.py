#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 19:44:12 2022
@authors: Andrea Peruffo
          Davide Grande
          
"""
import dreal
import timeit
import logging
import utilities.Functions as Functions
import time
from utilities.translator import translator, translator_efficiency_loss


def augm_falsifier(parameters, vars_, model, f_symb,
                   epsilon,
                   gamma_overbar,
                   config,
                   x):

    found_lyap_f = False
    to_fals = False
    # Computing the system symbolic dynamics
    u_learn, V_learn, f_out_sym = translator(parameters, vars_, model, f_symb)

    print("\nDiscrete Falsifier computing CEs ...")
    lie_derivative_of_V = Functions.LieDerivative_v2(vars_, f_out_sym, V_learn, parameters)

    if parameters['n_input'] == 2: 
        x, disc_viol_found = \
            Functions.AddLieViolationsOrder2_v4(x,
                                                gamma_overbar,
                                                parameters['grid_points'],
                                                parameters['zeta_D'],
                                                parameters['verbose_info'],
                                                V_learn,
                                                lie_derivative_of_V)

    elif parameters['n_input'] == 3: 
        x, disc_viol_found = \
            Functions.AddLieViolationsOrder3_v4(x,
                                                gamma_overbar,
                                                parameters['grid_points'],
                                                parameters['zeta_D'],
                                                parameters['verbose_info'],
                                                V_learn,
                                                lie_derivative_of_V)

    elif parameters['n_input'] == 4: 
        x, disc_viol_found = \
            Functions.AddLieViolationsOrder4_v4(x,
                                                gamma_overbar,
                                                parameters['grid_points'],
                                                parameters['zeta_D'],
                                                parameters['verbose_info'],
                                                V_learn,
                                                lie_derivative_of_V)

    else:
        dimension_sys = parameters['n_input']
        logging.error(f'Not implemented Falsifier for system of order {dimension_sys}!')
        raise ValueError('Functionality to be implemented in: utilities/falsifier.py')

    # If no CE is found, invoke the SMT Falsifier
    if disc_viol_found == 0:

        print("\nSMT Falsifier computing CE ...")
        try:
            start_ = timeit.default_timer()

            # Using spherical domain
            CE_SMT, \
            lie_derivative_of_V = Functions.CheckLyapunov(vars_,
                                                        f_out_sym,
                                                        V_learn,
                                                        epsilon,
                                                        gamma_overbar,
                                                        config)

        except TimeoutError:
            logging.error("SMT Falsifier Timed Out")
            to_fals = True
            stop_ = timeit.default_timer()
            fals_to_check = stop_ - start_

        if not to_fals:
            if CE_SMT:
                # if a counterexample is found
                print("SMT found a CE: ")
                print(CE_SMT)

                # Adding midpoint of the CE_SMT to the history
                x = Functions.AddCounterexamples(x, CE_SMT, parameters['zeta_SMT'])
                if parameters['verbose_info']:
                    print(f"'SMT Falsifier': Added {parameters['zeta_SMT']} points in the vicinity of the CE.\n")

            else:
                # no CE_SMT is returned hence V_learn is a valid Lyapunov
                # function
                print("NO CE_SMT found.")
                found_lyap_f = True
                print("\nControl Lyapunov Function synthesised as:")
                print(V_learn.Expand())

    else:
        print(f"Skipping SMT callback.\n")

    stop_ = timeit.default_timer()

    print('================================')

    return x, u_learn, V_learn, lie_derivative_of_V, f_out_sym, found_lyap_f, to_fals


# Standalone Discrete Falsifier function, i.e. separated from the SMT.
# This function is used to retrieve the symbolic dynamics in Fault Tolerant Control applications.    
def disc_falsifier_fault(parameters, vars_, model, f_symb,
                         epsilon,
	           	         gamma_overbar,
	           	         config,
	           	         x):

    # Computing the system symbolic dynamics
    u_learn, V_learn, f_out_sym = translator(parameters, vars_, model, f_symb)

    print("\nDiscrete Falsifier computing CEs ...")
    lie_derivative_of_V = Functions.LieDerivative_v2(vars_, f_out_sym, V_learn, parameters)

    if parameters['n_input'] == 2: 
        x, disc_viol_found = \
            Functions.AddLieViolationsOrder2_v4(x,
                                                gamma_overbar,
                                                parameters['grid_points'],
                                                parameters['zeta_D'],
                                                parameters['verbose_info'],
                                                V_learn,
                                                lie_derivative_of_V)

    elif parameters['n_input'] == 3: 
        x, disc_viol_found = \
            Functions.AddLieViolationsOrder3_v4(x,
                                                gamma_overbar,
                                                parameters['grid_points'],
                                                parameters['zeta_D'],
                                                parameters['verbose_info'],
                                                V_learn,
                                                lie_derivative_of_V)

    elif parameters['n_input'] == 4: 
        x, disc_viol_found = \
            Functions.AddLieViolationsOrder4_v4(x,
                                                gamma_overbar,
                                                parameters['grid_points'],
                                                parameters['zeta_D'],
                                                parameters['verbose_info'],
                                                V_learn,
                                                lie_derivative_of_V)

    else:
        dimension_sys = parameters['n_input']
        logging.error(f'Not implemented Falsifier for system of order {dimension_sys}!')
        raise ValueError('Functionality to be implemented in: utilities/falsifier.py')

    return x, u_learn, V_learn, lie_derivative_of_V, f_out_sym, disc_viol_found


# Standalone SMT Falsifier function, i.e. detached from the Discrete Falsifier.
# This function is used in Fault Tolerant Control applications.
def falsifier_SMT(parameters, vars_, model, f_symb,
                  epsilon,
                  gamma_overbar,
                  config,
                  x):
    
    found_lyap_f = False
    to_fals = False
    ce_smt_found = 0.
    print("\nSMT Falsifier computing CE ...")

    # Computing the system symbolic dynamics
    u_learn, V_learn, f_out_sym = translator(parameters, vars_, model, f_symb)

    try:
        start_ = timeit.default_timer()

        CE_SMT, \
        lie_derivative_of_V = Functions.CheckLyapunov(vars_,
                                                      f_out_sym,
                                                      V_learn,
                                                      epsilon,
                                                      gamma_overbar,
                                                      config)


    except TimeoutError:
        logging.error("SMT Falsifier Timed Out")
        to_fals = True
        stop_ = timeit.default_timer()
        fals_to_check = stop_ - start_
        lie_derivative_of_V = Functions.LieDerivative_v2(vars_, f_out_sym, V_learn, parameters)

    if not to_fals:
        if CE_SMT:
            # if a counterexample is found
            print("SMT found a CE: ")
            print(CE_SMT)

            # Adding midpoint of the CE_SMT to the history
            ce_smt_found = 1.

            x = Functions.AddCounterexamples(x, CE_SMT, parameters['zeta_SMT'])
            if parameters['verbose_info']:
                logging.debug(f"'SMT Falsifier': Added {parameters['zeta_SMT']} points in the vicinity of the CE.\n")

    return x, lie_derivative_of_V, f_out_sym, found_lyap_f, to_fals, ce_smt_found


# Discrete Falsifier function, used in Fault Tolerant Control applications
# when an actuator loss of efficiency is considered too.    
def disc_falsifier_fault_efficiency_loss(parameters, vars_, eff_, model, f_symb,
                                        epsilon,
                                        gamma_overbar,
                                        config,
                                        x):

    # Computing the system symbolic dynamics
    u_learn, V_learn, f_out_sym = translator_efficiency_loss(parameters, vars_, eff_, model, f_symb)

    print("\nDiscrete Falsifier computing CEs ...")
    lie_derivative_of_V = Functions.LieDerivative_v2(vars_, f_out_sym, V_learn, parameters)

    if (parameters['n_input'] + parameters['n_input_eff']) == 2: 

        x, disc_viol_found = \
            Functions.AddLieViolationsOrder2_v4(x,
                                                gamma_overbar,
                                                parameters['grid_points'],
                                                parameters['zeta_D'],
                                                parameters['verbose_info'],
                                                V_learn,
                                                lie_derivative_of_V)

    elif (parameters['n_input'] + parameters['n_input_eff']) == 3: 
    
        if parameters['n_input_eff'] == 1:
            x, disc_viol_found = \
                Functions.AddLieViolationsOrder3_Faults(x,
                                                        parameters,
                                                        V_learn,
                                                        lie_derivative_of_V)
            
        else:
            dimension_sys = parameters['n_input']
            dimension_eff = parameters['n_input_eff']
            logging.error(f'Not implemented Falsifier for system of order {dimension_sys} with {dimension_eff} !')
            raise ValueError('Functionality to be implemented in: utilities/falsifier.py')

    else:
        dimension_sys = (parameters['n_input'] + parameters['n_input_eff'])
        logging.error(f'Not implemented Falsifier for system of order {dimension_sys}!')
        raise ValueError('Functionality to be implemented in: utilities/falsifier.py')

    return x, u_learn, V_learn, lie_derivative_of_V, f_out_sym, disc_viol_found


# SMT Falsifier function standalone, i.e. detached from the Discrete Falsifier.
# This function is used in Fault Tolerant Control applications.
def falsifier_SMT_efficiency_loss(parameters, vars_, eff_, model, f_symb,
                                epsilon,
                                gamma_overbar,
                                config,
                                x):
    
    found_lyap_f = False
    to_fals = False
    ce_smt_found = 0.
    print("\nSMT Falsifier computing CE ...")
    
    # Computing the system symbolic dynamics
    u_learn, V_learn, f_out_sym = translator_efficiency_loss(parameters, vars_, eff_, model, f_symb)

    try:
        start_ = timeit.default_timer()

        CE_SMT, \
        lie_derivative_of_V = Functions.CheckLyapunovEffLoss(vars_, eff_,
                                                            f_out_sym,
                                                            V_learn,
                                                            config,
                                                            parameters)


    except TimeoutError:
        logging.error("SMT Falsifier Timed Out")
        to_fals = True
        stop_ = timeit.default_timer()
        fals_to_check = stop_ - start_
        
    if not to_fals:
        if CE_SMT:
            # if a counterexample is found
            print("SMT found a CE: ")
            print(CE_SMT)

            # Adding midpoint of the CE_SMT to the history
            ce_smt_found = 1.

            x = Functions.AddCounterexamples(x, CE_SMT, parameters['zeta_SMT'])
            if parameters['verbose_info']:
                logging.debug(f"'SMT Falsifier': Added {parameters['zeta_SMT']} points in the vicinity of the CE.\n")

    return x, lie_derivative_of_V, f_out_sym, found_lyap_f, to_fals, ce_smt_found


def compute_symbolic(parameters, vars_, model, f_symb):
    # Compute symbolic expressions to be saved at intermediate training points

    u_learn, V_learn, f_out_sym = translator(parameters, vars_, model, f_symb)

    lie_derivative_of_V = Functions.LieDerivative_v2(vars_, f_out_sym, V_learn, parameters)

    return u_learn, V_learn, lie_derivative_of_V 
