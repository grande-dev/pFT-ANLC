#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:21:49 2022

A utility function to substitute dReal syntax with the equivalent numpy one.

@authors: Davide Grande
          Andrea Peruffo
"""
import re
import logging

def sub(in_str):
       
    # substitute hyperbolic functions
    try:
        out_str_tanh = re.sub(r'tanh' , r'np.tanh', in_str)
    except:
        logging.debug("No tanh function")
    else:
        in_str = out_str_tanh
    
    try:
        out_str_sinh = re.sub(r'sinh' , r'np.sinh', in_str)
    except:
        logging.debug("No sinh function")
    else:
        in_str = out_str_sinh
        
    try:
        out_str_cosh = re.sub(r'cosh' , r'np.cosh', in_str)
    except:
        logging.debug("No cosh function")
    else:
        in_str = out_str_cosh
        
    # substitute trigonometric functions 
    try:
        out_str_cos = re.sub(r'cos' , r'np.cos', in_str)
    except:
        logging.debug("No cos function")
    else:
        in_str = out_str_cos
        
    try:
        out_str_sin = re.sub(r'sin' , r'np.sin', in_str)
    except:
        logging.debug("No sin function")
    else:
        in_str = out_str_sin

    try:
        out_str_asin = re.sub(r'arcsin' , r'np.arcsin', in_str)
    except:
        logging.debug("No arcsin function")
    else:
        in_str = out_str_asin        

    try:
        out_str_acos = re.sub(r'arccos' , r'np.arccos', in_str)
    except:
        logging.debug("No arccos function")
    else:
        in_str = out_str_acos       

    try:
        out_str_atan = re.sub(r'atan' , r'np.arctan', in_str)
    except:
        logging.debug("No arctan function")
    else:
        in_str = out_str_atan  

    try:
        out_str_atan2 = re.sub(r'arctan2' , r'np.arctan2', in_str)
    except:
        logging.debug("No arctan2 function")
    else:
        in_str = out_str_atan2  

    try:
        out_str_pow = re.sub(r'pow' , r'np.power', in_str)
    except:
        logging.debug("No pow function")
    else:
        in_str = out_str_pow

    try:
        out_str_log = re.sub(r'log' , r'np.log', in_str)
    except:
        logging.debug("No log function")
    else:
        in_str = out_str_log

    try:
        out_str_exp = re.sub(r'exp' , r'np.exp', in_str)
    except:
        logging.debug("No exp function")
    else:
        in_str = out_str_exp
    
    try:
        out_str_exp = re.sub(r'sqrt' , r'np.sqrt', in_str)
    except:
        logging.debug("No sqrt function")
    else:
        in_str = out_str_exp
    
    try:
        out_str_np = re.sub(r'np.np.' , r'np.', in_str)
    except:
        logging.debug("np. function")
    else:
        in_str = out_str_np

    return in_str
    
    
    
