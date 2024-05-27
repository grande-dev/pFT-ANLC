#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 14:11:49 2022

@authors: Davide Grande
          Andrea Peruffo

A script containing the discrete dynamics of 2-dimensional system.
The dynamics is integrated with a Forward Euler.

"""
import numpy as np
import torch

#
# 2-dimensional dynamics
#
def dyn(x, u, Dt, parameters):

    # Model dynamics parameters
    sigma = parameters['sigma']
    b = parameters['b']
    r = parameters['r']

    x1, x2 = x
    u1, u2, u3 = u

    dydt = [(-sigma*(x1-x2) + u1) * Dt + x1, 
            (r*x1 - x2 - x1 + u2 + u3) * Dt + x2]

    return torch.Tensor([dydt])
