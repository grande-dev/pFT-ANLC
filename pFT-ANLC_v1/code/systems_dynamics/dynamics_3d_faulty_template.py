#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 14:11:49 2022

@authors: Davide Grande
          Andrea Peruffo

A script containing the discrete dynamics of a 3-dimensional system.
The dynamics is integrated with a Forward Euler.

"""
import numpy as np
import torch

#
# 3-dimensional system
#
def dyn(x, u, Dt, parameters):

    # Model dynamics parameters
    alpha = parameters['alpha']
    beta = parameters['beta']
    h1 = parameters['h1']
    h2 = parameters['h2']
    h3 = parameters['h3']
    h4 = parameters['h3']

    x1, x2, x3 = x
    u1, u2, u3, u4 = u

    xdot = [(-alpha*x2**2 + 3*h1*u1)* Dt + x1,
            (beta*x1+h2*u2-h3*u3)* Dt + x2,
            (x3+np.sin(x1)+h1*u1-h4*u4)* Dt + x3]

    return torch.Tensor([xdot])
