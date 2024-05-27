#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:00:07 2023

@authors: Davide Grande
          Andrea Peruffo

A script containing the discrete dynamics of an AUV with 3 thrusters.

"""
import numpy as np
import torch

#
# 2-dimensional dynamics
#
def dyn(x, u, Dt, parameters):
    
    m = parameters['m']
    Jz = parameters['Jz']
    Xu = parameters['Xu']
    Xuu = parameters['Xuu']
    Nr = parameters['Nr']
    Nrr = parameters['Nrr']
    l1x = parameters['l1x']
    l1y = parameters['l1y']
    alpha1 = parameters['alpha1']
    l2x = parameters['l2x']
    l2y = parameters['l2y']
    alpha2 = parameters['alpha2']
    l3x = parameters['l3x']
    l3y = parameters['l3y']
    alpha3 = parameters['alpha3']

    h1 = parameters['h1']
    h2 = parameters['h2']
    h3 = parameters['h3']

    x1, x2 = x
    u1, u2, u3 = u

    F1x = u1*np.sin(torch.tensor(alpha1))
    F1y = u1*np.cos(torch.tensor(alpha1))
    F2x = u2*np.sin(torch.tensor(alpha2))
    F2y = u2*np.cos(torch.tensor(alpha2))
    F3x = u3*np.sin(torch.tensor(alpha3))
    F3y = u3*np.cos(torch.tensor(alpha3))
    
    xdot = [(-Xu*x1-Xuu*x1**2 + F1x*h1                + F2x*h2                + F3x*h3                  )/m * Dt + x1,
            (-Nr*x2-Nrr*x2**2 + (-F1x*l1y+F1y*l1x)*h1 + (-F2x*l2y+F2y*l2x)*h2 + (-F3x*l3y+F3y*l3x)*h3 ) /Jz * Dt + x2]

    return torch.Tensor([xdot])