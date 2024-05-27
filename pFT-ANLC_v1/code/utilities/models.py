#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:42:24 2022

Computing the symbolic dynamics of generic n-dimensional systems.

@authors: Andrea Peruffo
          Davide Grande

"""

import numpy as np
import dreal
import torch


class YourModel2D():

    def __init__(self):
        self.vars_ = [dreal.Variable("x1"), dreal.Variable("x2")]

    @staticmethod
    def f_torch(x, u, parameters):

        x1, x2 = x[:,0], x[:,1]
        x1_shift = x1 + parameters['x_star'][0]
        x2_shift = x2 + parameters['x_star'][1]
        u_NN0, u_NN1, u_NN2 = u[:,0], u[:,1], u[:,2]

        sigma = parameters['sigma']
        r = parameters['r']

        xdot = [-sigma * (x1_shift - x2_shift) + u_NN0, r * x1_shift - x2_shift - x1_shift + u_NN1 + u_NN2]

        xdot = torch.transpose(torch.stack(xdot), 0, 1)

        return xdot


    @staticmethod
    def f_symb(x, u, parameters):

        x1, x2 = x[0], x[1]
        x1_shift = x1 + parameters['x_star'].numpy()[0]
        x2_shift = x2 + parameters['x_star'].numpy()[1]
        u_NN0, u_NN1, u_NN2 = u[0], u[1], u[2]

        sigma = parameters['sigma']
        r = parameters['r']

        return [-sigma * (x1_shift - x2_shift) + u_NN0, r*x1_shift - x2_shift - x1_shift + u_NN1 + u_NN2]


class Pendulum():

    def __init__(self):
        self.vars_ = [dreal.Variable("x1"), dreal.Variable("x2")]

    @staticmethod
    def f_torch(x, u, parameters):

        x1, x2 = x[:,0], x[:,1]
        x1_shift = x1 + parameters['x_star'][0]
        x2_shift = x2 + parameters['x_star'][1]
        u_NN0 = u[:,0]

        G = parameters['G']
        L = parameters['L']
        m = parameters['m']
        b = parameters['b']

        xdot = [x2_shift,
                 (m*G*L*np.sin(x1_shift) - b*x2_shift + u_NN0) / (m*L**2)]

        xdot = torch.transpose(torch.stack(xdot), 0, 1)

        return xdot


    @staticmethod
    def f_symb(x, u, parameters):

        x1, x2 = x[0], x[1]
        x1_shift = x1 + parameters['x_star'].numpy()[0]
        x2_shift = x2 + parameters['x_star'].numpy()[1]
        u_NN0 = u[0]

        G = parameters['G']
        L = parameters['L']
        m = parameters['m']
        b = parameters['b']
        
        xdot = [x2_shift,
                 (m*G*L*dreal.sin(x1_shift) - b*x2_shift + u_NN0) / (m*L**2)]

        return xdot


class RedundantPendulum():

    # A model of a 2-dimensional redundant inverted pendulum with 2 actuators.
    # The faults entail both motors broken (at most one at each time).

    def __init__(self):
        self.vars_ = [dreal.Variable("x1"), dreal.Variable("x2")]

    @staticmethod
    def f_torch(x, u, parameters):

        x1_0, x2_0 = x[:,0], x[:,1]
        x1 = x1_0 + parameters['x_star'][0]
        x2 = x2_0 + parameters['x_star'][1]
        u_NN1 = u[:,0]
        u_NN2 = u[:,1]

        G = parameters['G']
        L = parameters['L']
        m = parameters['m']
        b = parameters['b']
        h1 = parameters['h1']
        h2 = parameters['h2']

        u1 = u_NN1 * h1
        u2 = u_NN2 * h2

        xdot = [x2,
                (m*G*L*np.sin(x1) - b*x2 + u1 - u2 ) /(m*L**2)]

        xdot = torch.transpose(torch.stack(xdot), 0, 1)

        return xdot


    def f_symb(x, u, parameters):

        x1_0, x2_0 = x[0], x[1]
        x1 = x1_0 + parameters['x_star'].numpy()[0]
        x2 = x2_0 + parameters['x_star'].numpy()[1]
        u_NN1 = u[0] 
        u_NN2 = u[1]   

        G = parameters['G']
        L = parameters['L']
        m = parameters['m']
        b = parameters['b']
        h1 = parameters['h1']
        h2 = parameters['h2']

        u1 = u_NN1 * h1
        u2 = u_NN2 * h2

        xdot = [x2,
                (m*G*L*dreal.sin(x1) - b*x2 + u1 - u2 ) /(m*L**2)]

        return xdot


class Glider2DFaults():

    # A model of a 2D glider (u, w) with 3 actuators (VBD and 2 stern planes).
    # The faults entail the stern planes blocked due to mechanical failure.

    def __init__(self):
        self.vars_ = [dreal.Variable("x1"), dreal.Variable("x2")]

    @staticmethod
    def f_torch(x, u, parameters):

        x1_0, x2_0 = x[:,0], x[:,1]
        x1 = x1_0 + parameters['x_star'][0]
        x2 = x2_0 + parameters['x_star'][1]
        u_NN1 = u[:,0]
        u_NN2 = u[:,1]
        u_NN3 = u[:,2]

        mh = parameters['mh']
        mw = parameters['mw']
        mp = parameters['mp']
        mf1 = parameters['mf1']
        mf3 = parameters['mf3']
        Vh = parameters['Vh']
        rho = parameters['rho']
        g = parameters['g']
        KD0 = parameters['KD0']
        KD = parameters['KD']
        KL0 = parameters['KL0']
        KL = parameters['KL']
        Kfd = parameters['Kfd']
        Kud = parameters['Kud']
        alpha = parameters['alpha']
        theta = parameters['theta']
        h1 = parameters['h1']
        h2 = parameters['h2']
        h3 = parameters['h3']
        off_dp = parameters['off_dp']
        off_ds = parameters['off_ds']

        u1 = u_NN1 * h1  # volume VBD
        u2 = u_NN2 * h2 + off_dp # delta1
        u3 = u_NN3 * h3 + off_ds # delta2

        ms = mh+mw  # static mass
        V2 = x1**2+x2**2   # flow speed squared
        L = (KL0+KL*alpha)*V2   # lift
        D = (KD0+KD*alpha**2)*V2  # drag
        Bh = rho*g*Vh
        Bv = rho*g*u1
        Gs = ms*g
        Gp = mp*g
        Fd_p = Kfd*Kud*V2*u2
        Fd_s = Kfd*Kud*V2*u3
        m1 = ms+mp+mf1
        m3 = ms+mp+mf3

        xdot = [(-D * np.cos(alpha) + L * np.sin(alpha) + np.sin(theta)*(Bh+Bv-Gs-Gp))/m1,
                 (-D * np.sin(alpha) - L * np.cos(alpha) + np.cos(theta)*(-Bh-Bv+Gs+Gp) + Fd_p + Fd_s )/m3]

        xdot = torch.transpose(torch.stack(xdot), 0, 1)

        return xdot


    def f_symb(x, u, parameters):

        x1_0, x2_0 = x[0], x[1]
        x1 = x1_0 + parameters['x_star'].numpy()[0]
        x2 = x2_0 + parameters['x_star'].numpy()[1]
        u_NN1 = u[0] 
        u_NN2 = u[1]   
        u_NN3 = u[2]

        mh = parameters['mh']
        mw = parameters['mw']
        mp = parameters['mp']
        mf1 = parameters['mf1']
        mf3 = parameters['mf3']
        Vh = parameters['Vh']
        rho = parameters['rho']
        g = parameters['g']
        KD0 = parameters['KD0']
        KD = parameters['KD']
        KL0 = parameters['KL0']
        KL = parameters['KL']
        Kfd = parameters['Kfd']
        Kud = parameters['Kud']
        alpha = parameters['alpha']
        theta = parameters['theta']
        h1 = parameters['h1']
        h2 = parameters['h2']
        h3 = parameters['h3']
        off_dp = parameters['off_dp']
        off_ds = parameters['off_ds']

        u1 = u_NN1*h1  # volume VBD
        u2 = u_NN2 * h2 + off_dp # delta1
        u3 = u_NN3 * h3 + off_ds # delta2

        ms = mh+mw  # static mass
        V2 = x1**2+x2**2   # flow speed squared
        L = (KL0+KL*alpha)*V2   # lift
        D = (KD0+KD*alpha**2)*V2  # drag
        Bh = rho*g*Vh
        Bv = rho*g*u1
        Gs = ms*g
        Gp = mp*g
        Fd_p = Kfd*Kud*V2*u2
        Fd_s = Kfd*Kud*V2*u3
        m1 = ms+mp+mf1
        m3 = ms+mp+mf3

        xdot = [(-D * np.cos(alpha) + L * np.sin(alpha) + np.sin(theta)*(Bh+Bv-Gs-Gp))/m1,
                 (-D * np.sin(alpha) - L * np.cos(alpha) + np.cos(theta)*(-Bh-Bv+Gs+Gp) + Fd_p + Fd_s )/m3]

        return xdot


class AUV2D_Faulty():

    def __init__(self):
        self.vars_ = [dreal.Variable("x1"), dreal.Variable("x2")]

    @staticmethod
    def f_torch(x, u, parameters):

        x1, x2 = x[:,0], x[:,1]
        x1_shift = x1 + parameters['x_star'][0]
        x2_shift = x2 + parameters['x_star'][1]
        u_NN0 = u[:,0]
        u_NN1 = u[:,1]
        u_NN2 = u[:,2]       

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

        F1x = u_NN0*torch.sin(torch.tensor(alpha1))
        F1y = u_NN0*torch.cos(torch.tensor(alpha1))
        F2x = u_NN1*torch.sin(torch.tensor(alpha2))
        F2y = u_NN1*torch.cos(torch.tensor(alpha2))
        F3x = u_NN2*torch.sin(torch.tensor(alpha3))
        F3y = u_NN2*torch.cos(torch.tensor(alpha3))

        xdot = [(-Xu*x1_shift-Xuu*x1_shift**2 + F1x*h1                + F2x*h2                + F3x*h3                )/m,
                 (-Nr*x2_shift-Nrr*x2_shift**2 + (-F1x*l1y+F1y*l1x)*h1 + (-F2x*l2y+F2y*l2x)*h2 + (-F3x*l3y+F3y*l3x)*h3 )/Jz]

        xdot = torch.transpose(torch.stack(xdot), 0, 1)

        return xdot


    @staticmethod
    def f_symb(x, u, parameters):

        x1, x2 = x[0], x[1]
        x1_shift = x1 + parameters['x_star'].numpy()[0]
        x2_shift = x2 + parameters['x_star'].numpy()[1]
        u_NN0 = u[0]
        u_NN1 = u[1]
        u_NN2 = u[2]    

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

        F1x = u_NN0*dreal.sin(alpha1)
        F1y = u_NN0*dreal.cos(alpha1)
        F2x = u_NN1*dreal.sin(alpha2)
        F2y = u_NN1*dreal.cos(alpha2)
        F3x = u_NN2*dreal.sin(alpha3)
        F3y = u_NN2*dreal.cos(alpha3)

        xdot = [(-Xu*x1_shift-Xuu*x1_shift**2 + F1x*h1                + F2x*h2                + F3x*h3                 )/m,
                 (-Nr*x2_shift-Nrr*x2_shift**2 + (-F1x*l1y+F1y*l1x)*h1 + (-F2x*l2y+F2y*l2x)*h2 + (-F3x*l3y+F3y*l3x)*h3  )/Jz]

        return xdot


class AUV2D_FaultyEfficiencyLoss():

    def __init__(self):
        self.vars_ = [dreal.Variable("x1"), dreal.Variable("x2")]
        self.eff_ = [dreal.Variable("e1")]
        # CAVEAT: the model is 2-dimensional: every variable after x2 represents a loss of efficiency variable,
        # not a 'legit' state-space variable.

    @staticmethod
    def f_torch(x, x_eff, u, parameters):

        x1, x2 = x[:,0], x[:,1]
        x1_shift = x1 + parameters['x_star'][0]
        x2_shift = x2 + parameters['x_star'][1]
        e1 = x_eff[:,0]
        u_NN0 = u[:,0]
        u_NN1 = u[:,1]
        u_NN2 = u[:,2]

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

        F1x = u_NN0*torch.sin(torch.tensor(alpha1))
        F1y = u_NN0*torch.cos(torch.tensor(alpha1))
        F2x = u_NN1*torch.sin(torch.tensor(alpha2))
        F2y = u_NN1*torch.cos(torch.tensor(alpha2))
        F3x = u_NN2*torch.sin(torch.tensor(alpha3))
        F3y = u_NN2*torch.cos(torch.tensor(alpha3))

        xdot = [(-Xu*x1_shift-Xuu*x1_shift**2 + F1x*e1                + F2x*h2                + F3x*h3                )/m,
                 (-Nr*x2_shift-Nrr*x2_shift**2 + (-F1x*l1y+F1y*l1x)*e1 + (-F2x*l2y+F2y*l2x)*h2 + (-F3x*l3y+F3y*l3x)*h3 )/Jz]

        xdot = torch.transpose(torch.stack(xdot), 0, 1)

        return xdot


    @staticmethod
    def f_symb(x, x_eff, u, parameters):

        x1, x2 = x[0], x[1]
        x1_shift = x1 + parameters['x_star'].numpy()[0]
        x2_shift = x2 + parameters['x_star'].numpy()[1]
        e1 = x_eff[0]
        u_NN0 = u[0]
        u_NN1 = u[1]
        u_NN2 = u[2]    

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

        F1x = u_NN0*dreal.sin(alpha1)
        F1y = u_NN0*dreal.cos(alpha1)
        F2x = u_NN1*dreal.sin(alpha2)
        F2y = u_NN1*dreal.cos(alpha2)
        F3x = u_NN2*dreal.sin(alpha3)
        F3y = u_NN2*dreal.cos(alpha3)

        xdot = [(-Xu*x1_shift-Xuu*x1_shift**2 + F1x*e1                + F2x*h2                + F3x*h3                 )/m,
                 (-Nr*x2_shift-Nrr*x2_shift**2 + (-F1x*l1y+F1y*l1x)*e1 + (-F2x*l2y+F2y*l2x)*h2 + (-F3x*l3y+F3y*l3x)*h3  )/Jz]

        return xdot


class Template_model3D_Faulty():

    def __init__(self):
        self.vars_ = [dreal.Variable("x1"), dreal.Variable("x2"), dreal.Variable("x3")]

    @staticmethod
    def f_torch(x, u, parameters):

        x1_0, x2_0, x3_0 = x[:,0], x[:,1], x[:,2]
        x1 = x1_0 + parameters['x_star'][0]
        x2 = x2_0 + parameters['x_star'][1]
        x3 = x3_0 + parameters['x_star'][2]
        u1 = u[:,0]
        u2 = u[:,1]
        u3 = u[:,2]   
        u4 = u[:,3]   

        alpha = parameters['alpha']
        beta = parameters['beta']
        h1 = parameters['h1']
        h2 = parameters['h2']
        h3 = parameters['h3']
        h4 = parameters['h3']

        xdot = [-alpha*x2**2 + 3*h1*u1,
                 beta*x1+h2*u2-h3*u3,
                 x3+np.sin(x1)+h1*u1-h4*u4]

        xdot = torch.transpose(torch.stack(xdot), 0, 1)

        return xdot


    @staticmethod
    def f_symb(x, u, parameters):

        x1_0, x2_0, x3_0 = x[0], x[1], x[2]
        x1 = x1_0 + parameters['x_star'].numpy()[0]
        x2 = x2_0 + parameters['x_star'].numpy()[1]
        x3 = x3_0 + parameters['x_star'].numpy()[2]
        u1 = u[0]
        u2 = u[1]
        u3 = u[2]  
        u4 = u[3]    

        alpha = parameters['alpha']
        beta = parameters['beta']
        h1 = parameters['h1']
        h2 = parameters['h2']
        h3 = parameters['h3']
        h4 = parameters['h3']
    
        xdot = [-alpha*x2**2 + 3*h1*u1,
                 beta*x1+h2*u2-h3*u3,
                 x3+dreal.sin(x1)+h1*u1-h4*u4]

        return xdot