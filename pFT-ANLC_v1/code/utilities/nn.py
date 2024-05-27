#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 10:47:24 2023

@authors: Andrea Peruffo
          Davide Grande

A class to define a Feedforward ANN composed of 3 branches:
1. a Lyapunov ANN;
2. a linear control ANN;
3. a nonlinear control ANN.

"""

import torch


#
# Class defining the ANN architecture
#
class Net(torch.nn.Module):

    # def init_arbitrary_layers(self, parameters, seed_):
    def __init__(self, parameters, seed_):
        super(Net, self).__init__()

        torch.manual_seed(seed_)

        self.n_input = parameters['n_input']
        self.size_layers = parameters['size_layers']
        self.layers = []
        self.ctrl_layers = []
        self.size_ctrl_layers = parameters['size_ctrl_layers']
        self.lyap_bias = parameters['lyap_bias']  # list of booleans
        self.ctrl_bias = parameters['ctrl_bias']  # list of booleans
        self.lin_contr_bias = parameters['lin_contr_bias']
        self.control_initialised = parameters['control_initialised']
        self.use_saturation = parameters['use_saturation']
        self.ctrl_sat = parameters['ctrl_sat']

        ## Lyapunov branch
        k = 1
        n_prev = self.n_input
        for idx in range(len(self.size_layers)):
            layer = torch.nn.Linear(n_prev, self.size_layers[idx], bias=self.lyap_bias[idx])
            self.register_parameter("W" + str(k), layer.weight)
            if self.lyap_bias[idx]:
                self.register_parameter("b" + str(k), layer.bias)
            self.layers.append(layer)
            n_prev = self.size_layers[idx]
            k = k + 1

        ## nonlinear control branch
        k = 1
        n_prev = self.n_input
        for idx in range(len(self.size_ctrl_layers)):
            layer = torch.nn.Linear(n_prev, self.size_ctrl_layers[idx], bias=self.ctrl_bias[idx])
            self.register_parameter("control" + str(k) + 'weight', layer.weight)
            if self.ctrl_bias[idx]:
                self.register_parameter("control" + str(k) + 'bias', layer.bias)
            self.ctrl_layers.append(layer)
            n_prev = self.size_ctrl_layers[idx]
            k = k + 1

        ## linear control branch
        self.control = torch.nn.Linear(self.n_input, self.size_ctrl_layers[-1], bias=self.lin_contr_bias)
        self.control.weight = torch.nn.Parameter(parameters['init_control']) 


        # activations
        self.activs = parameters['lyap_activations']

        self.use_lin_ctr = parameters['use_lin_ctr']
        self.ctrl_activs = parameters['ctrl_activations']

        self.beta_sfpl = parameters['beta_sfpl']

        print('============================================================')
        print("New Lyapunov Control architecture instantiated: \n")
        print("-- Lyapunov ANN: --\n")
        for j in range(len(parameters['lyap_activations'])):
            sig_j = parameters['lyap_activations'][j]
            print(f"sigma_{j+1} = {sig_j}")
        
        for j in range(len(parameters['lyap_activations'])):
            bias_j = parameters['lyap_bias'][j]
            print(f"bias_{j+1} = {bias_j} ")
            
        print(f"size_in = {self.n_input}")    
        for j in range(len(parameters['lyap_activations'])):
            size_j = parameters['size_layers'][j]
            print(f"size_{j+1} = {size_j}")
        
        print("-------------------------------------------------------------")
        if self.use_lin_ctr:
            print("-- Linear control ANN: -- \n")
            print(f"{self.n_input} x {self.size_ctrl_layers[-1]}")
            print(f"Use linear control bias = {self.lin_contr_bias}")
            print(f"Control weight initialised = {self.control_initialised}")
        
        else:
            print("-- Nonlinear control ANN: --\n")
            for j in range(len(parameters['size_ctrl_layers'])):
                sig_j = parameters['ctrl_activations'][j]
                print(f"sigma_{j+1} = {sig_j}")

            for j in range(len(parameters['size_ctrl_layers'])):
                bias_j = parameters['ctrl_bias'][j]
                print(f"bias_{j+1} = {bias_j} ")
            
            print(f"size_in = {self.n_input}")    
            for j in range(len(parameters['size_ctrl_layers'])):
                act_j = parameters['size_ctrl_layers'][j]
                print(f"size_{j+1} = {act_j}")

        if parameters['use_saturation']:
            print("Control function with saturated output.")

        print('============================================================\n')


    def use_in_control_loop(self, x):
        # act_fun_tanh = torch.nn.Tanh()
        sfpl = torch.nn.Softplus(beta=self.beta_sfpl, threshold=50)

        # Lyapunov ANN
        z = x.detach().clone()
        for idx in range(len(self.size_layers)):
            # linear pass
            zhat = self.layers[idx](z)
            # activation
            if self.activs[idx] == 'tanh':
                z = torch.tanh(zhat)
            elif self.activs[idx] == 'pow2':
                z = torch.pow(zhat, 2)
            elif self.activs[idx] == 'pow3':
                z = torch.pow(zhat, 3)
            elif self.activs[idx] == 'sfpl':
                z = sfpl(zhat)
            elif self.activs[idx] == 'linear':
                z = zhat
            else:
                str_fun = self.activs[idx]
                raise ValueError(f"The Activation Function '{str_fun}' is not implemented.")
        V = z

        # Control ANN
        if self.use_lin_ctr:
            u_lin = self.control(x)
            if self.use_saturation:
                u = torch.tensor(self.ctrl_sat)*torch.tanh(u_lin)
            else:
                u = u_lin

        else:
            # nonlinear control
            u = x.detach().clone()
            for idx in range(len(self.ctrl_layers)):
                uhat = self.ctrl_layers[idx](u)
                # activation
                if self.ctrl_activs[idx] == 'tanh':
                    u = torch.tanh(uhat)
                elif self.ctrl_activs[idx] == 'pow2':
                    u = torch.pow(uhat, 2)
                elif self.ctrl_activs[idx] == 'pow3':
                    u = torch.pow(uhat, 3)
                elif self.ctrl_activs[idx] == 'sfpl':
                    u = sfpl(uhat)
                elif self.ctrl_activs[idx] == 'linear':
                    u = uhat
                else:
                    str_fun = self.ctrl_activs[idx]
                    raise ValueError(f"The Activation Function '{str_fun}' is not implemented.")
            if self.use_saturation:
                u = torch.tensor(self.ctrl_sat)*torch.tanh(u)

        return V, u



    def forward(self, x, f_torch, parameters):
        # act_fun_tanh = torch.nn.Tanh()
        sfpl = torch.nn.Softplus(beta=self.beta_sfpl, threshold=50)

        # Control ANN
        if self.use_lin_ctr:
            u_lin = self.control(x)
            if self.use_saturation:
                u = torch.tensor(self.ctrl_sat)*torch.tanh(u_lin)
            else:
                u = u_lin
        else:
            # nonlinear control
            u = x.detach().clone()
            for idx in range(len(self.ctrl_layers)):
                uhat = self.ctrl_layers[idx](u)
                # activation
                if self.ctrl_activs[idx] == 'tanh':
                    u = torch.tanh(uhat)
                elif self.ctrl_activs[idx] == 'pow2':
                    u = torch.pow(uhat, 2)
                elif self.ctrl_activs[idx] == 'pow3':
                    u = torch.pow(uhat, 3)
                elif self.ctrl_activs[idx] == 'sfpl':
                    u = sfpl(uhat)
                elif self.ctrl_activs[idx] == 'linear':
                    u = uhat
                else:
                    str_fun = self.ctrl_activs[idx]
                    raise ValueError(f"The Activation Function '{str_fun}' is not implemented.")
            if self.use_saturation:
                u = torch.tensor(self.ctrl_sat)*torch.tanh(u)

        xdot = f_torch(x, u, parameters)
        V, Vdot, circle = self.compute_derivative_net(x, xdot)

        return V, Vdot, circle, u


    def forward_eff_loss(self, x_var, x_eff, f_torch, parameters):
            # A modified version of the forward() method accounting for the loss of efficiency vector.

            # act_fun_tanh = torch.nn.Tanh()
            sfpl = torch.nn.Softplus(beta=self.beta_sfpl, threshold=50)

            # Control ANN
            if self.use_lin_ctr:
                u_lin = self.control(x_var)
                if self.use_saturation:
                    u = torch.tensor(self.ctrl_sat)*torch.tanh(u_lin)
                else:
                    u = u_lin
            else:
                # nonlinear control
                u = x_var.detach().clone()
                for idx in range(len(self.ctrl_layers)):
                    uhat = self.ctrl_layers[idx](u)
                    # activation
                    if self.ctrl_activs[idx] == 'tanh':
                        u = torch.tanh(uhat)
                    elif self.ctrl_activs[idx] == 'pow2':
                        u = torch.pow(uhat, 2)
                    elif self.ctrl_activs[idx] == 'pow3':
                        u = torch.pow(uhat, 3)
                    elif self.ctrl_activs[idx] == 'sfpl':
                        u = sfpl(uhat)
                    elif self.ctrl_activs[idx] == 'linear':
                        u = uhat
                    else:
                        str_fun = self.ctrl_activs[idx]
                        raise ValueError(f"The Activation Function '{str_fun}' is not implemented.")
            xdot = f_torch(x_var, x_eff, u, parameters)
            V, Vdot, circle = self.compute_derivative_net(x_var, xdot)

            return V, Vdot, circle, u

    def compute_derivative_net(self, x, xdot):

        assert len(x) == len(xdot)

        nn, grad_nn = self.forward_tensors(x)
        #circle = torch.pow(x, 2).sum(dim=1)
        circle = torch.sqrt(torch.pow(x, 2).sum(dim=1))

        V = nn
        Vdot = torch.sum(torch.mul(grad_nn, xdot), dim=1)

        return V, Vdot, circle

    # compute V and Vdot together
    def forward_tensors(self, x):
        """
        :param x: tensor of data points
        :return:
                V: tensor, evaluation of x in net
                jacobian: tensor, evaluation of grad_net
        """

        sfpl = torch.nn.Softplus(beta=self.beta_sfpl, threshold=50)

        jacobian = torch.diag_embed(torch.ones(x.shape[0], self.n_input))

        y = x.detach().clone()
        for idx, layer in enumerate(self.layers):
            zhat = layer(y)
            # activation
            if self.activs[idx] == 'tanh':
                y = torch.tanh(zhat)
            elif self.activs[idx] == 'pow2':
                y = torch.pow(zhat, 2)
            elif self.activs[idx] == 'pow3':
                y = torch.pow(zhat, 3)
            elif self.activs[idx] == 'sfpl':
                y = sfpl(zhat)
            elif self.activs[idx] == 'linear':
                y = zhat
            else:
                str_fun = self.activs[idx]
                raise ValueError(f"The Activation Function '{str_fun}' is not implemented.")

            jacobian = torch.matmul(layer.weight, jacobian)
            jacobian = torch.matmul(
                torch.diag_embed(self.activation_derivative(self.activs[idx], zhat)), jacobian
            )

        # numerical_v = torch.matmul(y, self.layers[-1].weight.T)
        # jacobian = torch.matmul(self.layers[-1].weight, jacobian)

        return y[:, 0], jacobian[:, 0, :]

    def activation_derivative(self, activ, z):

        if activ == 'tanh':
            der_z = (1. - torch.tanh(z) ** 2)

        elif activ == 'pow2':
            der_z = 2. * z

        elif activ == 'pow3':
            der_z = 3. * z

        elif activ == 'linear':
            der_z = torch.ones(z.shape)

        elif activ == 'sfpl':
            der_z = torch.exp(self.beta_sfpl * z) / (1 + torch.exp(self.beta_sfpl * z))

        elif activ == 'sigm':
            der_z = 1. / (1. + torch.exp(-z))

        else:
            raise ValueError(f'Not Implemented Activation Derivative {activ}')

        return der_z


def optimizer_setup(parameters, model):
    """
    sets up the optimizer
    :param parameters:
    :param model:
    :return:
    """

    list_c = ['control.weight',
              'control1weight', 'control2weight', 'control3weight',
              'control1bias', 'control2bias', 'control3bias']
    params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in list_c, model.named_parameters()))))
    base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in list_c, model.named_parameters()))))
    
    optimizer = torch.optim.Adam([{'params': base_params},
                                  {'params': params, 'lr': parameters['learning_rate_c']}],
                                  lr=parameters['learning_rate'])

    # Basic callback
    # optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'])

    return optimizer
