# Passive Fault Tolerant-Augmented Neural Lyapunov Control WIKI
## A step-by-step guide

"I have a 2-dimensional system with 3 actuators and 2 possible faults that I want to control, how can I set the code up?"

This 20 minutes-long tutorial will guide you through the 6 steps encompassing all you need to start the synthesis of your control functions.

Let us consider the following dynamical system:

$$
\begin{cases}
\dot{x}_1 = -\alpha x_2^2 + 3 h_1 u_1  \\
\dot{x}_2 = x_1 x_3^3 + h_2 u_2 cos(\alpha) + \beta h_3 u_3  \\
\end{cases}
$$

where $\alpha, \beta$ represent two (scalar) dynamic parameters, $\textbf{x}=[x_1, x_2, x_3]^T$ the state-space vector, $\textbf{u}=[u_1, u_2, u_3]^T$ the control vector and $\textbf{h}=[h_1, h_2, h_3]^T$ the vector of the faults.

  
## Overall view
To use the framework, the following steps are recommended:

1. Define the *system dynamics*;
2. Set up the training *parameters*;
3. Set up the *Artificial Neural Networks* architecture;
4. Set up the configuration conditions for the *Falsifier*;
5. *Start* the training;  
6. The *procedure stops* when no counterexample is found or when a timeout is reached.  
  

## Step-by-step example
Only 4 files need to be modified to run a custom example: `main_2d_faulty_template.py`, `utilities/models`, `configuration/config_2d_faulty_template.py` and `systems_dynamics/dynamics_2d_faulty_template.py`.  

The files can be modified as follows:    
   
1. **Define the system dynamics**:
    
	1. Define your dynamic system in `utilities/models.py`, taking `class AUV2D_Faulty()` as an example. 
	Create a new class called, for instance, `class YourNewModel_Faulty()`, and define the dynamics as:  	
	```python
		class YourNewModel_Faulty():

			def __init__(self):
				self.vars_ = [dreal.Variable("x1"), dreal.Variable("x2")]
        
			def f_torch(x, u, parameters):

				x1, x2 = x[:,0], x[:,1]
				x1_shift = x1 + parameters['x_star'][0]
				x2_shift = x2 + parameters['x_star'][1]
				u_NN0 = u[:,0]
				u_NN1 = u[:,1]
				u_NN2 = u[:,2]

				alpha = parameters['alpha']
				beta = parameters['beta']
				h1 = parameters['h1']
				h2 = parameters['h2']
				h3 = parameters['h3']

				x_dot = [-alpha * x2_shift**2 + 3*h_1*u_NN0,
					x1_shift*x3_shift**3 + h_2*u_NN1*torch.cos(alpha) + beta*h_3*u_NN2]

				x_dot = torch.transpose(torch.stack(x_dot), 0, 1)

			return x_dot
	```
	  
	2. Similarly, the symbolic dynamics for the dReal verification step is defined in `def f_symb()`.  
  	```python
		def f_symb(x, u, parameters, x_star):
			x1, x2 = x[0], x[1]
			x1_shift = x1 + parameters['x_star'].numpy()[0]
			x2_shift = x2 + parameters['x_star'].numpy()[1]
			u_NN0 = u[0]
			u_NN1 = u[1]
			u_NN2 = u[2]
			
			alpha = parameters['alpha']
			beta = parameters['beta']
			h1 = parameters['h1']
			h2 = parameters['h2']
			h3 = parameters['h3']
				
			x_dot = [-alpha * x2_shift**2 + 3*h_1*u_NN0,
					x1_shift*x3_shift**3 + h_2*u_NN1*dreal.cos(alpha) + beta*h_3*u_NN2]
				
			return x_dot
	```
	   
	
	3. If your dynamics has scalar parameters, define them in `configuration/config_2d_faulty_template.py/dyn_sys_params` and add the actuator health stati parameters (h_j):
	```python
		dyn_sys_params = {
			'alpha': 2.1,
			'beta': 5.3,
			'h1': 1.,  # nominal health status of actuator 1
			'h2': 1.,  # nominal health status of actuator 2
			'h3': 1.,  # nominal health status of actuator 3
		}
	```

 	4. In the `main_2d_faulty_template.py` file, import the model class defined in step 1.,  as:
   	```python
	from utilities.models import YourNewModel_Faulty as UsedModel
	import configuration.config_2d_faulty_template as config_file
	import closed_loop_testing.cl_2d_faulty_template as cl
	import systems_dynamics.dynamics_2d_faulty_template as dynamic_sys
 	```  
 	where *YourNewModel_Faulty* should match your new 2-dimensional model name.  
  
  	5. Optional: if you want to test the closed-loop system upon convergence, the dynamics should also be defined in `systems_dynamics/dynamics_2d_faulty_template.py` (currently a Forward Euler integrator is implemented).
 	```python
		alpha = parameters['alpha']
		beta = parameters['beta']
		h1 = parameters['h1']
		h2 = parameters['h2']
		h3 = parameters['h3']

		x1, x2 = x
		u1, u2, u3 = u
			
		dydt = [(-alpha * x2**2 + 3*h_1*u1)*Dt + x1,
				(x1_shift*x3_shift**3 + h_2*u2*np.cos(alpha) + \beta*h_3*u3)*Dt + x2]
			
 	```  
  
2. **Set up the training *parameters***:  
	All the parameters are included in the `configuration/config_2d_faulty_template.py` file.

	1. Select the number of training runs (`tot_runs`) and the maximum learning iterations per run (`max_iters`) as:
	```python
		campaign_params = {
			
			'init_seed': 1,         # initial campaign seed
			'campaign_run': 4000,  # number of the run.
									# The results will be saved in /results/campaign_'campaign_run'
			'tot_runs': 10,        # total number of runs of the campaigns (each one with a different seed)
			'max_loop_number': 1,  # number of loops per run (>1 means that the weights will be re-initialised).
									# default value = 1.
			'max_iters': 1000,     # number of maximum learning iterations per run
			'system_name': "your_model_2d",  # name of the systems to be controlled
			'x_star': torch.tensor([0.0, 0.0]),  # target equilibrium point
		}    
	```  
	This parameters configuration will launch a simulation campaign composed of 10 runs of 1000 learning iterations each. 
	The results will be generated in: `/results/campaign_4000`.
	
	2. Select the loss function tuning parameters as:
	```python
		loss_function = {
			# Loss function tuning
			'alpha_1': 1.0,  # weight V
			'alpha_2': 1.0,  # weight V_dot
			'alpha_3': 1.0,  # weight V0
			'alpha_4': 0.1,  # weight tuning term V
			'alpha_roa': falsifier_params['gamma_overbar'],  # Lyapunov function steepness
			'alpha_5': 1.0,  # general scaling factor  
			'off_Lie': 0.0,   # additional penalisation of the Lie derivative  
		}
	```  
	
	3. Select the dataset initial and final dimensions as:
	```python
		# Parameters for learner
		learner_params = {
				'N': 500,                      # initial dataset size
				'N_max': 1000                  # maximum dataset size (if using a sliding window)
				...
			}
	```  
	  
3. **Set up the *Artificial Neural Networks* architecture**:   
	1. Select the structure of the Lyapunov ANN:  
	```python
		# Parameters for Lyapunov ANN
		lyap_params = {
			'n_input': 2, # input dimension (n = n-dimensional system)
			'beta_sfpl': 2,  # the higher, the steeper the Softplus, the better approx. sfpl(0) ~= 0
			'clipping_V': True,  # clip weight of Lyapunov ANN
			'size_layers': [10, 10, 1],  # CAVEAT: the last entry needs to be = 1 (this ANN outputs a scalar)!
			'lyap_activations': ['pow2', 'linear', 'linear'],
			'lyap_bias': [False, False, False],
		}
	```  
	This configuration defines a Lyapunov ANN composed of 2 hidden layers of 10 neurons each, with no bias, and with quadratic activation function on the first hidden layer. Make sure to leave the last entry of `size_layers` set to 1, as the output of the Lyapunov ANN is by definition a scalar.
	
	2. Select the structure of the control ANN:
	```python
    	# Parameters for control ANN
    	control_params = {
			'use_lin_ctr': True,  # use linear control law  -- defined as 'phi' in the ANLC publication
			'lin_contr_bias': False,  # use bias on linear control layer
			'control_initialised': False,  # initialised control ANN with pre-computed LQR law
			'init_control': torch.tensor([[-3.0, -1.7, 24.3]]),  # initial control solution
			'size_ctrl_layers': [5, 3],  # CAVEAT: the last entry is the number of control actions!
			'ctrl_bias': [True, False],
			'ctrl_activations': ['tanh', 'linear'],
    	}
	```  
	This configuration defines a linear control ANN with random initial values. If `control_initialised` was set to `True` instead, `init_control` would be chosen as initial control weight.  
	If a nonlinear control law is to be used instead, set `use_lin_ctr=False`; the nonlinear control law weight is by default initialised randomly.  
	The size of the control vector (both linear and nonlinear) is defined by the last entry of `size_ctrl_layers`; with the choice above, three control signals will be output.  


4.  **Set up the configuration conditions for the *Falsifier***:
	1. Select the SMT domain boundaries and the additional configuration parameters as:
    	```python
    	    falsifier_params = {
				# a) SMT parameters
				'epsilon': 0.1,         # domain lower boundary
				'gamma_overbar': 2.0,   # domain upper boundary
				'zeta_SMT': 200,        # how many points are added to the dataset 
							# after a CE box is found
				
				# b) Discrete Falsifier parameters
				'grid_points': 50,  # sampling size grid
				'zeta_D': 50,       # how many points are added at each DF callback
			}
    	``` 
	  
5. *Start* the training by launching the `main_2d_faulty_template.py` script.  

6. The procedure *stops* when no *counterexample* is found or when a *timeout* is hit:
	1. the timeout refers to either the maximum number of learning iterations (`max_iters`) OR to the maximum time dReal is allowed to search for a solution. 
	A value of 180 seconds is suggested for the latter and it is currently implemented. To modify it, change the value of `@timeout(180, use_signals=True)` in file `utilities/Functions/CheckLyapunov()`. 
	 

