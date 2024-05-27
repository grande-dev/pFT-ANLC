# Passive Fault Tolerant-Augmented Neural Lyapunov Control (pFT-ANLC)
This repository contains the software tool discussed in the paper:  
**Passive Fault-Tolerant Augmented Neural Lyapunov Control: a method to synthesise control functions for marine vehicles affected by actuators faults** 
  
  
The work can be read open-access [here](https://www.sciencedirect.com/science/article/pii/S0967066124000959).  
  
  
## Scope of the code
*pFT-ANLC* is a software tool to **automatically** synthesise:  
1. a **stabilising control law** for a desired equilibrium of a nonlinear system affected by **actuator faults**;  
2. a **Control Lyapunov Function** (CLF) to certify the stability of the equilibrium.    
  
The code is based on a loop between a *Learner* and a *Falsifier*. Starting from a finite set of state-space samples, the *Learner* trains two Artificial Neural Networks (ANNs), one representing a control law and the other a CLF.   
In parallel, the *Falsifier* is tasked with verifying whether the candidate CLF satisfies the theoretical Lyapunov conditions within the dense domain over the Reals.  
If the theoretical Lyapunov conditions are satisifed, the learning is halted and the resulting control law and CLF are returned. If the conditions are not satisfied, the Falsifier returns a set of points (denoted as counterexample) where the Lyapunov conditions are violated. These points are added to the dataset and the learning process is further iterated.    
The learning system attempts to simultaneously stabilise a set of dynamics, ecompassing the nominal dynamics (fault-free) and the faulty modes.
  
   
A schematic of the learning architecture is hereby illustrated:  
<img src="https://github.com/grande-dev/pFT-ANLC/blob/master/pFT-ANLC_v1/documentation/images/learning_scheme.png" width=100% height=100%>
  
### Key features
The *pFT-ANLC* tool features:
- [x] Synthesis of both linear and nonlinear control laws;  
- [x] No control gain initialisation required;
- [x] Stabilisation of generic equilibria (not necessarily the origin);
- [x] Coping with complete loss of actuator efficiency, partial loss of actuator efficiency, or jammed actuators.  
  
### Key property
- [x] The method and the software tool are *sound*: if a control law and a CLF are obtained with the *pFT-ANLC* tool, the result is correct, namely the closed-loop dynamics is stable around the desired equilibrium.
 
  
## Installation  
Instructions on installation are available within the ![INSTALLATION](./installation/INSTALLATION.md) file.    
  
## Step-by-step guide
To synthesise control laws and CLFs for your own dynamics, a step-by-step example is reported in the ![WIKI](./pFT-ANLC_v1/documentation/WIKI.md) file.  
  
Hereby an example of how a CLF and a corresponding Lie derivative function are updated over successive training iterations.  
Note the yellow patches of the Lie derivative gradually disappearing as the training proceeds. The training halts once, at the same time, the CLF is certified to be positive definite and the Lie derivative is certified to be negative definite. 
  
CLF evolution |  Lie derivative function evolution                    
:-------------------------:|:-------------------------:
<img src="https://github.com/grande-dev/pFT-ANLC/blob/master/pFT-ANLC_v1/documentation/images/V_training.gif"> | <img src="https://github.com/grande-dev/pFT-ANLC/blob/master/pFT-ANLC_v1/documentation/images/Vdot_training.gif">
  
  
## Library architecture
The library architecture is composed of three main modules:  
    
1. the main file loading the configuration and the system dynamics;  
2. the file running the learning steps and training the ANNs;  
3. the postprocessing subroutines to plot the synthesised functions and to run the closed-loop tests.  
    
## Framework limitations
1. The results presented in the associated article cover 2- and 3-dimensional dynamical systems. 
  
The code currently supports:
- [x] 2-dimensional systems  
- [x] 3-dimensional systems  
- [x] 4-dimensional systems  
- [ ] >= 5-dimensional systems  
To scale up to higher dimensional systems, additional **Discrete Falsifier** functions need to be developed.
As reference, use `utilities/Function/AddLieViolationsOrder4_v4`.  
  
## Contacts
The authors can be contacted for feedback, requests of clarifications or requests of support at:  
`grande.rdev@gmail.com`
  

## Reference  
The article can be read open-access [here](https://www.sciencedirect.com/science/article/pii/S0967066124000959).
  
This work can be cited with the following BibTeX entry:  

```bibtex
@article{grande2024passive,
  title={Passive Fault-Tolerant Augmented Neural Lyapunov Control: A method to synthesise control functions for marine vehicles affected by actuators faults},
  author={Grande, Davide and Peruffo, Andrea and Salavasidis, Georgios and Anderlini, Enrico and Fenucci, Davide and Phillips, Alexander B and Kosmatopoulos, Elias B and Thomas, Giles},
  journal={Control Engineering Practice},
  volume={148},
  pages={105935},
  year={2024},
  publisher={Elsevier}
}
``` 

