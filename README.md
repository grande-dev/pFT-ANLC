# Passive Fault Tolerant-Augmented Neural Lyapunov Control
This repository contains the code for the paper:  
**Passive Fault-Tolerant Augmented Neural Lyapunov Control: a method to synthesise control functions for marine vehicles affected by actuators faults** (pFT-ANLC)
  
The work can be read at [webpage link to be updated](https://ieeexplore.ieee.org/document/10171339).  
  
  
## Scope of the code
*pFT-ANLC* is a software framework to **automatically** synthesise:  
1. a **stabilising controller** for a desired equilibrium of a nonlinear system affected by **faults**;  
2. a **Control Lyapunov Function** (CLF) to certify its stability.    
  
The code is based on a loop between a *Learner* and a *Falsifier*. Starting from a finite set of state-space samples, the *Learner* trains two Neural Networks, one representing a control law and the other a CLF. In parallel, the *Falsifier* is tasked with verifying whether the candidate CLF satisfies the theoretical Lyapunov conditions within the dense domain over the Reals. If the conditions are satisifed, the learning is halted and the resulting control law and CLF are returned. On the contrary, if the conditions are not satisfied, the Falsifier returns a set of points (denoted as counterexample) where the Lyapunov conditions are violated. These points are added to the dataset and the learning process is further iterated. 
The system attempts at simultaneously stabilize a set of dynamics, ecompassing the nominal dynamics (fault-free) and the dynamics associated to each fault.
  
   
A schematic of the learning architecture is hereby illustrated:  
<img src="https://github.com/grande-dev/pFT-ANLC/blob/master/pFT-ANLC_v1/documentation/images/learning_scheme.png" width=100% height=100%>
  
  
The *pFT-ANLC* software framework can:
- [x] Synthesise both linear and nonlinear control laws;  
- [x] Control functions (linear and nonlinear) synthesised without initialising the control weights;
- [x] Option to stabilise generic equilibria (not necessarily the origin);
- [x] Cope with complete loss of actuator, partial loss of actuator efficiency, or jammed actuators.  
  
## Installation  
Instructions on installation are available within the ![INSTALLATION](./pFT-ANLC_v1/INSTALLATION.md) file.    
  
## Step-by-step guide
To synthesise control laws and CLFs for your own dynamics, a step-by-step example is reported in the ![WIKI](./pFT-ANLC_v1/documentation/WIKI.md) file.  
  
Hereby an example of how a CLF and Lie derivative(s) are updated over the training iterations. Note the yellow patches of the Lie derivative gradually disappearing as the training proceeds. 
  
CLF evolution |  Lie derivative function evolution                    
:-------------------------:|:-------------------------:
<img src="https://github.com/grande-dev/pFT-ANLC/blob/master/pFT-ANLC_v1/documentation/images/V_training.gif"> | <img src="https://github.com/grande-dev/pFT-ANLC/blob/master/pFT-ANLC_v1/documentation/images/Vdot_training.gif">
  
  
## Library architecture
The library architecture is composed of three main modules:  
    
1. the main file loading the configuration and the system dynamics;  
2. the CEGIS file training the ANNs and calling the Falsifier;  
3. the postprocessing subroutines to plot the synthesised functions and to run the closed-loop tests.  
    
## Framework limitations
1. The results presented cover 2- and 3-dimensional (nonlinear) systems. 
 
The code currently supports:
- [x] 1-dimensional systems  
- [x] 2-dimensional systems  
- [x] 3-dimensional systems  
- [ ] >= 4-dimensional systems  
To scale up to higher dimensional systems, additional **Discrete Falsifier** functions need to be developed.
As reference, use `utilities/Function/AddLieViolationsOrder3_v4`.  
  
## Contacts
The authors can be contacted for feedback, clarifications or requests of support at:  
`grande.rdev@gmail.com`
  

## Reference  
The article can be accessed at [webpage link to be updated](https://ieeexplore.ieee.org/document/10171339/).
  
This work can be cited with the following BibTeX entry:  

```bibtex
@article{,
  title={Passive Fault-Tolerant Augmented Neural Lyapunov Control: a method to synthesise control functions for marine vehicles affected by actuators faults},
  author={Davide Grande, and Andrea, Peruffo and Georgios, Salavasidis and Enrico, Anderlini and Davide, Fenucci and Alexander B. Phillips and Elias B. Kosmatopoulos and Giles Thomas},
  journal={},
  volume={},
  pages={},
  year={},
  publisher={},
  doi={}
}
``` 

