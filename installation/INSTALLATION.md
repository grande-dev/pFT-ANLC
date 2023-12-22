## Installation

This software tool relies on `Python3.9`.  
The code was tested on OS: `Linux Ubuntu 20.04`, `macOS 13`. The following installation methods allow for portability over different OSs.  

Three approaches are hereby provided to streamline the installation of the following main dependencies:

- [dReal4: v4.21.6.2](https://github.com/dreal/dreal4)
- [PyTorch: 1.4.0 or 1.7.1](https://pytorch.org/get-started/locally/)
- [Numpy: 1.21.5](https://pytorch.org/get-started/locally/)
  
First, clone the repository:
```
git clone https://github.com/grande-dev/pFT-ANLC.git
cd pFT-ANLC
```

  
### Approach 1: install the requirements at system level (not recommended)
Once Python3.9 is available on your machine, you can install the required packages at system level with:
```  
pip3 install -r ./installation/requirements_v39.txt  
```


### Approach 2: clone the Anaconda environment
If [Anaconda](https://docs.anaconda.com/free/anaconda/install/) is installed on your machine, you can clone the environment with: 

```
conda env create -f ./installation/inter_platform/env_pftanlc.yml
conda activate env_pftanlc
```

(use `conda deactivate` upon completion.)


### Approach 3: create a Python virtual environment
  
Once Python3.9 is available on your machine, the code can be run in a [virtual environment](https://docs.python.org/3/library/venv.html). Start as follows:
```
pip3 install virtualenv
python3 -m venv pftanlc_venv
source pftanlc_venv/bin/activate
python -V
```

Install the required dependencies within the virtual environment with:  
```
pip3.9 install -r ./installation/requirements_v39.txt 
```

(use `deactivate` upon completion.)

### Test succesful installation
These commands will run the training for the *template system*:
```
cd pFT-ANLC_v1/code
python3 template_to_add.py
```
Upon completion, you should expect to find in 'code/results/campaign_3000/0' figures such as:   
<img src="https://github.com/grande-dev/pFT-ANLC/blob/master/pFT-ANLC_v1/documentation/images/Lyapunov_function_example.png" width=30% height=30%>
