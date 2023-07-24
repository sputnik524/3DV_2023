# 252-0579-00L 3D Vision FS2023
This is the repository for the course project of 3D Vision from group 19.  

## Project name : Multi-Agent Reinforcement Learning Based Sample Consensus for Robust Estimation

## Before run
### On Euler
#### Load proper modules
```
module load gcc/8.2.0
module load python_gpu/3.10.4
module load eigen/3.3.4
module load cuda
module load eth_proxy
```
Make sure you load the correct modules by `module list` , the output should be like this.
```
Currently Loaded Modules:
  1) StdEnv      3) openblas/0.3.15   5) nccl/2.11.4-1       7) eigen/3.3.9   9) eth_proxy
  2) gcc/8.2.0   4) cudnn/8.2.1.32    6) python_gpu/3.10.4   8) cuda/11.8.0
```
Now, follow the steps in "On Desktop" part.
### On Desktop
First install the poselib by:
```
git clone https://github.com/vlarsson/PoseLib.git
cd PoseLib && cmake -S . -B _build/ -DPYTHON_PACKAGE=ON -DCMAKE_INSTALL_PREFIX=_install
cmake --build _build/ --target install -j 20
cmake --build _build/ --target pip-package
cmake --build _build/ --target install-pip-package
```

Ensure all the dependencies are installed by:
```
pip install -r requirements.txt
```

## Run code

Since this repository contains two different routes of this project, their corresponding running commands are given in their directories respectively. 

* MA-RLSAC: <code>src/ </code>

* MA-SAC: <code>src_tianshou_pipe/ </code>

Please check there!
