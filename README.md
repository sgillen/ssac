## Switched Soft Actor Critic

This repository contains code to accompany the paper !!1 paper

This is a python package, to use it please install it into your python environment (highly recommend some sort of virtual environment) wiht

```
pip install switched_rl/
```

You can then install the requirements with

```
$ pip install requirements.txt
```

## Directory Structure

- rl-baselines-zoo/  contains a fork of https://github.com/araffin/rl-baselines-zoo, we modified the code to use our custom acrobot environment, and added a script to launch the algorithms we wanted to compare with their specific hyper parameters. 
- switched_rl/gate_generation.ipynb notebook walking through how we train the gating function
- switched_rl/ssac.py This contains the implementation of SSAC as described in the paper
- switched_rl/make_figures.py This is the script that actually generates the images found in the paper

