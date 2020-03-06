import seagul.envs
import gym


import switched_rl.dm_gym
import datetime

import torch
import torch.nn as nn
import numpy as np
from numpy import pi
from multiprocessing import Process
from seagul.rl.run_utils import run_sg, run_and_save_bs
from seagul.rl.algos import ppo, ppo_switch
from seagul.rl.models import PPOModel, SwitchedPPOModel, PPOModelActHold, SwitchedPPOModelActHold
from seagul.nn import MLP, CategoricalMLP
import time

env_name = "dm_acrobot-v0"

# init policy, valuefn
input_size = 4
output_size = 1
layer_size = 32
num_layers = 2
activation = nn.ReLU

proc_list = []
trial_num = input("What trial is this?\n")


m1 = 1
m2 = 1
l1 = 1
l2 = 2
lc1 = .5
lc2 = 1
I1 = .083
I2 = .33
g = 9.8

def control(q):
    k = np.array([[-787.27034288, -321.29901324, -342.50908801, -158.94341213]])
    gs = np.array([pi / 2, 0, 0, 0])
    return -k.dot(q - gs)

def reward_fn(s, a):
    reward = 1e-2*np.sin(s[0]) + 2 * np.sin(s[0] + s[1])
    return reward, False

max_torque = 25

for seed in np.random.randint(0, 2**32, 4):
    for act_var in [1.0, 3.0]:
        max_t = 10.0

        model = PPOModel(
            policy=MLP(input_size, output_size, layer_size, num_layers),
            value_fn=MLP(input_size, output_size, layer_size, num_layers),
        )
            
                         

        # model = SwitchedPPOModelActHold(
        #     # policy = MLP(input_size, output_size, num_layers, layer_size, activation),
        #     policy=MLP(input_size, output_size, layer_size, num_layers),
        #     value_fn=MLP(input_size, output_size, layer_size, num_layers),
        #     gate_fn=torch.load("warm/gate_fn_good"),
        #     nominal_policy=control,
        #     hold_count=20,
        #     thresh=.85,
        # )

        env_config = {
            "init_state": [-pi/2, 0, 0, 0],
            "max_torque": max_torque,
            "init_state_weights": [1, 1, 0, 0],
            "dt": .01,
            "reward_fn" : reward_fn,
            "max_t" : max_t,
            "m2": m2,
            "m1": m1,
            "l1": l1,
            "lc1": lc1,
            "lc2": lc2,
            "i1": I1,
            "i2": I2,
            "act_hold" : 1
        }

        alg_config = {
            "env_name": env_name,
            "model": model,
            "act_var_schedule": [act_var],
            "seed": seed,  # int((time.time() % 1)*1e8),
            "total_steps" : 1e6,
            "epoch_batch_size": 2048,
            "reward_stop" : None,
            "gamma": 1,
            "pol_epochs": 30,
            "val_epochs": 30,
            "env_config" : env_config,
            "target_kl" : .1
        }

        run_name = "swingup" + str(seed)

        p = Process(target=run_sg, args=(alg_config, ppo, run_name , "normal reward", "/data4/switching_sortof/trial" + str(trial_num) + "_t" +  str(act_var) + "/"))
        p.start()
        proc_list.append(p)


for p in proc_list:
    p.join()

