#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control import suite
from dm_control import viewer

import switched_rl.dm_gym
import gym
import numpy as np


env = gym.make('dm_acrobot-v0')

# env = suite.load('acrobot', 'swingup')
# action_spec = env.action_spec()
#
# # Define a uniform random policy.
# def random_policy(time_step):
#     del time_step  # Unused.
#     return np.random.uniform(low=action_spec.minimum,
#                            high=action_spec.maximum,
#                            size=action_spec.shape)

#viewer.launch(env, policy=random_policy)
