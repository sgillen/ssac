# %%
from stable_baselines import PPO2, TRPO, DDPG, A2C
import seagul.envs
import gym

env_name = 'su_acrocot-v0'

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

import os
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('su_acrobot-v0')
vec_env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

models = []
model = PPO2('MlpPolicy', vec_env,
             #nb_rollout_steps=500,
             #normalize_observations=True,
             #batch_size = 512,
             verbose=False,
            )
model.learn(100)
model.save('./test.save')
# %%

model = PPO2('MlpPolicy', vec_env,
             #nb_rollout_steps=500,
             #normalize_observations=True,
             #batch_size = 512,
             verbose=False,
            )

model.load('./test.save')
action_hist = np.zeros((env.num_steps,1))
state_hist = np.zeros((env.num_steps, env.observation_space.shape[0]))
reward_hist = np.zeros((env.num_steps, 1))

obs = env.reset()

for i in range(env.num_steps):
    actions, _states, = model.predict(obs)
    # actions = np.ones(1)*100
    obs, reward, done, _ = env.step(actions)
    action_hist[i, :] = np.copy(actions)
    state_hist[i, :] = np.copy(obs)
    reward_hist[i, :] = np.copy(reward)
    env.render()
    if done:
        break
