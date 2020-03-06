from gym import core, spaces
from dm_control import suite
import numpy as np

class DMAcroEnv(core.Env):
    dtype = np.float32

    def __init__(self):
        self.dm_env = suite.load('acrobot', 'swingup')

        act_spec = self.dm_env.action_spec()
        self.act_high = np.array(act_spec.maximum, dtype=self.dtype)
        self.act_low = np.array(act_spec.minimum, dtype=self.dtype)
        self.action_space = spaces.Box(low=self.act_low, high=self.act_high, dtype=self.dtype)

        obs_low = np.array([-1, -1, -1, -1, -float('inf'), -float('inf')])
        obs_high = -obs_low
        self.observation_space = spaces.Box(low=obs_low-.2, high=obs_high+.2, dtype=self.dtype)

    def reset(self):
        time, reward, discount, dm_obs = self.dm_env.reset()
        obs = np.concatenate((dm_obs['orientations'], dm_obs['velocity']), dtype=self.dtype)
        return obs

    def step(self, act):
        act = np.clip(act, -self.act_low, self.act_high)
        time, reward, discount, dm_obs = self.dm_env.step(act)

        if time.last():
            done = True

    def render(self, mode='human'):
        raise NotImplementedError
