from gym.envs.registration import register
import numpy as np
import seagul.envs

def reward_fn_sin(s, a):
    reward = (np.sin(s[0]) + np.sin(s[0] + s[1]))
    return reward, False

env_config = {'init_state': [-1.5707963267948966, 0, 0, 0],
              'max_torque': 25,
              'init_state_weights': [3.141592653589793, 3.141592653589793, 0, 0],
              'dt': 0.01,
              'reward_fn': reward_fn_sin,
              'max_t': 10.0,
              'm2': 1,
              'm1': 1,
              'l1': 1,
              'lc1': 0.5,
              'lc2': 0.5,
              'i1': 0.2,
              'i2': 1.0,
              'act_hold':20,
}

register(id="su_acrobot_cdc-v0", entry_point="switched_rl/envs:SGAcroEnv", kwargs=env_config)
