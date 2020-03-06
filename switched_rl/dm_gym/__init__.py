from gym.envs.registration import register

register(id='dm_acrobot-v0', entry_point="switched_rl.dm_gym.acrobot:DMAcroEnv")
