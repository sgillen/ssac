import gym
import switched_rl.envs

from seagul.rl.run_utils import load_workspace
from seagul.plot import smooth_bounded_curve
from seagul.integration import wrap
import matplotlib.pyplot as plt

import numpy as np
from numpy import pi
from stable_baselines.results_plotter import load_results, ts2xy


import torch.utils.data
from torch.multiprocessing import Pool
from itertools import product
import os
from stable_baselines import TD3

jup_dir = os.getcwd()

def load_trials(trial_dir):
    directory = jup_dir + trial_dir

    ws_list = []
    model_list = []
    min_length = float('inf')
    for entry in os.scandir(directory):
        model, env, args, ws = load_workspace(entry.path)

        if len(ws["raw_rew_hist"]) < min_length:
            min_length = len(ws["raw_rew_hist"])

        ws_list.append(ws)
        model_list.append(model)

    min_length = int(min_length)
    rewards = np.zeros((min_length, len(ws_list)))
    for i, ws in enumerate(ws_list):
        rewards[:, i] = np.array(ws["raw_rew_hist"][:min_length])

    return ws_list, model_list, rewards


def do_rollout_switched(init_point = None):
    env = gym.make(ws['env_name'], **ws['env_config'])

    if init_point is not None:
        obs = env.reset(init_point)
    else:
        obs = env.reset()

    obs = torch.as_tensor(obs, dtype=torch.float32)

    acts_list = []; obs1_list = []
    obs2_list = []; rews_list = []
    path_list = []

    dtype = torch.float32
    act_size = env.action_space.shape[0]
    obs_size = env.observation_space.shape[0]

    done = False
    cur_step = 0

    #obs = env.reset()
    model.thresh = model.thresh_on
    full_obs = []

    while not done:
        obs1_list.append(obs)
        path = model.sig(model.gate_fn(np.array(obs, dtype=np.float32))) > model.thresh

        if path:
            model.thresh = model.thresh_off
            for _ in range(model.hold_count):
                acts = model.balance_controller(obs).reshape(-1, model.num_acts)
                obs, rew, done, _ = env.step(acts.numpy())
                full_obs.append(obs)
                obs = torch.as_tensor(obs, dtype=dtype)
        else:
            model.thresh = model.thresh_on
            # acts = model.swingup_controller(obs.reshape(-1, obs_size)).reshape(-1, model.num_acts)
            acts, val,_,logp = model.step(obs.reshape(-1,obs_size))
            acts = acts.reshape(-1, model.num_acts).detach()
            for _ in range(model.hold_count):
                obs, rew, done, _ = env.step(acts.numpy().reshape(-1))
                full_obs.append(obs)
                obs = torch.as_tensor(obs, dtype=dtype)


        acts_list.append(torch.as_tensor(acts.clone()))
        rews_list.append(torch.as_tensor(rew, dtype=dtype))
        path_list.append(path.clone())
        obs2_list.append(obs)

        cur_step += 1

    ep_obs1 = torch.stack(obs1_list)
    ep_acts = torch.stack(acts_list).reshape(-1, act_size)
    ep_rews = torch.stack(rews_list).reshape(-1, 1)
    ep_fobs = torch.tensor(full_obs).reshape(-1,4)
    ep_path = torch.tensor(path_list).reshape(-1,1)

    return ep_obs1, ep_acts, ep_rews,  ep_path, ep_fobs

def do_rollout_stable(init_point = None):
    env = gym.make(ws['env_name'], **ws['env_config'])
    model = TD3.load(
        "/home/sgillen/work/ssac/rl-baselines-zoo/baseline_log2/td3/su_acrobot_cdc-v0_2/su_acrobot_cdc-v0.zip")

    if init_point is not None:
        obs = env.reset(init_point)
    else:
        obs = env.reset()

    obs = torch.as_tensor(obs, dtype=torch.float32)

    acts_list = []; obs1_list = []
    rews_list = []

    dtype = torch.float32
    act_size = env.action_space.shape[0]
    obs_size = env.observation_space.shape[0]

    done = False
    cur_step = 0

    while not done:
        acts = model.predict(obs.reshape(-1,obs_size))[0]

        for _ in range(20):
            obs, rew, done, out = env.step(acts)

        #env.render()
        obs1_list.append(obs)
        obs = torch.as_tensor(obs, dtype=dtype)


        acts_list.append(torch.as_tensor(acts))
        rews_list.append(torch.as_tensor(rew, dtype=dtype))
        cur_step += 1

    ep_obs1 = torch.tensor(obs1_list).reshape(-1,4)
    ep_acts = torch.stack(acts_list).reshape(-1, act_size)
    ep_rews = torch.stack(rews_list).reshape(-1, 1)

    return ep_obs1, ep_acts, ep_rews,  None, ep_obs1

#%%
jup_dir = "/home/sgillen/work/"
trial_dir = "ssac/switched_rl/data_needle/50k_slow_longer"
directory = jup_dir + trial_dir

ws_list = []
model_list = []
min_length = float('inf')
for entry in os.scandir(directory):
    model, env, args, ws = load_workspace(entry.path)

    if len(ws["raw_rew_hist"]) < min_length:
        min_length = len(ws["raw_rew_hist"])

    ws_list.append(ws)
    model_list.append(model)

min_length = int(min_length)
rewards = np.zeros((min_length, len(ws_list)))
for i, ws in enumerate(ws_list):
    rewards[:, i] = np.array(ws["raw_rew_hist"][:min_length])

print("seagul", rewards[-1, :].mean(), rewards[-1, :].std())
fig, ax = smooth_bounded_curve(rewards)
ssac_size = rewards.shape[0]

color_iter = iter(['b', 'g', 'y', 'm', 'c'])
log_dir = jup_dir + 'ssac/rl-baselines-zoo/baseline_log2/'
for algo in os.scandir(log_dir):
    try:
        df_list = []
        min_length = float('inf')

        for entry in os.scandir(algo.path):
            df = load_results(entry.path)

            if len(df['r']) < min_length:
                min_length = len(df['r'])

            df_list.append(df)

        min_length = int(min_length)
        rewards = np.zeros((min_length, len(df_list)))

        for i, df in enumerate(df_list):
            rewards[:, i] = np.array(df['r'][:min_length])

        print(print(algo.path), rewards[-1, :].mean(), rewards[-1, :].std())
        smooth_bounded_curve(rewards[:ssac_size], ax=ax, color=color_iter.__next__())

    except:
        print(algo.path, "did not work")

#ax.set_ylim(0, 100)
ax.legend(['ssac (ours)', 'sac', 'ppo2', 'trpo', 'a2c', 'td3'])
ax.grid()
fig.savefig('reward.pdf')
plt.show()

pool = Pool()
th1_min = 0; th1_max = 2 * pi; num_th1 = 41
th1_vals = np.linspace(th1_min, th1_max, num_th1)

th2_min = -pi; th2_max = pi; num_th2 = 41
th2_vals = np.linspace(th2_min, th2_max, num_th2)

th_results = np.zeros((th1_vals.size, th2_vals.size))
rewards = np.zeros((th1_vals.size, th2_vals.size))
last_err = np.zeros((th1_vals.size, th2_vals.size))

end_point = torch.tensor([1.57079633, 0., 0., 0.])

import time
start = time.time()

print("lets go")

for i,res in enumerate(pool.imap(do_rollout_stable, product(th1_vals, th2_vals, [0], [0]))):
    print("did something")
    obs_hist, action_hist, reward_hist, _, _ = res
    last_err.flat[i] = torch.sum(abs(obs_hist[-1] - end_point))
    errs = torch.sum(abs(obs_hist[-1:] - end_point) , axis=1) < .5
    th_results.flat[i] = errs.all()
    rewards.flat[i] = sum(reward_hist)

#
# for i,res in enumerate(product(th1_vals, th2_vals, [0], [0])):
#     obs_hist, action_hist, reward_hist, _, _ = do_rollout_stable(res)
#     errs = torch.sum(abs(obs_hist[-10:] - end_point) , axis=1) < .2
#     th_results.flat[i] = errs.all()
#     rewards.flat[i] = sum(reward_hist)

end = time.time()
print(end - start)

# Generate "balance map" at slice th = 0
#
# dth1_min = -10; dth1_max = 10; num_dth1 = 41
# dth1_vals = np.linspace(dth1_min, dth1_max, num_dth1)
#
# dth2_min = -10; dth2_max = 10; num_dth2 = 41
# dth2_vals = np.linspace(dth2_min, dth2_max, num_dth2)
#
# dth_results = np.zeros((dth1_vals.size, dth2_vals.size))
# rewards = np.zeros((dth1_vals.size, dth2_vals.size))
#
# end_point = np.array([1.57079633, 0., 0., 0.])

import time
start = time.time()


# for i,res in enumerate(product([0],[0] , dth1_vals, dth2_vals)):
#     obs_hist, action_hist, reward_hist, _, _ = do_rollout_stable(res)
#     errs = torch.sum(abs(obs_hist[-10:] - end_point) , axis=1) < .2
#     th_results.flat[i] = errs.all()
#     rewards.flat[i] = sum(reward_hist)

#
# for i,res in enumerate(pool.imap(do_rollout_stable, product([0],[0] , dth1_vals, dth2_vals))):
#     print("did something")
#     obs_hist, action_hist, reward_hist, _, _ = res
#     errs = torch.sum(abs(obs_hist[-10:] - end_point), axis=1) < .2
#     dth_results.flat[i] = errs.all()
#     rewards.flat[i] = sum(reward_hist)


end = time.time()
print(end - start)

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
weird_list = []
for i in range(th1_vals.shape[0]):
    for j in range(th2_vals.shape[0]):
        if th_results[i, j]:
            ax.plot(th1_vals[i], th2_vals[j], 'o', color='k', alpha=1)
        else:
            ax.plot(th1_vals[i], th2_vals[j], 'o', color='r', alpha=1)


ax.set_title('Balance map (whatever that means)')
ax.set_xlabel('th1')
ax.set_ylabel('th2')
fig.savefig('th_map.pdf')
plt.show()
plt.figure()

#%%
obs_hist, act_hist, rew_hist, lqr_on, full_obs = do_rollout_stable()
#print(lqr_on)

t = np.array([i*2 for i in range(act_hist.shape[0])])
plt.step(t, act_hist, 'k')
plt.title('Actions')
plt.xlabel('Time (seconds)')
plt.ylabel('Torque (Nm)')
plt.grid()
plt.show(); plt.figure()


t = np.array([i*.01 for i in range(full_obs.shape[0])])
plt.plot(t, full_obs[:,0],'k')
plt.plot(t, full_obs[:,1],'r')

plt.axhline(np.pi/2, -1, 11,color='k',  linestyle='dashed', alpha=.5)
plt.axhline(0, -1, 11, color='k', linestyle='dashed', alpha=.5)

plt.title('States')
plt.xlabel('Time (seconds)')
plt.ylabel('Angle (rad)')
plt.legend(['th1', 'th2'])
plt.grid()
plt.show()
