import gym
import switched_rl.envs
from seagul.rl.run_utils import load_workspace
from seagul.plot import smooth_bounded_curve
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from stable_baselines import TD3
from stable_baselines.results_plotter import load_results
import torch.utils.data
from torch.multiprocessing import Pool
from itertools import product
import os
import matplotlib

script_path = os.path.realpath(__file__).split("/")[:-1]
script_path = "/".join(script_path) + "/"
print(script_path)

model, env, args, ws = load_workspace(script_path + "data_needle/50k_slow_longer/trial3738150792--3-11_18-0")
env_name = ws['env_name']
config = ws['env_config']


def load_trials(trial_dir):
    directory = script_path + trial_dir

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


def do_rollout_switched(init_point=None):
    env = gym.make(env_name, **config)

    if init_point is not None:
        obs = env.reset(init_point)
    else:
        obs = env.reset()

    obs = torch.as_tensor(obs, dtype=torch.float32)

    acts_list = [];
    obs1_list = []
    obs2_list = [];
    rews_list = []
    path_list = []

    dtype = torch.float32
    act_size = env.action_space.shape[0]
    obs_size = env.observation_space.shape[0]

    done = False
    cur_step = 0

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
            acts, val, _, logp = model.step(obs.reshape(-1, obs_size))
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
    ep_fobs = torch.tensor(full_obs).reshape(-1, 4)
    ep_path = torch.tensor(path_list).reshape(-1, 1)

    return ep_obs1, ep_acts, ep_rews, ep_path, ep_fobs


def do_rollout_stable(init_point=None):
    env = gym.make(env_name, **config)
    td3_model = TD3.load(script_path + "../rl-baselines-zoo/baseline_log2/td3/su_acrobot_cdc-v0_2/su_acrobot_cdc-v0.zip")

    if init_point is not None:
        obs = env.reset(init_point)
    else:
        obs = env.reset()

    obs = torch.as_tensor(obs, dtype=torch.float32)

    acts_list = [];
    obs1_list = []
    rews_list = []

    dtype = torch.float32
    act_size = env.action_space.shape[0]
    obs_size = env.observation_space.shape[0]

    done = False
    cur_step = 0

    while not done:
        acts = td3_model.predict(obs.reshape(-1, obs_size))[0]

        for _ in range(20):
            obs, rew, done, out = env.step(acts)

        # env.render()
        obs1_list.append(obs)
        obs = torch.as_tensor(obs, dtype=dtype)

        acts_list.append(torch.as_tensor(acts))
        rews_list.append(torch.as_tensor(rew, dtype=dtype))
        cur_step += 1

    ep_obs1 = torch.tensor(obs1_list).reshape(-1, 4)
    ep_acts = torch.stack(acts_list).reshape(-1, act_size)
    ep_rews = torch.stack(rews_list).reshape(-1, 1)

    return ep_obs1, ep_acts, ep_rews, None, ep_obs1

trial_dir = "data_needle/50k_slow_longer"
directory = script_path + trial_dir

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
ssac_size = rewards.shape[0]

shifted_reward = np.nan*np.ones((int(2e6/51),8))
shifted_reward[int(1e6/51):int(1e6/51) + ssac_size] = rewards
fig, ax = smooth_bounded_curve(shifted_reward, time_steps=[51*i for i in range(shifted_reward.shape[0])])

color_iter = iter(['b', 'g', 'y', 'm', 'c'])
log_dir = script_path + '../rl-baselines-zoo/baseline_log2/'
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
        smooth_bounded_curve(rewards[:min_length], time_steps=[51*i for i in range(min_length)], ax=ax, color=color_iter.__next__())
        print(algo.path)

    except:
        print(algo.path, "did not work")

ax.legend(['ssac (ours)', 'trpo', 'sac', 'a2c', 'ppo', 'td3'])
ax.grid()
ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
fig.savefig(script_path + '../figs/reward.png')
plt.show()

obs_hist, act_hist, rew_hist, lqr_on, full_obs = do_rollout_switched(init_point=np.array([-pi/2, 0, 0, 0]))

t = np.array([i*2 for i in range(act_hist.shape[0])])
plt.step(t, act_hist, 'k')
plt.title('Actions')
plt.xlabel('Time (seconds)')
plt.ylabel('Torque (Nm)')
plt.grid()
plt.savefig(script_path + '../figs/act_hist.png')
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
plt.savefig(script_path + '../figs/obs_hist.png')
plt.show()

font = {'family': 'normal',
        'weight': 'bold',
        'size': 18}

matplotlib.rc('font', **font)


# Generate "balance map" at slice dth = 0
with Pool() as pool:
    th1_min = 0; th1_max = 2 * pi; num_th1 = 41
    th1_vals = np.linspace(th1_min, th1_max, num_th1)

    th2_min = -pi; th2_max = pi; num_th2 = 41
    th2_vals = np.linspace(th2_min, th2_max, num_th2)

    th_results = np.zeros((th1_vals.size, th2_vals.size))
    rewards = np.zeros((th1_vals.size, th2_vals.size))

    goal_state = torch.tensor([1.57079633, 0., 0., 0.])

    for i,res in enumerate(pool.imap(do_rollout_switched, product(th1_vals, th2_vals, [0], [0]))):
        obs_hist, action_hist, reward_hist, _, _ = res
        errs = torch.sum(abs(obs_hist[-10:] - goal_state), axis=1) < .2
        th_results.flat[i] = errs.all()
        rewards.flat[i] = sum(reward_hist)

    pool.close()
    pool.join()

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
for i in range(th1_vals.shape[0]):
    for j in range(th2_vals.shape[0]):
        if th_results[i, j]:
            ax.plot(th1_vals[i], th2_vals[j], 'o', color='k', alpha=1)
        else:
            ax.plot(th1_vals[i], th2_vals[j], 'o', color='r', alpha=1)

ax.set_title('Balance map SSAC')
ax.set_xlabel(r'Initial $\theta_{1}$')
ax.set_ylabel(r'Initial $\theta_{2}$')
fig.savefig(script_path + '../figs/th_map_ssac.png')
plt.show()
plt.figure()

# %%

with Pool() as pool:
    th1_min = 0;
    th1_max = 2 * pi;
    num_th1 = 41
    th1_vals = np.linspace(th1_min, th1_max, num_th1)

    th2_min = -pi;
    th2_max = pi;
    num_th2 = 41
    th2_vals = np.linspace(th2_min, th2_max, num_th2)

    th_results = np.zeros((th1_vals.size, th2_vals.size))
    rewards = np.zeros((th1_vals.size, th2_vals.size))
    last_err = np.zeros((th1_vals.size, th2_vals.size))

    goal_state = torch.tensor([1.57079633, 0., 0., 0.])

    for i, res in enumerate(pool.imap(do_rollout_stable, product(th1_vals, th2_vals, [0], [0]))):
        obs_hist, action_hist, reward_hist, _, _ = res
        last_err.flat[i] = torch.sum(abs(obs_hist[-1] - goal_state))
        errs = torch.sum(abs(obs_hist[-1:] - goal_state), axis=1) < .5
        th_results.flat[i] = errs.all()
        rewards.flat[i] = sum(reward_hist)

        pool.close()
        pool.join()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    for i in range(th1_vals.shape[0]):
        for j in range(th2_vals.shape[0]):
            if th_results[i, j]:
                ax.plot(th1_vals[i], th2_vals[j], 'o', color='k', alpha=1)
            else:
                ax.plot(th1_vals[i], th2_vals[j], 'o', color='r', alpha=1)

    ax.set_title('Balance map TD3')
    ax.set_xlabel(r'Initial $\theta_{1}$')
    ax.set_ylabel(r'Initial $\theta_{2}$')
    fig.savefig(script_path + '../figs/th_map_td3.png')
    plt.show()
    plt.figure()

# %%

net = model.gate_fn

n_thdot = 1
n_th = 1000

th1_vals = np.linspace(0, pi, n_th)
th2_vals = np.linspace(pi/2, -pi/2, n_th)

th1dot_vals = np.linspace(-10, 10, n_th)
th2dot_vals = np.linspace(-30, 30, n_th)

coords = np.zeros((n_th, n_th, 4), dtype=np.float32)
for i, j in product(range(n_th), range(n_th)):
    coords[j, i, :] = np.array([th1_vals[i], th2_vals[j], 0, 0])

sig = torch.nn.Sigmoid()  # BCEWithLogits included the sigmoid layer for us, but we need to apply it ourselves outside that context
preds = sig(net(coords.reshape(-1, 4)).reshape(n_th, n_th).detach())

fig, ax = plt.subplots(n_thdot, n_thdot, figsize=(8, 8))
# generate 2 2d grids for the x & y bounds
x, y = np.meshgrid(th1_vals, th2_vals)
z = preds

# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]
z_min, z_max = 0, np.abs(z).max()

c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.grid()
ax.set_title('Gate map slice at dth1 = dth2 = 0, after training')
ax.set_xlabel('Th1')
ax.set_ylabel('Th2')

# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)
plt.show()
plt.figure()

coords = np.zeros((n_th, n_th, 4), dtype=np.float32)

for i, j in product(range(n_th), range(n_th)):
    coords[j, i, :] = np.array([pi / 2, 0, th1dot_vals[i], th2dot_vals[j]])

preds = sig(net(coords.reshape(-1, 4)).reshape(n_th, n_th).detach())

fig, ax = plt.subplots(n_thdot, n_thdot, figsize=(8, 8))
# generate 2 2d grids for the x & y bounds
x, y = np.meshgrid(th1dot_vals, th2dot_vals)
z = preds

# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]
z_min, z_max = 0, np.abs(z).max()

c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('Gate map slice at th1 = th2 = 0')
ax.set_xlabel('dth1')
ax.set_ylabel('dth2')
# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)
plt.show()

