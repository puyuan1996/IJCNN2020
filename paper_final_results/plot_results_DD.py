import pandas as pd
import seaborn as sns
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
# mpl.rcParams['font.size'] = 8

env_id_dict = {0: 'Ant-v2', 1: 'HalfCheetah-v2', 2: 'Hopper-v2', 3: 'Humanoid-v2', 4: 'Pusher-v2', 5: 'Reacher-v2',
               6: 'Striker-v2', 7: 'Swimmer-v2', 8: 'Thrower-v2'}
learner_path_dict = {0: 'ant', 1: 'halfcheetah', 2: 'hopper', 3: 'humanoid', 4: 'pusher', 5: 'reacher', 6: 'striker',
                     7: 'swimmer', 8: 'thrower'}

env_id = env_id_dict[1]
df0 = pd.read_csv('./LC-SAC-DD_train4000_no-clear_H-100/HalfCheetah-v2/2020_01_12_23_36_42/progress.csv')

# data = []
data = [df0]
if isinstance(data, list):
    data = pd.concat(data, ignore_index=True)

sns.set(style="darkgrid", font_scale=1.5)

# ax0 = sns.lineplot(x='Number of train steps total', y='total_rewards_mean', hue=None,data=data[:2000000//4000],ci='sd', err_style='band')
ax0 = sns.lineplot(x='Number of train steps total', y='total_rewards_mean', hue=None, data=df0, ci='sd', err_style='band')

with open('../LC-SAC-DD/generated_demos/' + f'/{env_id}' + '/agent_rewards' + '.pickle', 'rb') as f:
    agent_rewards = pickle.load(f)
    # x = np.arange(25)
    x = np.linspace(0, 2e6 - 20 * 4000, 25)
    # plt.plot(x, agent_rewards, marker='o', linestyle='-', color='g')
    plt.plot(x, agent_rewards, 'g-o')

    plt.xlabel('stage')
    plt.ylabel('agent_rewards')
    plt.title(f'{env_id}')
    # plt.legend()

plt.xlabel('Training Steps')
plt.ylabel('Performance')
plt.title(f'{env_id}')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

# plt.legend(loc='upper left', labels=['LC-SAC-DD H=20', 'demos average return'])
plt.legend(loc='upper left', labels=['LC-SAC-DD H=100', 'demos average return'])
plt.show()
# # 默认保存为png格式
# plt.savefig(f'{env_id}_LC-SAC-DD')
