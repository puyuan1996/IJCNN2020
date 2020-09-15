import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

env_id_dict = {0: 'Ant-v2', 1: 'HalfCheetah-v2', 2: 'Hopper-v2', 3: 'Humanoid-v2', 4: 'Pusher-v2', 5: 'Reacher-v2',
               6: 'Striker-v2', 7: 'Swimmer-v2', 8: 'Thrower-v2'}
learner_path_dict = {0: 'ant', 1: 'halfcheetah', 2: 'hopper', 3: 'humanoid', 4: 'pusher', 5: 'reacher', 6: 'striker',
                     7: 'swimmer', 8: 'thrower'}
env_id = env_id_dict[0]

df0=pd.read_csv("./LC-SAC-Seq_no-clear_no-next_seq_train1000_H-20_5e5/Ant-v2/2020_01_11_03_15_31/progress.csv")
df1 = pd.read_table('../sac_results/sac_Ant-v2/sac_Ant-v2_s0/progress.txt',sep='\t')

sns.set(style="darkgrid", font_scale=1.5)

ax0 = sns.lineplot(x='Number of train steps total', y='total_rewards_mean', hue=None,data=df0[:500000//1000],ci='sd', err_style='band')
ax2= sns.lineplot(x='TotalEnvInteracts', y='AverageTestEpRet', hue=None,data=df1[:500000//5000],err_style='band')

plt.xlabel('Training Steps')
plt.ylabel('Performance')
plt.title(f'{env_id}')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

plt.legend(loc='upper left',labels=['LC-SAC-Seq', 'SAC'])
# plt.legend(loc='upper left',labels=['LC-SAC', 'SAC'])
# plt.legend(loc='upper left',labels=['LC-SAC no-next', 'LC-SAC next'])

plt.show()
# plt.savefig(f'{env_id}_LC-SAC-Seq')
# plt.savefig(f'{env_id}_LC-SAC')
# plt.savefig(f'{env_id}_LC-SAC_no-next_next')
# plt.clf()  # 清除当前 figure 的所有axes，但是不关闭这个 window，所以能继续复用于其他的 plot。