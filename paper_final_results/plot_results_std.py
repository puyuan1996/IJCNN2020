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
env_id = env_id_dict[1]
# 改变相关的路径，画不同实验的图
df0 = pd.read_csv('./LC-SAC-Seq_train4000_no-clear_H-20_xe6/HalfCheetah-v2/2020_01_13_04_47_43/progress.csv')
df1 = pd.read_csv('./LC-SAC-Seq_train4000_no-clear_H-20_xe6/HalfCheetah-v2/2020_01_13_08_36_18/progress.csv')
df2 = pd.read_table('../sac_results/sac_HalfCheetah-v2/sac_HalfCheetah-v2_s0/progress.txt',sep='\t')

# data=df1
# print(df1.head(n=0)) # Head of the data

data = [df0,df1]

if isinstance(data, list):
    data = pd.concat(data, ignore_index=True)

sns.set(style="darkgrid", font_scale=1.5)
ax0 = sns.lineplot(x='Number of train steps total', y='total_rewards_mean', hue=None,data=data, err_style='band')#,legend='LC-SAC')#, **kwargs)
ax1= sns.lineplot(x='TotalEnvInteracts', y='AverageEpRet', hue=None,data=df2[:int(2e6)//5000], err_style='band')#,legend='SAC')#, **kwargs)
# handles, labels = ax0.get_legend_handles_labels()

# plt.plot(x, y, '-', label='LC-SAC', color='g')

# plt.xlabel('train steps')
# plt.ylabel('average return')
plt.title(f'{env_id}')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

plt.legend(loc='upper left',labels=['LC-SAC-Seq', 'SAC'])

plt.show()