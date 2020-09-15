import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

env_id_dict = {0: 'Ant-v2', 1: 'HalfCheetah-v2', 2: 'Hopper-v2', 3: 'Humanoid-v2', 4: 'Pusher-v2', 5: 'Reacher-v2',
               6: 'Striker-v2', 7: 'Swimmer-v2', 8: 'Thrower-v2'}
learner_path_dict = {0: 'ant', 1: 'halfcheetah', 2: 'hopper', 3: 'humanoid', 4: 'pusher', 5: 'reacher', 6: 'striker',
                     7: 'swimmer', 8: 'thrower'}
env_id = env_id_dict[1]

birth_data = []
with open('../sac_results/sac_HalfCheetah-v2/sac_HalfCheetah-v2_s0/progress.txt') as txtfile:
    # for line in open('../sac_results/sac_HalfCheetah-v2/sac_HalfCheetah-v2_s0/progress.txt'):
    # birth_data = txtfile.readlines()
    # birth_data_str = txtfile.read()
    for i, row in enumerate(txtfile):
        # print(i,type(row))
        if i == 0:
            birth_header = row.split()
        else:
            birth_data.append(row.split())
        # if i==5:
        #     break
# birth_data= np.loadtxt('../sac_results/sac_HalfCheetah-v2/sac_HalfCheetah-v2_s0/progress.txt')

birth_data = [[float(x) for x in row] for row in birth_data]  # 将数据从string形式转换为float形式

birth_data = np.array(birth_data)  # 将list数组转化成array数组便于查看数据结构
# birth_header = np.array(birth_header)
print(birth_data.shape)  # 利用.shape查看结构。
# print(birth_header.shape)
x = birth_data[:, birth_header.index('TotalEnvInteracts')]
y = birth_data[:, birth_header.index('AverageEpRet')]
plt.plot(x, y, '-', label='SAC', color='g')

plt.xlabel('train steps')
plt.ylabel('average return')

plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
# plt.tight_layout(pad=0.5)

plt.title(f'{env_id}')
plt.legend()

plt.show()
# plt.savefig(f'{env_id}')
