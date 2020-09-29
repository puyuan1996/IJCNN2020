### Latent Context Based Soft Actor-Critic 
Experements for LC-SAC-Seq

### 主要
- LC-SAC-Seq主要实现位于 /spinup/algos/pytorch/sac/
- latent encoder MLP和LSTM实现位于latent_encoder.py
- test_model.py 测试训练得到的model


### 网络结构

q, pi networks (256,256)
latent_encoder (128,lstm,128)

### 测试实验 
main_sac.py
agent_sac.py
sac.py
把z置为0，hidden_dim=1, latent_encoder hidden_size 1
即测试原sac算法是否可行，实验证明是可以跑出spinup sac的结果


### sac_q_mlp.py
latent encoder为MLP
q,pi 和latent encoder交替训练
训练latent encoder时loss为q loss和kld
rl训练时，将z存入rl replay buffer,
Ant-v2 HalfCheetah-v2 只能在这个条件下跑出结果
 
 ### sac_q_v3.py
q,pi 和latent encoder交替训练
训练latent encoder时loss为q loss和kld
rl训练时，将z存入rl replay buffer,
Ant-v2 HalfCheetah-v2 只能在这个条件下跑出结果

### sac_q_v4.py
q,pi 和latent encoder交替训练
训练latent encoder时loss为q loss和kld
rl训练时，将z存入rl replay buffer,s
采样得到长度为8的sequence,传入encoder,得到的8个z，得到4步的q_loss,相加后梯度回传更新encoder参数
