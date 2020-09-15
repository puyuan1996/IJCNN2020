### Latent Context Based Soft Actor-Critic 
Experements for LC-SAC-Seq

### 主要
- LC-SAC-Seq主要实现位于 /spinup/algos/pytorch/sac/
- latent encoder 基于LSTM实现位于latent_encoder.py
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

### 怎样更新latent_encoder 
pi_loss q_loss同时用来更新latent encoder loss发散
只用q_loss q_loss下降，pi_loss不断上升，最终发散
- pi_loss q_loss 交替用来更新latent encoder 4:1 
- 只用pi_loss pi_loss, q_oss都可以很好的下降，没有发散

###
 Humanoid-v2
 latent_hd 128 latent_dim 20
 