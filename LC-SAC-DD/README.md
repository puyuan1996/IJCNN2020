This code is modified from https://github.com/katerakelly/oyster  
and https://github.com/hiwonjoon/ICML2019-TREX.

### Latent Context Based Soft Actor-Critic 
Experements for LC-SAC-DD

- 1 运行gnerate_demos/generate_trajs.py 产生LC-SAC-DD所需的轨迹 
- 2 运行main.py

LC--SAC-DD与LC-SAC-Seq的主要不同：
- 1 增加/generate_demos 与 /generated_demos 2个文件夹 
- 2 /spinup/algos/pytorch/sac/sac.py train() collect_data_from_demos()
