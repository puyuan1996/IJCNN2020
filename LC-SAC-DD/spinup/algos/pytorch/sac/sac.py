from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.sac.core as core
from spinup.utils.logx import EpochLogger
from .replay_buffer import ReplayBuffer
from .sampler import obtain_rollout_samples
from tensorboardX import SummaryWriter
import os
import pytorch_util as ptu


class SAC(object):
    def __init__(self, env, env_name, agent, q1_net, q2_net, ac_kwargs=dict(), seed=0,
                 steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
                 polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, seq_len=20, start_steps=10000,
                 update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
                 save_freq=10, model_path='./model/', device='cpu', train_steps=1000
                 , collect_data_samples=10000, logger_kwargs=dict()):
        super().__init__()
        self.env, self.test_env = env, env
        self.env_name = env_name
        self.agent = agent
        self.q1_net = q1_net
        self.q2_net = q2_net
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.lr = lr
        self.alpha = alpha
        # self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq
        self.model_path = model_path
        self.kl_lambda = 0.1
        self.latent_batch_size = batch_size
        self.seq_len = seq_len
        self.total_train_steps = 0
        self.total_env_steps = 0
        self.duration = 0
        self.use_next_obs_in_context = False
        self.device = device
        self.train_steps = train_steps
        self.collect_data_samples = collect_data_samples
        # self.sampler=sampler
        self.writer = SummaryWriter('tblogs')
        # self.logger = EpochLogger(**logger_kwargs)
        # self.logger.save_config(locals())

        torch.manual_seed(seed)
        np.random.seed(seed)

        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = env.action_space.high[0]

        # Create actor-critic module and target networks

        self.q1_net_targ = deepcopy(q1_net)
        self.q2_net_targ = deepcopy(q2_net)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q1_net_targ.parameters():
            p.requires_grad = False
        for p in self.q2_net_targ.parameters():
            p.requires_grad = False
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(q1_net.parameters(), q2_net.parameters())

        # Experience buffer
        self.rl_replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        self.latent_replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [agent.latent_encoder, agent.pi, q1_net, q2_net])
        # self.logger.log('\nNumber of parameters: \t encoder:%d,\t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)
        print('\nNumber of parameters: \t encoder:%d,\t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(agent.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)
        self.latent_optimizer = Adam(agent.latent_encoder.parameters(), lr=lr)

        # Set up model saving
        # logger.setup_pytorch_saver(ac)

    def get_action(self, o, deterministic=False):
        with torch.no_grad():
            a, _ = self.agent.policy(torch.as_tensor(o, dtype=torch.float32), deterministic, False)
            return a.numpy()

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data, z):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        o.squeeze_(0)
        o = o.to(self.device)
        a = a.to(self.device)
        o2 = o2.to(self.device)
        r = r.to(self.device)
        d = d.to(self.device)

        q1 = self.q1_net(o, a, z)
        q2 = self.q2_net(o, a, z)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.agent.pi.get_act_logp_from_o_z(o2, z)

            # Target Q-values
            q1_pi_targ = self.q1_net_targ(o2, a2, z)
            q2_pi_targ = self.q2_net_targ(o2, a2, z)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data, pi, logp_pi, z):
        o = data['obs']
        o = o.to(self.device)
        # pi, logp_pi = self.agent.pi(o, z)

        q1_pi = self.q1_net(o, pi, z)
        q2_pi = self.q2_net(o, pi, z)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def update_step(self, context_batch_indices, context_seq_batch):

        data = self.rl_replay_buffer.sample_data(context_batch_indices)
        # data = data.to(self.device)
        policy_outputs, z = self.agent(data['obs'].to(self.device), context_seq_batch.to(self.device))
        # new_actions, policy_mean, policy_log_std, logp_pi = policy_outputs[:4]
        pi_action, logp_pi = policy_outputs

        # KL constraint on z if probabilistic
        self.latent_optimizer.zero_grad()

        # if self.use_information_bottleneck:
        # kl_div = self.agent.compute_kl_div()
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.sum(1 + self.agent.z_logvar - self.agent.z_mu.pow(2) - self.agent.z_logvar.exp())
        kl_loss = self.kl_lambda * kl_div
        kl_loss.backward(retain_graph=True)

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()

        loss_q, q_info = self.compute_loss_q(data, z)
        loss_q.backward(retain_graph=True)
        self.q_optimizer.step()

        self.latent_optimizer.step()

        # Record things
        # self.logger.store(LossQ=loss_q.item(), **q_info)
        self.writer.add_scalar('LossQ', float(loss_q.item()), self.total_train_steps)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data, pi_action, logp_pi, z.detach())
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        # self.logger.store(LossPi=loss_pi.item(), **pi_info)
        self.writer.add_scalar('LossPi', float(loss_pi.item()), self.total_train_steps)
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.q1_net.parameters(), self.q1_net_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(self.q2_net.parameters(), self.q2_net_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def train_step(self):
        seq_len = 20  # self.embedding_mini_batch_size  # 100
        context_bs = 80  # self.embedding_batch_size // mb_size  # 100//100

        # sample context batch
        indices, data = self.rl_replay_buffer.random_sequence(batch_size=self.latent_batch_size)
        # data = data.to(self.device)
        # data = { obs，obs2, act, rew, done}
        if self.use_next_obs_in_context:
            context = torch.cat((data['obs'], data['obs2'], data['act'], data['rew'].unsqueeze(-1)), dim=-1)
        else:
            context = torch.cat((data['obs'], data['act'], data['rew'].unsqueeze(-1)), dim=-1)
        context.unsqueeze_(0)
        # context = (obs, act, rew)
        """采集连续100个transitions 
        """

        # zero out context and hidden encoder state
        self.agent.clear_z()
        context_seq_batch = []
        context_batch_indices = []
        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(context_bs):  # context batch size
            # context_seq = context[:, i: i + seq_len, :]
            context_seq = context[:, i: i + seq_len, :]
            context_seq_batch.append(context_seq.squeeze())
            context_batch_indices.append(indices[i + seq_len])
        """从采集的连续100 steps的 transitions 以步长为20，分为80个sequences
        """
        context_seq_batch = torch.stack(context_seq_batch)
        # context_seq_batch = context_seq_batch.to(self.device)
        # (80，20，dim) dim=obs_dim + act_dim + 1      e=s,a,r
        # 比如hopper dim=11+3+1=15
        self.update_step(context_batch_indices, context_seq_batch)  # sac rl update

        # stop backprop
        self.agent.detach_z()

    def train_step_new(self):
        # self.latent_batch_size = 100
        # self.seq_len =20
        self.agent.clear_z()
        context_seq_batch = []
        context_batch_indices = []
        # sample context batch
        for i in range(self.latent_batch_size):  # 随机采集100个长度为seq_len的sequence
            indices, data = self.rl_replay_buffer.random_sequence(seq_len=self.seq_len)
            # 采集长度为seq_len的1个sequence
            # data = data.to(self.device)
            # data = { obs，obs2, act, rew, done}
            if self.use_next_obs_in_context:
                context = torch.cat((data['obs'], data['obs2'], data['act'], data['rew'].unsqueeze(1)), dim=1)
            else:
                context = torch.cat((data['obs'], data['act'], data['rew'].unsqueeze(1)), dim=1)  # 20,15  (seq_len, feat)
            # context.unsqueeze_(0)
            # context = (obs, act, rew)

            context_seq_batch.append(context)
            context_batch_indices.append(indices[-1] + 1)

        context_seq_batch = torch.stack(context_seq_batch)  # 100,20,15  (batch_size, seq_len, feat)
        # context_seq_batch = context_seq_batch.to(self.device)
        # (80，20，dim) dim=obs_dim + act_dim + 1      e=s,a,r
        # 比如hopper dim=11+3+1=15
        self.update_step(context_batch_indices, context_seq_batch)  # sac rl update

        # stop backprop
        self.agent.detach_z()

    def collect_data(self, nums_steps, add_to_latent_buffer=True):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param nums_steps: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        self.agent.clear_z()
        cur_steps = 0
        while cur_steps < nums_steps:
            paths, nums_steps = obtain_rollout_samples(self.env, self.agent, max_samples=nums_steps - cur_steps,
                                                       max_path_length=self.max_ep_len,
                                                       max_nums_paths=1000)
            cur_steps += nums_steps
            self.rl_replay_buffer.add_paths(paths)
            # if add_to_latent_buffer:
            #     self.latent_replay_buffer.add_paths(paths)
        print('rew_mean:', np.array([path['rew'].sum() for path in paths]).mean())
        self.total_env_steps += cur_steps

    def collect_data_from_demos(self, trajs):
        # start from the prior
        self.agent.clear_z()

        num_transitions = 0
        paths = []
        for i in range(len(trajs)):  # 200
            observations, actions, rewards, next_observations, terminals = trajs[i]
            agent_infos = np.zeros_like(rewards.reshape(-1, 1))
            env_infos = np.zeros_like(rewards.reshape(-1, 1))
            num_transitions += len(trajs[i][0])

            path = dict(
                observations=observations,
                actions=actions,
                rewards=np.array(rewards).reshape(-1, 1),
                next_observations=next_observations,
                terminals=np.array(terminals).reshape(-1, 1),
                agent_infos=agent_infos,
                env_infos=env_infos,
            )
            paths.append(path)

        self.rl_replay_buffer.add_paths(paths)
        # self.enc_replay_buffer.add_paths(self.task_idx, paths)

        # if update_posterior_rate != np.inf:
        #     context = self.sample_context(self.task_idx)
        #     self.agent.infer_posterior(context)

        self.total_env_steps += num_transitions

    def train(self):

        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()
        import pickle
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        for itr in range(500):
            # if self.total_train_steps < 0:
            #     self.collect_data(20, False, add_to_latent_buffer=False)  # 2000 self.num_initial_steps
            # collect some trajectories with z ~ prior
            self.collect_data(10000, True, add_to_latent_buffer=True)
            print(f'iteration {itr}: collect_data from rollout done')
            if itr % 20 == 0:  # 每20局加载一次模型
                # with open('D:/code/ICML2019-TREX-master-revised/mujoco/trajs ' + f'/{self.env_id}' + '/trajs_'+f'{it_//20}' + '.pickle', 'rb') as f:
                with open('../../ generated_demos ' + f'/{self.env}' + '/trajs_' + f'{itr // 20}' + '.pickle',
                          'rb') as f:
                    trajs = pickle.load(f)
                print(f'iteration {itr}: collect_data_from_demos trajs_{itr // 20} done')
            self.collect_data_from_demos(trajs[itr % 20 * 10:(itr % 20 + 1) * 10])
            print('begin trainning...')

            # self.collect_data(4000, True, add_to_latent_buffer=True)
            # self.collect_data(6000, True, add_to_latent_buffer=True)
            # Sample train tasks and compute gradient updates on parameters.
            for train_step in range(self.train_steps):  # self.num_train_steps_per_itr):  # 4000
                self.train_step_new()
                # 从buffer中选取长度为100的transition序列，然后以20为seq_length 分成 80个sequence,作为一个batch更新
                # (80，20，dim) dim=obs_dim + act_dim + 1      e=s,a,r
                # 比如hopper dim=11+3+1=15
                self.total_train_steps += 1
                # End of epoch handling
            self.duration = time.time() - start_time
            print(f'total_env_steps:{self.total_env_steps}',
                  f'total_train_steps:{self.total_train_steps}', f'duration:{self.duration}')
            if (self.total_train_steps + 1) % self.steps_per_epoch == 0:
                epoch = (self.total_train_steps + 1) // self.steps_per_epoch

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    # self.logger.save_state({'env': self.env}, None)
                    model_path = self.model_path + self.env_name + f'_s{self.seed}_{self.collect_data_samples}_{self.train_steps}/'
                    # if not os.path.exists(model_path):
                    os.makedirs(model_path, exist_ok=True)
                    torch.save(self.agent.latent_encoder.state_dict(), model_path + f'latent_encoder_{epoch}.pt')
                    torch.save(self.agent.pi.state_dict(), model_path + f'pi_{epoch}.pt')
                    torch.save(self.q1_net.state_dict(), model_path + f'q1_net_{epoch}.pt')
                    torch.save(self.q2_net.state_dict(), model_path + f'q2_net_{epoch}.pt')
