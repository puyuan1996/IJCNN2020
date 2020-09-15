import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import gym
from tqdm import tqdm
import matplotlib
import pickle

matplotlib.use('agg')

# Import my own libraries
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/learner/baselines/')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show ERROR log
from tf_commons.ops import *


class PPO2Agent(object):
    def __init__(self, env, env_type, path, stochastic=False, gpu=True):
        from baselines.common.policies import build_policy
        from baselines.ppo2.model import Model
        # from learner.baselines.baselines.ppo2.model import Model
        self.graph = tf.Graph()

        if gpu:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
        else:
            config = tf.ConfigProto(device_count={'GPU': 0})

        self.sess = tf.Session(graph=self.graph, config=config)

        with self.graph.as_default():
            with self.sess.as_default():
                ob_space = env.observation_space
                ac_space = env.action_space

                if env_type == 'atari':
                    policy = build_policy(env, 'cnn')
                elif env_type == 'mujoco':
                    policy = build_policy(env, 'mlp')
                else:
                    assert False, ' not supported env_type'

                make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1,
                                           nbatch_train=1,
                                           nsteps=1, ent_coef=0., vf_coef=0.,
                                           max_grad_norm=0.)
                self.model = make_model()

                self.model_path = path
                self.model.load(path)

        if env_type == 'mujoco':
            with open(path + '.env_stat.pkl', 'rb') as f:
                import pickle
                s = pickle.load(f)
            self.ob_rms = s['ob_rms']
            self.ret_rms = s['ret_rms']
            self.clipob = 10.
            self.epsilon = 1e-8
        else:
            self.ob_rms = None

        self.stochastic = stochastic

    def act(self, obs, reward, done):
        if self.ob_rms:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)

        with self.graph.as_default():
            with self.sess.as_default():
                if self.stochastic:
                    a, v, state, neglogp = self.model.step(obs)
                else:
                    a = self.model.act_model.act(obs)
        return a

class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space
        self.model_path = 'random_agent'

    def act(self, observation, reward, done):
        return self.action_space.sample()[None]

class GTDataset(object):
    def __init__(self, env, env_id):
        self.env = env
        self.env_id = env_id
        self.unwrapped = env
        while hasattr(self.unwrapped, 'env'):
            self.unwrapped = self.unwrapped.env

    def gen_traj(self, agent, min_length):
        max_x_pos = -99999
        obs_init = self.env.reset()
        obs, actions, rewards, next_obs, terms = [], [], [], [], []
        ob = obs_init
        while True:
            action = agent.act(ob, None, None)
            next_ob, reward, done, _ = self.env.step(action)
            if self.unwrapped.sim.data.qpos[0] > max_x_pos:
                max_x_pos = self.unwrapped.sim.data.qpos[0]

            obs.append(ob)
            actions.append(action)
            rewards.append(reward)
            next_obs.append(next_ob)
            terms.append(done)
            ob = next_ob

            if done:
                if len(obs) < min_length:
                    obs.pop()
                    obs.append(self.env.reset())
                else:
                    # obs.pop()
                    break

        return [np.stack(obs, axis=0).astype(np.float32), np.concatenate(actions, axis=0),
                np.array(rewards).astype(np.float32), np.stack(next_obs, axis=0).astype(np.float32),
                np.array(terms)], max_x_pos

    def prebuilt(self, agents, min_length):
        assert len(agents) > 0, 'no agent given'
        trajs = []
        for agent in tqdm(agents):
            traj, max_x_pos = self.gen_traj(agent, min_length)

            trajs.append(traj)
            tqdm.write('model: %s avg reward: %f max_x_pos: %f' % (agent.model_path, np.sum(traj[2]), max_x_pos))
        obs, actions, rewards = zip(*trajs)
        self.trajs = (np.concatenate(obs, axis=0), np.concatenate(actions, axis=0), np.concatenate(rewards, axis=0))

        print(self.trajs[0].shape, self.trajs[1].shape, self.trajs[2].shape)

    def sample(self, num_samples, steps=40, include_action=False):
        obs, actions, rewards = self.trajs

        D = []
        for _ in range(num_samples):
            x_ptr = np.random.randint(len(obs) - steps)
            y_ptr = np.random.randint(len(obs) - steps)

            if include_action:
                D.append((np.concatenate((obs[x_ptr:x_ptr + steps], actions[x_ptr:x_ptr + steps]), axis=1),
                          np.concatenate((obs[y_ptr:y_ptr + steps], actions[y_ptr:y_ptr + steps]), axis=1),
                          0 if np.sum(rewards[x_ptr:x_ptr + steps]) > np.sum(rewards[y_ptr:y_ptr + steps]) else 1)
                         )
            else:
                D.append((obs[x_ptr:x_ptr + steps],
                          obs[y_ptr:y_ptr + steps],
                          0 if np.sum(rewards[x_ptr:x_ptr + steps]) > np.sum(rewards[y_ptr:y_ptr + steps]) else 1)
                         )

        return D


class GTTrajLevelNoStepsDataset(GTDataset):
    def __init__(self, env, env_id, max_steps):
        super().__init__(env, env_id)
        self.max_steps = max_steps
        self.env_id = env_id
        # self.max_steps = 100

    def prebuilt(self, agents, min_length):
        assert len(agents) > 0, 'no agent is given'

        trajs = []
        for agent_idx, agent in enumerate(tqdm(agents)):
            agent_trajs = []
            for i in range(200):  # 每个stage的agent采样200局trajs 一共24+1个agent 5e6=25*1000*200
                [obs, actions, rewards, next_obs, terms], _ = self.gen_traj(agent, -1)  # 无论traj多长都保存下来
                agent_trajs.append([obs, actions, rewards, next_obs, terms])

            with open('./trajs' + f'/{self.env_id}' + '/trajs_' + f'{agent_idx}' + '.pickle', 'wb') as f:
                pickle.dump(agent_trajs, f)

            trajs.append(agent_trajs)

        agent_rewards = [np.mean([np.sum(rewards) for _, _, rewards, _, _ in agent_trajs]) for agent_trajs in trajs]
        # print(agent_rewards)
        with open('./trajs' + f'/{self.env_id}' + '/agent_rewards' + '.pickle', 'wb') as f:
            pickle.dump(agent_rewards, f)

    def sample(self, num_samples, steps=None, include_action=False):
        assert steps == None
        D = []
        D_q = []
        GT_preference = []
        for _ in tqdm(range(num_samples)):
            x_idx = np.random.choice(len(self.trajs), 1, replace=False)  # 随机选1个agent
            x_idx = int(x_idx)
            x_traj = self.trajs[x_idx][np.random.choice(len(self.trajs[x_idx]))]  # 从选定的agent中随机选一个轨迹

            if len(x_traj[0]) > self.max_steps:  # obs的长度，即有多少步 # self.max_steps=40
                ptr = np.random.randint(len(x_traj[0]) - self.max_steps)
                x_slice = slice(ptr, ptr + self.max_steps)
                len_segm = self.max_steps
            else:
                print("轨迹步数小于40，重新采样")
                continue

            if include_action:
                D.append((np.concatenate((x_traj[0][x_slice], x_traj[1][x_slice]), axis=1),  # obs actions 即s_a
                          np.concatenate((x_traj[0][x_slice], np.repeat(self.x, len_segm, axis=0)), axis=1),
                          # obs x 即s_x !!!
                          np.sum(x_traj[2][x_slice])  # rewards的和
                          ))

        return D


def generate_trajs(args):
    logdir = Path(args.log_dir)

    if logdir.exists():
        import shutil
        shutil.rmtree(str(logdir))
    logdir.mkdir(parents=True)
    with open(str(logdir / 'args.txt'), 'w') as f:
        f.write(str(args))

    logdir = str(logdir)
    env = gym.make(args.env_id)

    train_agents = [RandomAgent(env.action_space)] if args.random_agent else []
    train_agents_all = []
    for i in range(25):
        train_agents = []
        if i == 0:
            models = sorted([p for p in Path(args.learners_path).glob('?????') if int(p.name) == 1])
        else:
            models = sorted([p for p in Path(args.learners_path).glob('?????') if int(p.name) / 20 == i])

        for path in models:
            agent = PPO2Agent(env, args.env_type, str(path), stochastic=args.stochastic)
            train_agents.append(agent)
        train_agents_all.extend(train_agents)

    dataset = GTTrajLevelNoStepsDataset(env, args.env_id, args.max_steps)
    dataset.prebuilt(train_agents_all, args.min_length)


if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='', help='Select the environment to run')
    parser.add_argument('--env_type', default='', help='mujoco or atari')
    parser.add_argument('--learners_path', default='', help='path of learning agents')
    parser.add_argument('--max_chkpt', default=240, type=int, help='decide upto what learner stage you want to give')
    parser.add_argument('--steps', default=None, type=int, help='length of snippets')
    parser.add_argument('--max_steps', default=None, type=int, help='length of max snippets (gt_traj_no_steps only)')
    parser.add_argument('--traj_noise', default=None, type=int,
                        help='number of adjacent swaps (gt_traj_no_steps_noise only)')
    parser.add_argument('--min_length', default=1000, type=int,
                        help='minimum length of trajectory generated by each agent')
    parser.add_argument('--num_layers', default=2, type=int, help='number layers of the reward network')
    parser.add_argument('--embedding_dims', default=256, type=int, help='embedding dims')
    parser.add_argument('--num_models', default=3, type=int, help='number of models to ensemble')
    parser.add_argument('--l2_reg', default=0.01, type=float, help='l2 regularization size')
    parser.add_argument('--noise', default=0.1, type=float, help='noise level to add on training label')
    parser.add_argument('--num_samples', default=1000, type=int, help='|D| in the preference paper')
    parser.add_argument('--iters', default=10000, type=int)
    parser.add_argument('--N', default=10, type=int, help='number of trajactory mix (gt_traj_no_steps_n_mix only)')
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--preference_type',
                        help='gt or gt_traj or time or gt_traj_no_steps, gt_traj_no_steps_n_mix; if gt then '
                             'preference will be given as a GT reward, otherwise, it is given as a time index')
    parser.add_argument('--min_margin', default=1, type=int,
                        help='when prefernce type is "time", the minimum margin that we can assure there exist a margin')
    parser.add_argument('--include_action', action='store_true', help='whether to include action for the model or not')
    parser.add_argument('--stochastic', action='store_true', help='whether want to use stochastic agent or not')
    parser.add_argument('--random_agent', action='store_true', help='whether to use default random agent')
    parser.add_argument('--eval', action='store_true', help='path to log base (env_id will be concatenated at the end)')
    # Args for PPO
    parser.add_argument('--rl_runs', default=1, type=int)
    parser.add_argument('--ppo_log_path', default='ppo2')
    parser.add_argument('--custom_reward', required=True, help='preference or preference_normalized')
    parser.add_argument('--ctrl_coeff', default=0.0, type=float)
    parser.add_argument('--alive_bonus', default=0.0, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--batch_size_r', default=1, type=int)
    parser.add_argument('--batch_size_q', default=50, type=int)
    # args = parser.parse_args()

    parsers = "--env_id HalfCheetah-v2 --env_type mujoco --learners_path ./learner/demo_models/halfcheetah/checkpoints " \
              "--max_chkpt 480 --num_models 1 --max_steps 40 --noise 0.0 --traj_noise 0 --num_layers 2 --log_dir " \
              "./log/hopper/max60/gt_traj_no_steps/ --preference_type gt_traj_no_steps --custom_reward " \
              "preference_normalized --ppo_log_path preference_norm_ctrl_coeff_011 --ctrl_coeff 0.1 --num_samples 10000 --iters 10000 " \
              "--include_action --stochastic --rl_runs 1".split()
    args = parser.parse_args(parsers)
    env_id_dict={0:'Ant-v2',1:'HalfCheetah-v2',2:'Hopper-v2',3:'Humanoid-v2',4:'Pusher-v2',5:'Reacher-v2',6:'Striker-v2',7:'Swimmer-v2',8:'Thrower-v2'}#,9:'Walker-v2'}
    learner_path_dict = {0: 'ant', 1: 'halfcheetah', 2: 'hopper', 3: 'humanoid', 4: 'pusher', 5: 'reacher',6: 'striker', 7: 'swimmer', 8: 'thrower'}  # ,9:'Walker-v2'}
    for i in range(1):
        args.env_id=env_id_dict[i]
        args.learners_path='./learner/demo_models/'+learner_path_dict[i]+'/checkpoints'
        generate_trajs(args) # 模型加载保存的checkpoints，并rollout 轨迹保存下来 不render
        print(f'{args.env_id} Done!')
    print('All Done!')
