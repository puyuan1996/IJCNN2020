import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

import pytorch_util as ptu


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class Agent(nn.Module):

    def __init__(self,
                 latent_dim,
                 hidden_dim,
                 latent_encoder,
                 policy,
                 args
                 ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.latent_encoder = latent_encoder
        self.pi = policy
        self.device = args.device

        self.recurrent = args.recurrent
        self.use_ib = True
        self.use_next_obs_in_context = args.use_next_obs_in_context

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.clear_z()

    def clear_z(self):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(1, self.latent_dim)
        if self.use_ib:
            log_var = ptu.ones(1, self.latent_dim)
        else:
            log_var = ptu.zeros(1, self.latent_dim)
        self.z_mu = mu
        self.z_logvar = log_var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        # self.latent_encoder.reset(1)

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        # if self.recurrent:
        #     self.latent_encoder.hidden = self.latent_encoder.hidden.detach()

    def update_context(self, inputs):
        ''' append single transition to the current context '''

        o, a, r, no, d = [torch.as_tensor(v, dtype=torch.float32) for v in inputs]

        r.unsqueeze_(0)

        # o = ptu.from_numpy(o[None, None, ...])
        # a = ptu.from_numpy(a[None, None, ...])
        # r = ptu.from_numpy(np.array([r])[None, None, ...])
        # no = ptu.from_numpy(no[None, None, ...])

        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=-1)
            data.unsqueeze_(0)

        else:
            # data = torch.cat([o, a, r], dim=2)
            data = torch.cat([o, a, r], dim=-1)
            data.unsqueeze_(0)

        self.context = data

        # if self.context is None:
        #     self.context = data
        # else:
        #     self.context = torch.cat([self.context, data], dim=0)

    def compute_kl_div(self):
        ''' compute KL( p(z|c) || q(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.exp(log_var)) for mu, log_var in
                      zip(torch.unbind(self.z_mu), torch.unbind(self.z_logvar))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        # kl_div_sum = -0.5 * torch.sum(1 + self.z_logvar - self.mu.pow(2) - self.z_logvar.exp())
        return kl_div_sum


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample_z(self):
        # if self.use_ib:
        #         #     posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in
        #         #                   zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        #         #     z = [d.rsample() for d in posteriors]
        #         #     self.z = torch.stack(z)
        #         # else:
        #         #     self.z = self.z_means
        self.z = self.reparameterize(self.z_mu, self.z_logvar)

    def get_action(self, hidden_in, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        # hidden_in = (torch.zeros([1, 1, self.hidden_dim], dtype=torch.float).to(self.device),
        #              torch.zeros([1, 1, self.hidden_dim], dtype=torch.float).to(self.device))
        params, hidden_out = self.latent_encoder(self.context.unsqueeze_(0), hidden_in)
        """recurrent latent encoder"""
        params = params.view(-1, self.latent_encoder.output_size)
        self.z_mu = params[..., :self.latent_dim]
        self.z_logvar = F.softplus(params[..., self.latent_dim:])
        self.sample_z()
        obs = ptu.from_numpy(obs[None])
        o_z = torch.cat([obs, self.z], dim=1)
        return self.pi.get_action_from_cat_o_z(o_z, deterministic=deterministic), hidden_out

    def set_num_steps_total(self, n):
        self.pi.set_num_steps_total(n)

    def infer_latent(self, context_seq_batch):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        hidden_in = (torch.zeros([1, context_seq_batch.shape[0], self.hidden_dim], dtype=torch.float).to(self.device),
                     torch.zeros([1, context_seq_batch.shape[0], self.hidden_dim], dtype=torch.float).to(self.device))
        params, hidden_out = self.latent_encoder(context_seq_batch, hidden_in)
        """recurrent latent encoder"""
        params = params.view(-1, self.latent_encoder.output_size)
        self.z_mu = params[..., :self.latent_dim]
        self.z_logvar = F.softplus(params[..., self.latent_dim:])

        # self.z_means = torch.stack([p for p in mu])
        # self.z_vars = torch.stack([p for p in sigma_squared])
        """feedforward latent encoder"""
        # params = self.latent_encoder(context)
        # params = params.view(context.size(0), -1, self.latent_encoder.output_size)
        # # with probabilistic z, predict mean and variance of q(z | c)
        # if self.use_ib:
        #     mu = params[..., :self.latent_dim]
        #     sigma_squared = F.softplus(params[..., self.latent_dim:])
        #     z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
        #     self.z_means = torch.stack([p[0] for p in z_params])
        #     self.z_vars = torch.stack([p[1] for p in z_params])
        # # sum rather than product of gaussians structure
        # else:
        #     self.z_means = torch.mean(params, dim=1)
        # self.sample_z()

    def forward(self, obs_batch, context_seq_batch):
        ''' given context, get statistics under the current policy of a set of observations '''
        self.infer_latent(context_seq_batch)
        self.sample_z()

        if obs_batch.dim() == 2:
            obs_batch.unsqueeze_(0)

        b, s, _ = obs_batch.size()
        obs = obs_batch.view(b * s, -1)
        # z = [z.repeat(b, 1) for z in z]
        # z = torch.cat(z, dim=0)

        # run policy, get log probs and new actions
        o_z = torch.cat([obs, self.z.detach()], dim=1)
        # policy_outputs = self.pi(in_, reparameterize=True, return_log_prob=True)
        policy_outputs = self.pi(o_z, deterministic=False, with_logprob=True)

        return policy_outputs, self.z

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mu = np.mean(np.abs(ptu.get_numpy(self.z_mu[0])))
        z_sig = np.mean(ptu.get_numpy(self.z_log_var[0]))
        eval_statistics['Z mean eval'] = z_mu
        eval_statistics['Z variance eval'] = z_sig

    @property
    def networks(self):
        return [self.latent_encoder, self.pi]
