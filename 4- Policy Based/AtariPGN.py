#!/usr/bin/env python3
import gymnasium as gym
import ptan
import numpy as np
import argparse
from tensorboardX import SummaryWriter
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim

import common

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50

REWARD_STEPS = 4
CLIP_GRAD = 0.1


class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()

        self.common_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.actor_net = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.critic_net = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.common_net(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.common_net(fx).view(fx.size()[0], -1)
        return self.actor_net(conv_out), self.critic_net(conv_out)


def unpack_batch(batch, net, device='cpu'):

    batch_states_list = []                                                                                     # a list to store all first steps 
    batch_actions_list = []
    batch_rewards_list = []
    not_done_idx = []
    batch_last_states_list = []
    for idx, exp in enumerate(batch):
        batch_states_list.append(np.array(exp.state, copy=False))
        batch_actions_list.append(int(exp.action))
        batch_rewards_list.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            batch_last_states_list.append(np.array(exp.last_state, copy=False))

    batch_satets_tensorized = torch.FloatTensor(np.array(batch_states_list, copy=False)).to(device)
    batch_actions_tensorized = torch.LongTensor(batch_actions_list).to(device)
    batch_rewards_np = np.array(batch_rewards_list, dtype=np.float32)

    if not_done_idx:
        batch_last_states_tensorized = torch.FloatTensor(np.array(batch_last_states_list, copy=False)).to(device)
        
        v_s_prime = net(batch_last_states_tensorized)[1]
        v_s_prime_np = v_s_prime.data.cpu().numpy()[:, 0]
        v_s_prime_np *= GAMMA ** REWARD_STEPS
        batch_rewards_np[not_done_idx] += v_s_prime_np

    v_s_bellman = torch.FloatTensor(batch_rewards_np).to(device)

    return batch_satets_tensorized, batch_actions_tensorized, v_s_bellman

def make_env() -> gym.Env:
    env=gym.make("PongNoFrameskip-v4")
    env=ptan.common.wrappers.wrap_dqn(env)
    return env

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    envs = [make_env() for _ in range(NUM_ENVS)]
    writer = SummaryWriter(comment=f"-pong-a2c_{current_time}")
    net = AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print(net)

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    batch = []

    with common.RewardTracker(writer, stop_reward=18) as tracker:                                  # start of Training loop , will continue loop untill mean reward of 100 most recent calls or RewardTracker.reward() method is greater than stop_reward
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:                 # use to track metrics during training loop every 10 batch
            for step_idx, exp in enumerate(exp_source):                                            # at each iteration it will return ExpeirienceFirstLast object of one env
                batch.append(exp)

                
                new_rewards = exp_source.pop_total_rewards()                                       # this method has output only at the end of episode
                if new_rewards:                                                                    # means if episode ends
                    if tracker.reward(new_rewards[0], step_idx):                                   # output of reward() method is True if and nly if mean reward > stop_reward
                        break                                                                      # it will break the training loop

                if len(batch) < BATCH_SIZE:                                                        # optimizer will not work untill number of elements in batch is greater than BATCH_SIZE
                    continue

                batch_states_tensor, batch_actions_tensor, v_s_bellman = unpack_batch(batch, net, device=device)
                batch.clear()

                optimizer.zero_grad()
                policy_net_output_pi_s_a_logits, critic_net_v_s = net(batch_states_tensor)

                value_loss = F.mse_loss(critic_net_v_s.squeeze(-1), v_s_bellman)
                adv_v = v_s_bellman - critic_net_v_s.squeeze(-1).detach()

                pi_s_a_matrix_logsm = F.log_softmax(policy_net_output_pi_s_a_logits, dim=1)
                pi_s_a_taken_action=pi_s_a_matrix_logsm[range(BATCH_SIZE), batch_actions_tensor]   
                policy_loss = -torch.dot(adv_v, pi_s_a_taken_action) / len(adv_v)

                pi_s_a_matrix_sm = F.softmax(policy_net_output_pi_s_a_logits, dim=1)
                
                entropy_loss = ENTROPY_BETA * (pi_s_a_matrix_sm * pi_s_a_matrix_logsm).sum(dim=1).mean()

                # calculate policy gradients only
                policy_loss.backward(retain_graph=True)
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in net.parameters()
                                        if p.grad is not None])

                # apply entropy and value gradients
                loss_v = entropy_loss + value_loss
                loss_v.backward()
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                # get full loss
                total_loss =loss_v + policy_loss

                tb_tracker.track("advantage",       adv_v, step_idx)
                tb_tracker.track("values",          critic_net_v_s, step_idx)
                tb_tracker.track("batch_rewards",   v_s_bellman, step_idx)
                tb_tracker.track("loss_entropy",    entropy_loss, step_idx)
                tb_tracker.track("policy_loss",     policy_loss, step_idx)
                tb_tracker.track("loss_value",      value_loss, step_idx)
                tb_tracker.track("loss_total",      total_loss, step_idx)
                tb_tracker.track("grad_l2",         np.sqrt(np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max",        np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var",        np.var(grads), step_idx)
