import gym
import ptan
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


HIDDEN_SIZE = 128
BATCH_SIZE = 16
TGT_NET_SYNC = 10
GAMMA = 0.9
REPLAY_BUFFER_SIZE = 1000
LR = 1e-3
EPS_DECAY=0.99


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x.float())


@torch.no_grad()
def unpack_batch(batch, net, gamma):
    batch_state = []                                 # a list of states , dim=0 is batch_size ,rest of dims , state representation
    batch_action = []
    batch_reward = []
    batch_is_done = []
    batch_next_state = []
    for exp in batch:
        batch_state.append(exp.state)
        batch_action.append(exp.action)
        batch_reward.append(exp.reward)
        batch_is_done.append(exp.last_state is None)
        if exp.last_state is None:
            batch_next_state.append(exp.state)
        else:
            batch_next_state.append(exp.last_state)

    batch_state_tensor = torch.tensor(batch_state)
    batch_action_tensor = torch.tensor(batch_action)
    batch_reward_tensor = torch.tensor(batch_reward)
    batch_next_state_tensor = torch.tensor(batch_next_state)
    q_s_prime_a_matrix = net(batch_next_state_tensor)
    q_sprime_aprime = torch.max(q_s_prime_a_matrix, dim=1)[0]
    q_sprime_aprime[batch_is_done] = 0.0
    tgt_net_q_s_a=q_sprime_aprime * gamma + batch_reward_tensor
    return batch_state_tensor, batch_action_tensor, tgt_net_q_s_a


if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    base_Q_net = Net(env.observation_space.shape[0], HIDDEN_SIZE, env.action_space.n)
    tgt_net = ptan.agent.TargetNet(base_Q_net)
    action_selector = ptan.actions.EpsilonGreedyActionSelector(
        epsilon=1, selector=ptan.actions.ArgmaxActionSelector())
    agent = ptan.agent.DQNAgent(base_Q_net, action_selector)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=GAMMA)                                                         #step_count is by default =1
    replay_buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=REPLAY_BUFFER_SIZE)
    optimizer = optim.Adam(base_Q_net.parameters(), LR)

    step = 0
    episode = 0
    solved = False

    while True:
        step += 1
        replay_buffer.populate(1)

        for reward, steps in exp_source.pop_rewards_steps():                           #if episode is not finished yet , output of pop_rewards_steps is empty 
            episode += 1
            print("%d: episode %d done, reward=%.3f, epsilon=%.2f" % (
                step, episode, reward, action_selector.epsilon))
            solved = reward > 150
        if solved:
            print("Congrats!")
            break

        if len(replay_buffer) < 2*BATCH_SIZE:
            continue

        batch = replay_buffer.sample(BATCH_SIZE)

        batch_state_tensor, batch_action_tensor, tgt_net_q_s_a = unpack_batch(
            batch, tgt_net.target_model, GAMMA)
        optimizer.zero_grad()
        predicted_q_s_a = base_Q_net(batch_state_tensor)
        predicted_q_s_a = predicted_q_s_a.gather(1, batch_action_tensor.unsqueeze(-1)).squeeze(-1)

        loss_v = F.mse_loss(predicted_q_s_a, tgt_net_q_s_a)
        loss_v.backward()
        optimizer.step()
        action_selector.epsilon *= EPS_DECAY

        if step % TGT_NET_SYNC == 0:
            tgt_net.sync()
