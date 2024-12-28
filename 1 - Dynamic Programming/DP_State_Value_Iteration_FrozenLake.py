
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v1"
#ENV_NAME = "FrozenLake8x8-v0"      # uncomment for larger version
GAMMA = 0.9
num_of_test_episode = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state, _ = self.env.reset()                                 # return initial state of env
        self.rewards = collections.defaultdict(float)                    # reward table , a default dictionary which expect to rceive float data as value part
        self.transits = collections.defaultdict(collections.Counter)     # transition probability distribution , a default dictionary which expect collection.counters as value part
        self.state_values = collections.defaultdict(float)               # state value table , a default dictionary which expect to rceive float data as value part

    def play_n_random_steps(self, count):                                # play n=count steps , to initiate reward table , transit table and also initial estimation of state value part
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done,is_trunc, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1

            if is_done :
                self.state , _ = self.env.reset()
                
            else:
                self.state=new_state

    def calc_state_value_base_part(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        state_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            val = reward + GAMMA * self.state_values[tgt_state]
            state_value += (count / total) * val
        return state_value

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            state_value_base_part = self.calc_state_value_base_part(state, action)
            if best_value is None or best_value < state_value_base_part:
                best_value = state_value_base_part
                best_action = action
        return best_action

    def play_one_episode(self, env):
        total_reward = 0.0
        state, _ = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done,is_truncated, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def state_value_iteration(self):                                                         #calculate and update state values for entire states
        for state in range(self.env.observation_space.n):
            state_values = []
            for action in range(self.env.action_space.n):
                state_values.append(self.calc_state_value_base_part(state, action))          #calculate the state value base part for specific state and all posible actions
            self.state_values[state] = max(state_values)                                     # in order to implement bellman optimality equation we need to select action with maximum base part


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.state_value_iteration()

        reward = 0.0
        for _ in range(num_of_test_episode):
            reward += agent.play_one_episode(test_env)
        reward /= num_of_test_episode
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (
                best_reward, reward))
            best_reward = reward
        if reward > 0.95:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
    print(sorted(agent.state_values.items()))
