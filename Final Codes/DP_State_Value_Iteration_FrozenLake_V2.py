import gymnasium as gym
import numpy
import torch
from collections import defaultdict , Counter
import matplotlib.pyplot as plt
import math
import numpy as np

HYPER_PARAMETERS={
                'env_name': "FrozenLake-v1",
                'stop_reward': .8,
                'Number_of_Episode': 20,            
                'learning_rate': 0.001,
                'gamma': 0.9,
                }

output_metrics=defaultdict(list)
output_metrics_average=defaultdict(list)

class StateValueIteration:
    def __init__(self,env):
        self.env = env
        self.state, _ = self.env.reset()                                 
        self.rewards = defaultdict(float)                    
        self.transits = defaultdict(Counter)     
        self.state_values = defaultdict(float)        

    def play_n_random_steps(self, count):                                
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done,is_trunc, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1

            if is_done or is_trunc:
                self.state , _ = self.env.reset()
                
            else:
                self.state=new_state

    def calc_state_value_base_part(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        state_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            val = reward + HYPER_PARAMETERS['gamma'] * self.state_values[tgt_state]
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

    def play_one_episode(self):
        total_reward = 0.0
        state, _ = self.env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done,is_truncated, _ = self.env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done or is_truncated:
                break
            state = new_state
        return total_reward

    def state_value_iteration(self):                                                         
        for state in range(self.env.observation_space.n):
            state_values = []
            for action in range(self.env.action_space.n):
                state_values.append(self.calc_state_value_base_part(state, action))          
            self.state_values[state] = max(state_values)                                     


def print_metrics(metrics):
    print(" | ".join(f"{key}: {values[-1]:.2f}" for key, values in metrics.items() if values))


def metric_average_calc(metrics,metrics_average,alpha=0.99,MA_length=100):
    for key, values in metrics.items():
        if len(values)>MA_length:
            metrics_average['MA_'+key].append(sum(values[-1*MA_length +1:])/MA_length)
        else:
            metrics_average['MA_'+key].append(0)
        if len(values)==1:
            metrics_average['RunningAverage_'+key].append(values[0])
        else:
            metrics_average['RunningAverage_'+key].append(alpha*values[-2]+(1-alpha)*values[-1])
    return metrics_average




env=gym.make(HYPER_PARAMETERS['env_name'])
agent = StateValueIteration(env)

iter_no = 0
best_reward = 0.0
while True:
    iter_no += 1
    agent.play_n_random_steps(1000)
    agent.state_value_iteration()
    
    for _ in range(HYPER_PARAMETERS['Number_of_Episode']):
        output_metrics['Episode_Reward'].append(agent.play_one_episode())


    if output_metrics['Episode_Reward'][-1] > best_reward:
        best_reward = output_metrics['Episode_Reward'][-1]
        print("Best reward updated %.3f" % (best_reward))
    
    output_metrics_average=metric_average_calc(output_metrics,output_metrics_average,alpha=0.99,MA_length=100)

    if output_metrics_average['MA_Episode_Reward'][-1]>HYPER_PARAMETERS['stop_reward']:
        break

    if iter_no % 50 ==0:
        print_metrics(output_metrics)
        print_metrics(output_metrics_average)

env = gym.make(HYPER_PARAMETERS['env_name'],render_mode="human")  
agent.env=env
state, _ = env.reset()
done = False
while not done:
    env.render() 

    action=agent.select_action(state)
    state, reward, done, _ ,_= env.step(action)

env.close()  


def plot_dicts(dict1):

    all_keys = sorted(set(dict1.keys()))  # Unique keys
    num_keys = len(all_keys)
    n = math.ceil(num_keys / 2)  # Calculate rows for 2 columns
    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))  # Create subplots
    axes = np.array(axes).reshape(-1)  # Flatten for easy indexing
    for i, key in enumerate(all_keys):
        ax = axes[i]
        ax.plot(dict1[key], label="Dict 1", marker="o")
        ax.set_title(f"Key: {key}")
        ax.legend()
        ax.grid(True)

    # Hide unused subplots if odd number of keys
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

plot_dicts(output_metrics)
plot_dicts(output_metrics_average)