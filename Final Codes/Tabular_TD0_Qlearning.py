import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gym import Wrapper
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import models , utilities

HYPER_PARAMETERS={
                'env_name': "CliffWalking-v0",
                'stop_reward': -8, 
                'n_step_unrolling':1,                 
                'learning_rate': 0.1,
                'MA_win':10,
                'initial_epsilon':1,
                'min_epsilon':0.01,
                'epsilon_delta':0.01,
                'gamma':.9,
                }

output_metrics=defaultdict(list)
output_metrics_average=defaultdict(list)

###################################################################
#ENV
###################################################################
class My_wrapper(Wrapper):
    def __init__(self, env):
        super(My_wrapper, self).__init__(env)

    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        if next_state in range(37,47,1):
            done=True
        return next_state, reward, done, truncated, info


env=gym.make(HYPER_PARAMETERS['env_name'])
env=My_wrapper(env)

agent=models.TabularAgent(env,HYPER_PARAMETERS['gamma'],HYPER_PARAMETERS['learning_rate'],epsilon=HYPER_PARAMETERS['initial_epsilon'])
eps_obj=utilities.EpsilonDecay(HYPER_PARAMETERS['initial_epsilon'],HYPER_PARAMETERS['min_epsilon'],HYPER_PARAMETERS['epsilon_delta'])
agent.Known_State_action_table_generator()

best_reward=-1000
episode_reward=[]
epsidoe_done_tracker=[]
state, _ = env.reset()

training_counters=defaultdict(int)
while best_reward < HYPER_PARAMETERS['stop_reward']:

    agent.epsilon=eps_obj.update_epsilon()
    output_metrics['epsilon'].append(agent.epsilon)
    state,action,reward,next_state,terminated , truncated = agent.agent_step_TD0_Qlearning_greedy(state)
    episode_reward.append(reward)
    epsidoe_done_tracker.append(terminated or truncated)

    if epsidoe_done_tracker[-1]:
       
        _,Episodes_reward_tracker,Episodes_Steps_Tracker= utilities.rewards_to_go_calc(episode_reward,epsidoe_done_tracker,HYPER_PARAMETERS['gamma'])
        output_metrics['Episode_reward'].append(Episodes_reward_tracker[0])
        output_metrics['Episode_Steps'].append(Episodes_Steps_Tracker[0])
        training_counters['Number_Episodes']+=1
        output_metrics['epsilon'].append(agent.epsilon)
        state,_=env.reset()
        episode_reward.clear()
        epsidoe_done_tracker.clear()
        output_metrics_average=utilities.metric_average_calc(output_metrics,output_metrics_average,alpha=0.99,MA_length=HYPER_PARAMETERS['MA_win'])
        agent.epsilon=eps_obj.update_epsilon()
        output_metrics['epsilon'].append(agent.epsilon)
    else:
        state=next_state
    
    if len(output_metrics_average['MA_Episode_reward'])>HYPER_PARAMETERS['MA_win']:
        if output_metrics_average['MA_Episode_reward'][-1]> best_reward :
            best_reward=output_metrics_average['MA_Episode_reward'][-1]

    if best_reward > HYPER_PARAMETERS['stop_reward']:
        break


    if training_counters['Number_Episodes'] % 5000 ==0:
        utilities.print_metrics(output_metrics)
        utilities.print_metrics(output_metrics_average)

    

env = gym.make(HYPER_PARAMETERS['env_name'],render_mode="human")  


agent.epsilon=0.01
for _ in range(5):
    done = False
    state, _ = env.reset()
    while not done:
        env.render() 
        action=agent.eps_greedy_action_selector(state)
        state, reward, termin, trunc ,_= env.step(action)
        done = termin or trunc

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
        


    
