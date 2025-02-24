import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import ptan
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import math

HYPER_PARAMETERS={
                'env_name': "CartPole-v1",
                'stop_reward': 350,
                'run_name': 'CartPole',
                'Number_of_Episode': 4,                  # we need compelete episodes to calculate q(s,a) for all state/actions in trajectory using recurssive formula
                'learning_rate': 0.001,
                'gamma': 0.9,
                'ENTROPY_BETA': 0.01,
                'SEED':124
                }

####################################################################set the seed
def set_seed(seed: int = 42):
    random.seed(seed)  # Python's built-in random
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Avoids adaptive optimizations

set_seed(42)

device='cuda' if torch.cuda.is_available() else 'cpu'
output_metrics=defaultdict(list)
output_metrics_average=defaultdict(list)

###################################################################
#ENV
###################################################################
env=gym.make(HYPER_PARAMETERS['env_name'])
###################################################################
# Policy  Gradient Net
###################################################################
class PGN(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(PGN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
    def forward(self, x):
        return self.net(x)

hiden_size=128
net=PGN(env.observation_space.shape[0],hiden_size,env.action_space.n)
###################################################################
# Agent 
###################################################################
class PGAgent(ptan.agent.BaseAgent):
    def __init__(self, model, device="cpu", apply_softmax=True):
        self.model = model
        self.device = device
        self.apply_softmax = apply_softmax

    def __call__(self, states, agent_states=None):
        states_v = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        if np.isnan(self.model(states_v).cpu().data).any():
                print("NaN detected!")
        logits_v = self.model(states_v)
        if self.apply_softmax:
            probs_v = torch.nn.functional.softmax(logits_v, dim=1)
        else:
            probs_v = logits_v  # Assume model outputs probabilities directly
        probs = probs_v.cpu().data.numpy()
        actions = [np.random.choice(len(p), p=p) for p in probs]      #probs batch_size*act_size , p in probs mean a row in probs        
        return actions, agent_states

agent=PGAgent(net,device=device, apply_softmax=True)
exp_source=ptan.experience.ExperienceSource(env,agent,steps_count=1)


###################################################################
# utility functions
###################################################################
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

def rewards_to_go_calc(immediate_reward_list,is_done_trunc,gamma):

    T=len(immediate_reward_list)
    Episode_reward_tracker=[]
    Episode_Steps_Tracker=[]
    rewards_to_go=[0]*T
    discounted_reward=0
    N_Steps=0
    for t in range(T-1,-1,-1):
        if (is_done_trunc[t] and t!=T-1) or t==0:
            Episode_reward_tracker.append(discounted_reward)
            Episode_Steps_Tracker.append(N_Steps+1)
            N_Steps=0
            discounted_reward=0
            discounted_reward=immediate_reward_list[t] + gamma * discounted_reward
            rewards_to_go[t]=discounted_reward
        else:
            N_Steps+=1
            discounted_reward=immediate_reward_list[t] + gamma * discounted_reward
            rewards_to_go[t]=discounted_reward
    return rewards_to_go,Episode_reward_tracker,Episode_Steps_Tracker

def unpack_function_expsource_to_nparray(batch): 

    states, actions, rewards, dones = [],[],[],[]
    for exp in batch:
        state = np.array(exp[0].state)
        states.append(state)
        actions.append(exp[0].action)
        rewards.append(exp[0].reward)
        dones.append(exp[0].done_trunc)
        
    return np.array(states, copy=False), np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8)

###################################################################
# Training Loop
###################################################################

optimizer=optim.Adam(net.parameters(),HYPER_PARAMETERS['learning_rate'])


training_flag=True
best_reward=0
training_counters=defaultdict(int)
while training_flag:

    batch = []
    episode_count=0
    Episodes_rewards=[]
    for exp in exp_source:

        batch.append(exp)

        if exp[0].done_trunc:
            episode_count+=1

        if episode_count < HYPER_PARAMETERS['Number_of_Episode']:
            continue

        state_np,action_np,reward_np,done_trunc_np=unpack_function_expsource_to_nparray(batch) 

        q_s_a_list,Episodes_rewards,Episode_steps=rewards_to_go_calc(reward_np,done_trunc_np,HYPER_PARAMETERS['gamma'])
        baseline=np.mean(Episodes_rewards)
        ending_criteria=np.mean(Episode_steps)

        output_metrics['steps_per_episode'].append(ending_criteria)

        if ending_criteria > best_reward:
            best_reward=ending_criteria
            torch.save(net.state_dict(),'xxxxx.pth')

        if best_reward > HYPER_PARAMETERS['stop_reward']:
            training_flag=False
            break

        output_metrics['baseline'].append(baseline)
        q_s_a_list-=baseline
        batch_size=len(q_s_a_list)
    
        optimizer.zero_grad()
        state_batch_tensor=torch.tensor(state_np,device=device,dtype=torch.float32)        
        action_batch_tensor=torch.tensor(action_np,device=device)
        q_s_a_tensor=torch.tensor(q_s_a_list,device=device)

        logits_matrix=net(state_batch_tensor)
        log_prob_matrix=nn.functional.log_softmax(logits_matrix,dim=1)
        index=action_batch_tensor.long().unsqueeze(-1)
        log_prob_v=log_prob_matrix.gather(1, index).squeeze(-1)
        loss_policy=-1*(q_s_a_tensor * log_prob_v).mean()

        prob_matrix=nn.functional.softmax(logits_matrix,dim=1)
        entropy_v=-(prob_matrix * log_prob_matrix).sum(dim=1).mean()
        ENTROPY_BETA=HYPER_PARAMETERS['ENTROPY_BETA']
        loss_entropy=-ENTROPY_BETA * entropy_v

        loss_t=loss_entropy+loss_policy
        loss_t.backward()
        optimizer.step()

        output_metrics['loss_per_batch'].append(loss_t.item())
        output_metrics_average=metric_average_calc(output_metrics,output_metrics_average,alpha=0.99,MA_length=100)

        if training_counters['Num_of_batches'] % 50 ==0:
            print_metrics(output_metrics)
            print(training_counters['Num_of_batches'])
            print_metrics(output_metrics_average)
        
        batch = []
        Episodes_rewards=[]
        episode_count=0
        training_counters['Num_of_batches']+=1

################watch the result of best saved net visually
net.load_state_dict(torch.load("xxxxx.pth"))  # Load the best model
net.eval()  # Set to evaluation mode
agent=PGAgent(net,device=device, apply_softmax=True)


env = gym.make(HYPER_PARAMETERS['env_name'],render_mode="human")  
state, _ = env.reset()
done = False
while not done:
    env.render() 
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():  
        action = agent(state_tensor)
    state, reward, done, _ ,_= env.step(action[0][0])

env.close()  

def plot_dicts(dict1, dict2):

    all_keys = sorted(set(dict1.keys()) | set(dict2.keys()))  # Unique keys
    num_keys = len(all_keys)
    n = math.ceil(num_keys / 2)  # Calculate rows for 2 columns
    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))  # Create subplots
    axes = np.array(axes).reshape(-1)  # Flatten for easy indexing
    for i, key in enumerate(all_keys):
        ax = axes[i]
        if key in dict1:
            ax.plot(dict1[key], label="Dict 1", marker="o")
        if key in dict2:
            ax.plot(dict2[key], label="Dict 2", marker="s")
        ax.set_title(f"Key: {key}")
        ax.legend()
        ax.grid(True)

    # Hide unused subplots if odd number of keys
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

plot_dicts(output_metrics,output_metrics_average)