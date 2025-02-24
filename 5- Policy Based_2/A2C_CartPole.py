import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import ptan
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.pyplot as plt
import math


HYPER_PARAMETERS={
                'env_name': "CartPole-v1",
                'stop_reward': 350,
                'run_name': 'CartPole',
                'batch_size': 128, 
                'n_step_unrolling':4,                 
                'learning_rate': 0.001,
                'gamma': 0.9,
                'ENTROPY_BETA': 0.01,
                'Num_Env':1,
                'Clip_Grad':0.1,
                'SEED':124
                }

####################################################################set the seed

device='cuda' if torch.cuda.is_available() else 'cpu'
output_metrics=defaultdict(list)
output_metrics_average=defaultdict(list)

###################################################################
#ENV
###################################################################
def make_env(env,n):
    envs=[]
    for _ in range(n):
        envs.append(env)
    return envs

env=gym.make(HYPER_PARAMETERS['env_name'])
env=make_env(env,HYPER_PARAMETERS['Num_Env'])
###################################################################
# Policy  Gradient Net
###################################################################
class A2CNet(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(A2CNet, self).__init__()

        self.policy = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.policy(x) , self.value(x)

hiden_size=128
net=A2CNet(env[0].observation_space.shape[0],hiden_size,env[0].action_space.n)
###################################################################
# Agent 
###################################################################
class A2CAgent(ptan.agent.BaseAgent):
    def __init__(self, model, device="cpu", apply_softmax=True):
        self.model = model
        self.device = device
        self.apply_softmax = apply_softmax

    def __call__(self, states, agent_states=None):
        states_v = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)

        logits_v,_ = self.model(states_v)
        if self.apply_softmax:
            probs_v = torch.nn.functional.softmax(logits_v, dim=1)
        else:
            probs_v = logits_v  # Assume model outputs probabilities directly
        probs = probs_v.cpu().data.numpy()
        actions = [np.random.choice(len(p), p=p) for p in probs]      #probs batch_size*act_size , p in probs mean a row in probs     

        return actions,agent_states

agent=A2CAgent(net,device=device, apply_softmax=True)
exp_source=ptan.experience.ExperienceSourceFirstLast(env,agent,gamma=HYPER_PARAMETERS['gamma'],steps_count=HYPER_PARAMETERS['n_step_unrolling'])


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
        elif len(values)>1:
            metrics_average['RunningAverage_'+key].append(alpha*values[-2]+(1-alpha)*values[-1])
    return metrics_average

def unpack_function_expsourcefirstlast_to_nparray(batch): 

    states, actions, rewards, dones, last_states = [],[],[],[],[]
    for exp in batch:
        state = np.array(exp.state)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            lstate = state  # the result will be masked anyway
        else:
            lstate = np.array(exp.last_state)
        last_states.append(lstate)
    return np.array(states, copy=False), np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), \
           np.array(last_states, copy=False)


def bellman_equation_estimator(net, reward_tensor,gamma,next_state_tensor,n_step_unrolling):

    with torch.no_grad():

        #if last_state is terminal state v_s is equal to discounted reward reported by env
        #if last state is not terminal state , v_s is discounted reward of experienced segment + gamma^n * v_s_prime
        not_done_idx=next_state_tensor!=None
        _,v_s_prime_tensor=net(next_state_tensor)
        v_s_prime_np=v_s_prime_tensor.data.cpu().numpy()[:, 0]
        v_s_np=np.array(reward_tensor)
        v_s_np[not_done_idx]=v_s_np[not_done_idx] + gamma**n_step_unrolling * v_s_prime_np[not_done_idx]
        return torch.FloatTensor(v_s_np).to(device)


###################################################################
# Training Loop
###################################################################

optimizer=optim.Adam(net.parameters(),HYPER_PARAMETERS['learning_rate'])


training_flag=True
best_reward=0
training_counters=defaultdict(int)
while training_flag:

    batch = []
    Episodes_rewards=[]
    Episodes_steps=[]
    for idx,exp in enumerate(exp_source):

        reward_steps=exp_source.pop_rewards_steps()

        if reward_steps:
            output_metrics['Episode_reward'].append(reward_steps[0][0])
            output_metrics['Episode_Steps'].append(reward_steps[0][1])

        batch.append(exp)
        if len(batch) < HYPER_PARAMETERS['batch_size']:
            continue


        ######################################
        #End of Training Condition & saving trained net
        ######################################

        moving_average_100_reward=output_metrics_average['MA_Episode_reward']

        if len(moving_average_100_reward)>0:
            if moving_average_100_reward[-1] > best_reward:
                best_reward=moving_average_100_reward[-1]
                torch.save(net.state_dict(),'A2C_Cartpole.pth')

        if best_reward > HYPER_PARAMETERS['stop_reward']:
            training_flag=False
            break



        
        state_np,action_np,reward_np,done_trunc_np,last_state_np=unpack_function_expsourcefirstlast_to_nparray(batch) 
    
        state_batch_tensor=torch.tensor(state_np,device=device,dtype=torch.float32)        
        action_batch_tensor=torch.tensor(action_np,device=device)
        reward_batch_tensor=torch.tensor(reward_np,device=device)
        last_state_batch_tensor=torch.tensor(last_state_np,device=device,dtype=torch.float32)

        ref_val_bellman=bellman_equation_estimator(net, reward_batch_tensor,HYPER_PARAMETERS['gamma'],last_state_batch_tensor,HYPER_PARAMETERS['n_step_unrolling'])

        optimizer.zero_grad()
        logits_matrix,ref_val_tensor=net(state_batch_tensor)

        loss_value=nn.functional.mse_loss(ref_val_tensor.squeeze(-1),ref_val_bellman)

        adv_tensor = ref_val_bellman - ref_val_tensor.detach()
        log_prob_matrix=nn.functional.log_softmax(logits_matrix,dim=1)
        index=action_batch_tensor.long().unsqueeze(-1)
        log_prob_v=log_prob_matrix.gather(1, index).squeeze(-1)
        loss_policy=-1*(adv_tensor * log_prob_v).mean()
        
        
        prob_matrix=nn.functional.softmax(logits_matrix,dim=1)
        entropy_v=-(prob_matrix * log_prob_matrix).sum(dim=1).mean()
        ENTROPY_BETA=HYPER_PARAMETERS['ENTROPY_BETA']
        loss_entropy=-ENTROPY_BETA * entropy_v

        loss_t=loss_policy + loss_entropy + loss_value
        loss_t.backward()
        nn.utils.clip_grad_norm_(net.parameters(), HYPER_PARAMETERS['Clip_Grad'])
        optimizer.step()

        #KL Divergence
        with torch.no_grad():
            new_logits_matrix,_=net(state_batch_tensor)
            new_prob_matrix=nn.functional.softmax(new_logits_matrix,dim=1)
            kl_div_t = -((new_prob_matrix / prob_matrix).log() * prob_matrix).sum(dim=1).mean()


        output_metrics['loss_per_batch'].append(loss_t.item())
        output_metrics['kl_div_t'].append(kl_div_t.item())
        output_metrics_average=metric_average_calc(output_metrics,output_metrics_average,alpha=0.99,MA_length=100)

        if training_counters['Num_of_batches'] % 50 ==0:
            print_metrics(output_metrics)
            #print(training_counters['Num_of_batches'])
            print_metrics(output_metrics_average)
        
        batch = []
        training_counters['Num_of_batches']+=1


################watch the result of best saved net visually
net.load_state_dict(torch.load("A2C_Cartpole.pth"))  # Load the best model
net.eval()  # Set to evaluation mode
agent=A2CAgent(net,device=device, apply_softmax=True)


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