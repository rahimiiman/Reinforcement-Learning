import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import models , utilities

HYPER_PARAMETERS={
                'env_name': "CartPole-v1",
                'stop_reward': 50,
                'batch_size': 128, 
                'n_step_unrolling':1,                 
                'learning_rate': 0.001,
                'gamma': 0.9,
                'ENTROPY_BETA': 0.01,
                'Num_Env':1,
                'SEED':124,
                'MA_win':10,
                }

####################################################################set the seed

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
hiden_size=128
policy_net_cfg=[env.observation_space.shape[0] ,hiden_size,'ReLU',hiden_size,hiden_size,'ReLU',hiden_size,env.action_space.n]
net=models.CustomLinear(policy_net_cfg)
###################################################################
# Agent 
###################################################################

agent=models.ReinforceAgent(net,device=device, apply_softmax=True)

###################################################################
# Training Loop
###################################################################

optimizer=optim.Adam(net.parameters(),HYPER_PARAMETERS['learning_rate'])

training_flag=True
best_reward=0
training_counters=defaultdict(int)
while training_flag:

    batch = []
    baseline=0
    Episodes_rewards=[]
    Episodes_steps=[]
    for idx,exp in enumerate(utilities.exp_source_first_last(env,agent,HYPER_PARAMETERS['gamma'],HYPER_PARAMETERS['n_step_unrolling'],epsilon_obj=None)):


        if exp.total_reward:
            output_metrics['Episode_reward'].append(exp.total_reward)
            output_metrics['Episode_Steps'].append(exp.total_steps)

        batch.append(exp)
        baseline+=exp.reward
        if len(batch) < HYPER_PARAMETERS['batch_size']:
            continue

        baseline=baseline / HYPER_PARAMETERS['batch_size']
       
        state_np,action_np,reward_np,done_trunc_np,last_state_np=utilities.unpack_function_expsourcefirstlast_to_nparray(batch) 

        adv_np=reward_np - baseline

        state_batch_tensor=torch.tensor(state_np,device=device,dtype=torch.float32)        
        action_batch_tensor=torch.tensor(action_np,device=device)
        adv_batch_tensor=torch.tensor(adv_np,device=device)
        
        optimizer.zero_grad()

        logits_matrix=net(state_batch_tensor)
        log_prob_matrix=nn.functional.log_softmax(logits_matrix,dim=1)
        index=action_batch_tensor.long().unsqueeze(-1)
        log_prob_v=log_prob_matrix.gather(1, index).squeeze(-1)
        loss_policy=-1*(adv_batch_tensor * log_prob_v).mean()
        
        
        prob_matrix=nn.functional.softmax(logits_matrix,dim=1)
        entropy_v=-(prob_matrix * log_prob_matrix).sum(dim=1).mean()
        ENTROPY_BETA=HYPER_PARAMETERS['ENTROPY_BETA']
        loss_entropy=-ENTROPY_BETA * entropy_v

        loss_t=loss_policy + loss_entropy
        loss_t.backward()
        optimizer.step()
       

        output_metrics['loss_per_batch'].append(loss_t.item())
        output_metrics_average=utilities.metric_average_calc(output_metrics,output_metrics_average,alpha=0.99,MA_length=HYPER_PARAMETERS['MA_win'])

        if training_counters['Num_of_batches'] % 50 ==0:
            utilities.print_metrics(output_metrics)
            utilities.print_metrics(output_metrics_average)
        
        batch = []
        training_counters['Num_of_batches']+=1

        ######################################
        #End of Training Condition & saving trained net
        ######################################

        moving_average_reward=output_metrics_average['MA_Episode_Steps']

        if len(moving_average_reward)>0:
            if moving_average_reward[-1] > best_reward:
                best_reward=moving_average_reward[-1]
                torch.save(net.state_dict(),'A2C_Discrete.pth')

        if best_reward > HYPER_PARAMETERS['stop_reward']:
            training_flag=False
            break



################watch the result of best saved net visually
net.load_state_dict(torch.load("A2C_Discrete.pth"))  # Load the best model
net.eval()  # Set to evaluation mode
agent=models.ReinforceAgent(net,device=device, apply_softmax=True)


env = gym.make(HYPER_PARAMETERS['env_name'],render_mode="human")  
state, _ = env.reset()
done = False
while not done:
    env.render() 
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():  
        action = agent(state_tensor)
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