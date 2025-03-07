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
import time


HYPER_PARAMETERS={
                'env_name':"Ant-v4",
                'stop_reward':-10,
                'batch_size': 1e4, 
                'n_step_unrolling':1,                 
                'learning_rate': 1e-3,
                'gamma': 0.99,
                'ENTROPY_BETA': 0.01,
                'MA_win':10,
                'SEED':124
                }

####################################################################set the seed

device='cuda' if torch.cuda.is_available() else 'cpu'
output_metrics=defaultdict(list)
output_metrics_average=defaultdict(list)

###################################################################
#ENV
###################################################################
env = gym.make(HYPER_PARAMETERS['env_name'])
###################################################################
# Policy  Gradient Net
###################################################################

hiden_size=128
base_net_output=128
basse_config=[env.observation_space.shape[0],hiden_size,'ReLU',hiden_size,base_net_output,'ReLU']
mu_net_config=[base_net_output,env.action_space.shape[0],'Tanh']
var_net_config=[base_net_output,env.action_space.shape[0],'Softplus']
value_net_config=[base_net_output,1]

net=models.A2CContinuous(basse_config,mu_net_config,var_net_config,value_net_config).to(device)
###################################################################
# Agent 
###################################################################
agent=models.AgentA2CContinuous(net,-1,1,device=device)

###################################################################
# utility functions
###################################################################

def calc_logprob(mu_tensor, var_tensor, actions_tensor):
    p1 = - ((actions_tensor - mu_tensor) ** 2) / (2*var_tensor.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_tensor))
    return p1 + p2


###################################################################
# Training Loop
###################################################################

optimizer=optim.Adam(net.parameters(),HYPER_PARAMETERS['learning_rate'])


training_flag=True
best_reward=-10000
training_counters=defaultdict(int)
while training_flag:

    batch = []
    Episodes_rewards=[]
    Episodes_steps=[]
    for idx,exp in enumerate(utilities.exp_source_first_last(env,agent,HYPER_PARAMETERS['gamma'],HYPER_PARAMETERS['n_step_unrolling'],epsilon_obj=None)):


        if exp.total_reward:
            output_metrics['Episode_reward'].append(exp.total_reward)
            output_metrics['Episode_Steps'].append(exp.total_steps)

        batch.append(exp)
        if len(batch) < HYPER_PARAMETERS['batch_size']:
            continue


        ######################################
        #End of Training Condition & saving trained net
        ######################################       
        state_np,action_np,reward_np,done_trunc_np,last_state_np=utilities.unpack_function_expsourcefirstlast_to_nparray(batch) 
    
        state_batch_tensor=torch.tensor(state_np,device=device,dtype=torch.float32)        
        action_batch_tensor=torch.tensor(action_np,device=device)
        reward_batch_tensor=torch.tensor(reward_np,device=device)
        last_state_batch_tensor=torch.tensor(last_state_np,device=device,dtype=torch.float32)
        done_batch_tensor=torch.tensor(done_trunc_np,device=device)

        ref_val_bellman=models.bellman_equation_estimator_a2ccontinuous(net, reward_batch_tensor,HYPER_PARAMETERS['gamma'],last_state_batch_tensor,done_batch_tensor,HYPER_PARAMETERS['n_step_unrolling'])

        optimizer.zero_grad()
        mu_tensor, var_tensor, value_tensor = net(state_batch_tensor)

        loss_value = nn.functional.mse_loss(value_tensor.squeeze(-1), ref_val_bellman)
        adv_tensor = ref_val_bellman.unsqueeze(dim=-1) - value_tensor.detach()
        log_prob_v = adv_tensor * calc_logprob(mu_tensor, var_tensor, action_batch_tensor)
        loss_policy = -log_prob_v.mean()

        ent_v=0.5 * (torch.log(2 * torch.pi * var_tensor) + 1)
        loss_entropy = -HYPER_PARAMETERS['ENTROPY_BETA'] * ent_v.mean()

        loss_total = loss_value + loss_policy + loss_entropy
        loss_total.backward()
        optimizer.step()


        output_metrics['loss_per_batch'].append(loss_total.item())
        output_metrics_average=utilities.metric_average_calc(output_metrics,output_metrics_average,alpha=0.99,MA_length=HYPER_PARAMETERS['MA_win'])

        if training_counters['Num_of_batches'] % 1 ==0:
            utilities.print_metrics(output_metrics)
            utilities.print_metrics(output_metrics_average)
        
        batch = []
        training_counters['Num_of_batches']+=1

        moving_average_reward=output_metrics_average['MA_Episode_reward']

        if len(moving_average_reward)>0:
            if moving_average_reward[-1] > best_reward:
                best_reward=moving_average_reward[-1]
                torch.save(net.state_dict(),'A2C_Continuous.pth')

        if best_reward > HYPER_PARAMETERS['stop_reward']:
            training_flag=False
            break


################watch the result of best saved net visually
net.load_state_dict(torch.load("A2C_Continuous.pth"))  # Load the best model
net.eval()  # Set to evaluation mode
agent=models.AgentA2CContinuous(net,-1,1,device=device)


start_time = time.time()
 
env=gym.make(HYPER_PARAMETERS['env_name'],render_mode="human")
state, _ = env.reset()
done = False
for _ in range(1):

    while not done:
        env.render() 
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():  
            actions= agent(state_tensor) 
        state, reward, done, _ ,_= env.step(actions.squeeze(0))
        if time.time() - start_time > 120:
            break

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