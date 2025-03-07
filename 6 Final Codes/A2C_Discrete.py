import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import ptan
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import models , utilities

HYPER_PARAMETERS={
                'env_name': "CartPole-v1",
                'stop_reward': 400,
                'batch_size': 16, 
                'n_step_unrolling':4,                 
                'learning_rate': 0.001,
                'gamma': 0.9,
                'ENTROPY_BETA': 0.01,
                'Num_Env':1,
                'Clip_Grad':0.1,
                'SEED':124,
                'MA_win':100,
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
value_net_cfg=[env.observation_space.shape[0] ,hiden_size,'ReLU',hiden_size,1]
net=models.A2CNetLinear(policy_net_cfg,value_net_cfg)
###################################################################
# Agent 
###################################################################

agent=models.A2CAgent(net,device=device, apply_softmax=True)

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
    for idx,exp in enumerate(utilities.exp_source_first_last(env,agent,HYPER_PARAMETERS['gamma'],HYPER_PARAMETERS['n_step_unrolling'],epsilon_obj=None)):

        if exp.total_reward:
            output_metrics['Episode_reward'].append(exp.total_reward)
            output_metrics['Episode_Steps'].append(exp.total_steps)

        batch.append(exp)
        if len(batch) < HYPER_PARAMETERS['batch_size']:
            continue

       
        state_np,action_np,reward_np,done_trunc_np,last_state_np=utilities.unpack_function_expsourcefirstlast_to_nparray(batch) 
        state_batch_tensor=torch.tensor(state_np,device=device,dtype=torch.float32)        
        action_batch_tensor=torch.tensor(action_np,device=device)
        reward_batch_tensor=torch.tensor(reward_np,device=device)
        last_state_batch_tensor=torch.tensor(last_state_np,device=device,dtype=torch.float32)
        done_trunc_tensor=torch.tensor(done_trunc_np,device=device)
        ref_val_bellman=models.bellman_equation_estimator_a2c(net, reward_batch_tensor,HYPER_PARAMETERS['gamma'],last_state_batch_tensor,done_trunc_tensor,HYPER_PARAMETERS['n_step_unrolling'])

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
agent=models.A2CAgent(net,device=device, apply_softmax=True)


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