import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import math
from collections import namedtuple , deque ,defaultdict
import models , utilities


###############################################
# Hyperparameters Setting
###############################################
HYPER_PARAMETERS={
                'env_name': "CartPole-v1",
                'stop_reward':150 ,
                'REPLAY_BUFFER_SIZE':100000,
                'Replay_Buffer_sampling_start':2,
                'batch_size': 32, 
                'TGT_NET_SYNC':100,                 
                'learning_rate': 0.001,
                'gamma': 0.9,
                'initial_epsilon':1,
                'min_epsilon':.01,
                'epsilon_delta':0.0001,
                'n_step_unrolling':1,
                'MA_win':10,
                }


###############################################
# Model Definition
###############################################
device='cuda' if torch.cuda.is_available() else 'cpu'
output_metrics=defaultdict(list)
output_metrics_average=defaultdict(list)

env = gym.make(HYPER_PARAMETERS['env_name'])
#env=models.FrozenLakeWrapper(env)

HIDDEN_SIZE=256
layer_config=[env.observation_space.shape[0],HIDDEN_SIZE,'ReLU',HIDDEN_SIZE,HIDDEN_SIZE,'ReLU',HIDDEN_SIZE,env.action_space.n]
q_net=models.DQNNetLinear(layer_config)
tgt_net=models.TargetNet(q_net)
agent=models.DQNAgent(q_net)
epsilon_obj=utilities.EpsilonDecay(HYPER_PARAMETERS['initial_epsilon'] , HYPER_PARAMETERS['min_epsilon'] , HYPER_PARAMETERS['epsilon_delta'])

replay_buffer_obj=utilities.ReplayBuffer(HYPER_PARAMETERS['REPLAY_BUFFER_SIZE'])

###############################################
# Training Loop
###############################################

optimizer=optim.Adam(q_net.parameters(),HYPER_PARAMETERS['learning_rate'])
training_flag=True
best_reward=0
training_counters=defaultdict(int)
while training_flag:
    Episodes_rewards=[]
    Episodes_steps=[]
    output_metrics['Epsilon'].append(agent.epsilon)
    for idx,exp in enumerate(utilities.exp_source_first_last(env,agent,HYPER_PARAMETERS['gamma'],HYPER_PARAMETERS['n_step_unrolling'],epsilon_obj)):
        if exp.total_reward:
            output_metrics['Episode_reward'].append(exp.total_reward)
            output_metrics['Episode_Steps'].append(exp.total_steps)

        replay_buffer_obj._add(exp)
        output_metrics['Epsilon'].append(agent.epsilon)
        if len(replay_buffer_obj) < HYPER_PARAMETERS['REPLAY_BUFFER_SIZE'] / HYPER_PARAMETERS['Replay_Buffer_sampling_start']:
            continue
        
        output_metrics['Epsilon'].append(agent.epsilon)

        batch=replay_buffer_obj.sample(HYPER_PARAMETERS['batch_size'])      
        state_np,action_np,reward_np,done_trunc_np,last_state_np=utilities.unpack_function_expsourcefirstlast_to_nparray(batch) 
        state_batch_tensor=torch.tensor(state_np,device=device,dtype=torch.float32)        
        action_batch_tensor=torch.tensor(action_np,device=device,dtype=torch.int64)
        reward_batch_tensor=torch.tensor(reward_np,device=device)
        last_state_batch_tensor=torch.tensor(last_state_np,device=device,dtype=torch.float32)
        done_trunc_tensor=torch.tensor(done_trunc_np,device=device)

        optimizer.zero_grad()
        q_prime_s_a_bellman=models.bellman_equation_estimator_DQN(tgt_net,reward_batch_tensor,HYPER_PARAMETERS['gamma'],last_state_batch_tensor,done_trunc_tensor,HYPER_PARAMETERS['n_step_unrolling'])
        
        q_s_a_matrix = q_net(state_batch_tensor)
        predicted_q_s_a = q_s_a_matrix.gather(1, action_batch_tensor.unsqueeze(-1)).squeeze(-1)

        loss_v = nn.functional.mse_loss(predicted_q_s_a, q_prime_s_a_bellman)
        loss_v.backward()
        optimizer.step()
        output_metrics['loss_per_batch'].append(loss_v.item())
        output_metrics_average=utilities.metric_average_calc(output_metrics,output_metrics_average,alpha=0.99,MA_length=HYPER_PARAMETERS['MA_win'])

        if training_counters['Num_of_batches'] % 200 ==0:
            utilities.print_metrics(output_metrics)
            utilities.print_metrics(output_metrics_average)
        
        training_counters['Num_of_batches']+=1

        if training_counters['Num_of_batches'] % HYPER_PARAMETERS['TGT_NET_SYNC']:
            tgt_net.sync()
        ######################################
        #End of Training Condition & saving trained net
        ######################################
        moving_average_reward=output_metrics_average['MA_Episode_Steps']
        if len(moving_average_reward)>0:
            if moving_average_reward[-1] > best_reward:
                best_reward=moving_average_reward[-1]
                torch.save(q_net.state_dict(),'DQN.pth')

        if best_reward > HYPER_PARAMETERS['stop_reward']:
            training_flag=False
            break


################watch the result of best saved net visually
q_net.load_state_dict(torch.load("DQN.pth"))  # Load the best model
q_net.eval()  # Set to evaluation mode
agent=models.DQNAgent(q_net)
agent.epsilon=-1
env=gym.make(HYPER_PARAMETERS['env_name'],render_mode="human")

for _ in range(5):
    state, _ = env.reset()
    done = False
    while not done:
        env.render() 

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():  
            actions= agent(state_tensor) 
        state, reward, done, _ ,_= env.step(actions)

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