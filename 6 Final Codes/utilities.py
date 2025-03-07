import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple , deque



class EpsilonDecay():
    def __init__(self,init_epsilon,min_epsilon,epsilon_delta):
        self.init_epsilon=init_epsilon
        self.cur_epsilon=init_epsilon
        self.min_epsilon=min_epsilon
        self.update_counts=0
        self.epsilon_delta=epsilon_delta
    
    def update_epsilon(self):
        self.update_counts+=1

        if self.cur_epsilon > self.min_epsilon + self.epsilon_delta:
            self.cur_epsilon= self.init_epsilon - self.update_counts * self.epsilon_delta
            return self.cur_epsilon
        else:
            return self.min_epsilon

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def _add(self, experince):
        self.buffer.append(experince)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        return batch

    def __len__(self):
        return len(self.buffer)

def print_metrics(metrics):
    print(" | ".join(f"{key}: {values[-1]:.2f}" for key, values in metrics.items() if values))

def metric_average_calc(metrics,metrics_average,alpha=0.99,MA_length=100):
    for key, values in metrics.items():
        if len(values)>MA_length:
            metrics_average['MA_'+key].append(sum(values[-1*MA_length:])/MA_length)
        else:
            metrics_average['MA_'+key].append(sum(values)/len(values))
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
        dones.append(exp.done_trunc)
        if exp.next_state is None:
            lstate = state  # the result will be masked anyway
        else:
            lstate = np.array(exp.next_state)
        last_states.append(lstate)
    return np.array(states, copy=False), np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), \
           np.array(last_states, copy=False)

def rewards_to_go_calc(immediate_reward_list,is_done_trunc,gamma):

    T=len(immediate_reward_list)
    Episodes_reward_tracker=[]           #len() = number of total episodes(last episode may not be completed)
    Episodes_Steps_Tracker=[]            ##len() = number of total episodes(last episode may not be completed)
    rewards_to_go=[0]*T
    discounted_reward=0
    N_Steps=0
    for t in range(T-1,-1,-1):
        if (is_done_trunc[t] and t!=T-1):
            Episodes_reward_tracker.append(discounted_reward)
            Episodes_Steps_Tracker.append(N_Steps)
            N_Steps=0
            discounted_reward=0
            discounted_reward=immediate_reward_list[t] + gamma * discounted_reward
            rewards_to_go[t]=discounted_reward
        
        elif t==0:
            N_Steps+=1
            discounted_reward=immediate_reward_list[t] + gamma * discounted_reward
            rewards_to_go[t]=discounted_reward
            Episodes_reward_tracker.append(discounted_reward)
            Episodes_Steps_Tracker.append(N_Steps)        
        else:
            N_Steps+=1
            discounted_reward=immediate_reward_list[t] + gamma * discounted_reward
            rewards_to_go[t]=discounted_reward
    return rewards_to_go,Episodes_reward_tracker,Episodes_Steps_Tracker


experince_first_last=namedtuple('experince_first_last',['state','action','reward','next_state','done_trunc','total_reward','total_steps'])
def exp_source_first_last(env, agent, gamma,n_step_unrolling,epsilon_obj=None):
    episode_steps = 0
    episode_rewards = []
    is_done_trunc = []
    state, _ = env.reset()  

    while True:
        cnt_steps=0
        discounted_reward=0
        for i in range(n_step_unrolling):
            cnt_steps+=1
            if hasattr(agent, 'epsilon'):                     # check if agent has dependency to epsilon, used in eps_greedy agents
                agent.epsilon=epsilon_obj.update_epsilon()  
            actions= agent(state) 
            next_state, reward, terminated, truncated, _ = env.step(actions)
            discounted_reward=reward + gamma * discounted_reward
            episode_rewards.append(reward)
            episode_steps += 1
            is_done_trunc.append(int(terminated or truncated))

            if terminated or truncated:
                total_reward = rewards_to_go_calc(episode_rewards, is_done_trunc, gamma)[1][0]
                yield experince_first_last(
                    state=state, action=actions, reward=discounted_reward, 
                    next_state=None, done_trunc=True, 
                    total_reward=total_reward, total_steps=episode_steps
                )
                # Reset for next episode
                state, _ = env.reset()
                episode_rewards.clear()
                is_done_trunc.clear()
                episode_steps = 0
                discounted_reward=0
                cnt_steps=0
                i=n_step_unrolling-1
            elif cnt_steps==n_step_unrolling:
                yield experince_first_last(
                    state=state, action=actions, reward=discounted_reward, 
                    next_state=next_state, done_trunc=False, 
                    total_reward=None, total_steps=None
                )
                state = next_state  # Move to next state
                discounted_reward=0
                cnt_steps=0
            else:
                state = next_state  # Move to next state
