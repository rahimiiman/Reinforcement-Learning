import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import gymnasium as gym
from gym import ObservationWrapper , Wrapper
from gym.spaces import Discrete, Box

device='cuda' if torch.cuda.is_available() else 'cpu'
class CustomLinear(nn.Module):
    def __init__(self, layer_config):
        super(CustomLinear, self).__init__()
        layers = []
        i = 0
        
        while i < len(layer_config):
            in_features = layer_config[i]
            out_features = layer_config[i + 1]
            layers.append(nn.Linear(in_features, out_features))
            
            if i + 2 < len(layer_config) and isinstance(layer_config[i + 2], str):
                activation = layer_config[i + 2].lower()
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif activation == 'softplus':
                    layers.append(nn.Softplus())

                else:
                    raise ValueError(f"Unsupported activation '{activation}'")
                i += 3
            else:
                i += 2
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        x = x.float()
        return self.net(x)

class DQNNetLinear(CustomLinear):
    def __init__(self, layer_config):
        super(DQNNetLinear, self).__init__(layer_config)

class DuelingDQNLinear(nn.Module):

    def __init__(self, value_net_cfg , adv_net_cfg): 
        super().__init__() 
        self.value_net=CustomLinear(value_net_cfg)
        self.adv_net=CustomLinear(adv_net_cfg)

    def forward(self,state):
        v_s= self.value_net(state)
        adv= self.adv_net(state)
        if adv.dim() == 1:
            adv = adv.unsqueeze(0)
            q_s_a_matrix= v_s + adv  - adv.mean(dim=1, keepdim=True)
            q_s_a_matrix= q_s_a_matrix.squeeze(-1)

        else:
            q_s_a_matrix= v_s + adv  - adv.mean(dim=1, keepdim=True)
        
        return  q_s_a_matrix

class TargetNet:
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        x = x.float()
        return self.target_model(x)
    
    def __call__(self, x):
        return self.forward(x)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)

class DQNAgent():
    def __init__ (self,q_net,epsilon=1):
        self.net=q_net
        self.epsilon=epsilon

    def forward(self,state):
        q_s_a=self.net(state)
        random_val=np.random.uniform(0, 1)
        if random_val <= self.epsilon:
            action = np.random.randint(len(q_s_a))
        else:
            q_s_a_np = q_s_a.detach().cpu().numpy()
            max_value = np.max(q_s_a_np)
            max_indices = np.flatnonzero(q_s_a_np == max_value)
            action = np.random.choice(max_indices)
            
        return action
    
    def __call__(self, state):
        return self.forward(state)

class A2CNetLinear(nn.Module):
    def __init__(self, policy_net_cfg,value_net_cfg):
        super(A2CNetLinear, self).__init__()

        self.policy = CustomLinear(policy_net_cfg)

        self.value = CustomLinear(value_net_cfg)

    def forward(self, x):
        return self.policy(x) , self.value(x)


class ReinforceAgent():
    def __init__(self, model, device="cpu", apply_softmax=True):
        self.model = model
        self.device = device
        self.apply_softmax = apply_softmax

    def __call__(self, states):
        states_v = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)

        logits_v= self.model(states_v)
        if logits_v.dim()==1:
            logits_v=logits_v.unsqueeze(0)

        if self.apply_softmax:
            probs_v = torch.nn.functional.softmax(logits_v, dim=1)
        else:
            probs_v = logits_v  # Assume model outputs probabilities directly
        probs = probs_v.cpu().data.numpy()
        actions = [np.random.choice(len(p), p=p) for p in probs]      #probs batch_size*act_size , p in probs mean a row in probs     

        if len(actions)==1:
            actions=actions[0]
        return actions


class A2CNetLinear(nn.Module):
    def __init__(self, policy_net_cfg,value_net_cfg):
        super(A2CNetLinear, self).__init__()

        self.policy = CustomLinear(policy_net_cfg)

        self.value = CustomLinear(value_net_cfg)

    def forward(self, x):
        return self.policy(x) , self.value(x)

class A2CAgent():
    def __init__(self, model, device="cpu", apply_softmax=True):
        self.model = model
        self.device = device
        self.apply_softmax = apply_softmax

    def __call__(self, states):
        states_v = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)

        logits_v,_ = self.model(states_v)
        if logits_v.dim()==1:
            logits_v=logits_v.unsqueeze(0)

        if self.apply_softmax:
            probs_v = torch.nn.functional.softmax(logits_v, dim=1)
        else:
            probs_v = logits_v  # Assume model outputs probabilities directly
        probs = probs_v.cpu().data.numpy()
        actions = [np.random.choice(len(p), p=p) for p in probs]      #probs batch_size*act_size , p in probs mean a row in probs     

        if len(actions)==1:
            actions=actions[0]
        return actions


class A2CContinuous(nn.Module):
    def __init__(self, base_net_cfg, mu_net_config , var_net_config,value_net_cfg):
        super(A2CContinuous, self).__init__()

        self.base_net = CustomLinear(base_net_cfg)
        self.mu_net = CustomLinear(mu_net_config)
        self.var_net = CustomLinear(var_net_config)
        self.value_net = CustomLinear(value_net_cfg)

    def forward(self, x: torch.Tensor):
        base_net_out = self.base_net(x)
        return self.mu_net(base_net_out), self.var_net(base_net_out)+1e-6, self.value_net(base_net_out)

class AgentA2CContinuous():
    def __init__(self, net,min_action_value,max_action_value,device):
        self.net = net
        self.device = device
        self.min_action_value=min_action_value
        self.max_action_value=max_action_value

    def __call__(self, states):
        states_v = torch.tensor(states,dtype=torch.float32)
        states_v = states_v.to(self.device)

        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, self.min_action_value,self.max_action_value)
        return actions


def bellman_equation_estimator_DQN(tgt_net, reward_tensor,gamma,next_state_tensor,done_trunc,n_step_unrolling):

    with torch.no_grad():

        #if last_state is terminal state v_s is equal to discounted reward reported by env
        #if last state is not terminal state , v_s is discounted reward of experienced segment + gamma^n * v_s_prime
        not_done_idx=done_trunc==False
        q_s_prime_a_prime_matrix=tgt_net(next_state_tensor)
        q_s_prime_a_prime_np = q_s_prime_a_prime_matrix.max(dim=1)[0].detach().cpu().numpy()
        q_s_a=np.array(reward_tensor)
        q_s_a[not_done_idx]=q_s_a[not_done_idx] + gamma**n_step_unrolling * q_s_prime_a_prime_np[not_done_idx]

        return torch.FloatTensor(q_s_a).to(device)


def bellman_equation_estimator_Double_DQN(q_net,tgt_net, reward_tensor,gamma,next_state_tensor,done_trunc,n_step_unrolling):

    with torch.no_grad():
        not_done_idx=done_trunc==False
        idx=q_net(next_state_tensor).argmax(dim=1)
        idx = idx.unsqueeze(-1)
        q_prime_s_prime_a_prime_matrix = tgt_net(next_state_tensor)
        q_prime_s_prime_a_prime_v = q_prime_s_prime_a_prime_matrix.gather(1, idx)
        q_prime_s_prime_a_prime_v = q_prime_s_prime_a_prime_v.squeeze(-1)  # Remove last dimension
        q_prime_s_prime_a_prime_v = q_prime_s_prime_a_prime_v.cpu().detach().numpy()  # Move to CPU and convert to numpy


        q_s_a=np.array(reward_tensor)
        q_s_a[not_done_idx]=q_s_a[not_done_idx] + gamma**n_step_unrolling * q_prime_s_prime_a_prime_v[not_done_idx]
        return torch.FloatTensor(q_s_a).to(device)


def bellman_equation_estimator_a2c(net, reward_tensor,gamma,next_state_tensor,done_trunc,n_step_unrolling):

    with torch.no_grad():

        #if last_state is terminal state v_s is equal to discounted reward reported by env
        #if last state is not terminal state , v_s is discounted reward of experienced segment + gamma^n * v_s_prime
        not_done_idx=done_trunc==False
        _,v_s_prime_tensor=net(next_state_tensor)
        v_s_prime_np=v_s_prime_tensor.data.cpu().numpy()[:, 0]
        v_s_np=np.array(reward_tensor)
        v_s_np[not_done_idx]=v_s_np[not_done_idx] + gamma**n_step_unrolling * v_s_prime_np[not_done_idx]
        return torch.FloatTensor(v_s_np).to(device)

def bellman_equation_estimator_a2ccontinuous(net, reward_tensor,gamma,next_state_tensor,done_tensor,n_step_unrolling):

    with torch.no_grad():

        not_done_idx=done_tensor==False
        _,_,v_s_prime_tensor=net(next_state_tensor)
        v_s_prime_np=v_s_prime_tensor.data.cpu().numpy()[:, 0]
        v_s_np=np.array(reward_tensor)
        v_s_np[not_done_idx]=v_s_np[not_done_idx] + gamma**n_step_unrolling * v_s_prime_np[not_done_idx]
        return torch.FloatTensor(v_s_np).to(device)



class FrozenLakeWrapper(Wrapper):
    def __init__(self, env):
        super(FrozenLakeWrapper, self).__init__(env)

        assert isinstance(env.observation_space, gym.spaces.Discrete), \
            "FrozenLakeWrapper only works with Discrete observation spaces."

        n = env.observation_space.n
        self.observation_space = Box(0, 1, shape=(n,), dtype=np.float32)

    def observation(self, obs):
        one_hot = np.zeros(self.observation_space.shape, dtype=np.float32)
        one_hot[obs] = 1.0
        return one_hot
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
    
    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        if next_state in [5, 7, 11, 12]:
            reward = -5
        elif next_state == 15:
            reward = +15
        else:
            reward = -1
        return self.observation(next_state), reward, done, truncated, info


class TabularAgent ():

    def __init__(self,env,gamma,alpha,epsilon=0):
        self.env=env
        self.num_states=env.observation_space.n
        self.num_actions=env.action_space.n
        self.state=None
        self.next_state=None
        self.gamma=gamma
        self.alpha=alpha
        self.epsilon=epsilon
    
    def Known_State_action_table_generator(self):
        self.q = np.zeros((self.num_states, self.num_actions))
     
    def agent_end(self, observation):

        reward= observation
        prev_state = self.prev_state
        prev_action = self.prev_action
        self.q[prev_state, prev_action] += self.step_size * (reward - self.q[prev_state, prev_action])

    def agent_step_TD0_Qlearning_greedy(self, observation):

        state = observation
        action=self.eps_greedy_action_selector(state)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        

        self.q[state, action] += self.alpha * (reward + self.gamma * np.max(self.q[next_state, :]) - self.q[state, action])


        return state,action,reward,next_state, terminated,truncated
    
    def eps_greedy_action_selector(self, observation):
 
        state = observation
        q_s_a_v = self.q[state,:]
        rand_val=np.random.uniform(0, 1)
        if  rand_val<= self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            max_value = np.max(q_s_a_v)
            max_indices = np.flatnonzero(q_s_a_v == max_value)
            action=np.random.choice(max_indices)

        return action
    
    def agent_step_TD0_Expected_Sarsa_greedy(self, observation,reward):

        state = observation
        prev_state = self.prev_state
        prev_action = self.prev_action
        
        current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon_greedy:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = np.argmax(current_q)

        ##Expected Sarsa Update
        expected_q = 0
        q_max = np.max(current_q) 
        epsilon_prob = np.ones(self.num_actions) * self.epsilon_greedy / self.num_actions # for all actions by mean
        greedy_prob = (current_q == q_max) * (1 - self.epsilon_greedy) / np.sum(current_q == q_max)
        pi = epsilon_prob + greedy_prob 
        expected_q = np.sum(pi * current_q) 
        self.q[prev_state, prev_action] += self.step_size * (reward + self.discount * expected_q - self.q[prev_state, prev_action])
        self.prev_state = state
        self.prev_action = action
        return action