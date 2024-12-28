import numpy as np



class my_agent ():

    def __init__(self,**agent_init_info):
        
        if 'num_states' in agent_init_info:
            self.num_states=agent_init_info['num_states']

        if 'step_size' in agent_init_info:
            self.step_size=agent_init_info['step_size']

        if 'num_actions' in agent_init_info:
            self.num_actions=agent_init_info['num_actions']

        if 'epsilon_greedy' in agent_init_info:
            self.epsilon_greedy=agent_init_info['epsilon_greedy']
        else:
            epsilon_greedy=0.1

        if 'discount' in agent_init_info:
            self.discount=agent_init_info['discount']
        else:
            discount=0.9

        if 'seed' in agent_init_info:
            self.rand_generator = np.random.RandomState(agent_init_info["seed"])
        else:
            self.rand_generator = np.random.RandomState(42)

    
    
    def Known_State_action_table_generator(self):                    #!
        self.q = np.zeros((self.num_states, self.num_actions))
    
    def agent_start_epsilon_greedy(self, observation):               #!
 
        state = observation
        current_q = self.q[state,:]
        if self.rand_generator.rand() < self.epsilon_greedy:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = np.argmax(current_q)
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_end(self, observation):                                #!

        reward= observation
        prev_state = self.prev_state
        prev_action = self.prev_action
        self.q[prev_state, prev_action] += self.step_size * (reward - self.q[prev_state, prev_action])

    def agent_step_TD0_Qlearning_greedy(self, observation,reward):    #!

        state = observation
        prev_state = self.prev_state
        prev_action = self.prev_action
        
        # Q-Learning update rule
        self.q[prev_state, prev_action] += self.step_size * (reward + self.discount * np.max(self.q[state, :]) - self.q[prev_state, prev_action])

        current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon_greedy:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = np.argmax(current_q)
        
        self.prev_state = state
        self.prev_action = action
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