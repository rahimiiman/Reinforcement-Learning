import gym
import numpy as np
import My_Agents
import matplotlib.pyplot as plt

env_Name='Taxi-v3'
env = gym.make(env_Name)

#Hyper Parameter setting
num_states=env.observation_space.n
num_actions=env.action_space.n
agent_init_info = {"num_actions": num_actions, "num_states": num_states, "epsilon_greedy": 0.1, "step_size": 0.8, "discount": 0.9}
convergence_thd = 1e-5  
max_steps_per_episode = 200
num_episodes = 500

############################################################ Q Learning
agent = My_Agents.my_agent(**agent_init_info)
agent.Known_State_action_table_generator() 
Total_Reward_Tracker_Q = []
converged = False
episode = 0

while  episode < num_episodes:
    state, _ = env.reset()
    done = False
    total_reward_current_episode = 0  
    action = agent.agent_start_epsilon_greedy(state)


    for step in range(max_steps_per_episode):
        state, reward, done, _, _ = env.step(action)
        total_reward_current_episode += reward
        q_before_update=agent.q
        if not done:
            action = agent.agent_step_TD0_Qlearning_greedy(state, reward)
        else:
            agent.agent_end(reward)
            break
        q_after_update = agent.q
        max_absolute_difference = np.max(np.abs(q_after_update - q_before_update))
        if max_absolute_difference <= convergence_thd:
            converged=True

    Total_Reward_Tracker_Q.append(total_reward_current_episode)

    print('************episode num= ', episode,' ***************total_reward_current_episode= ', total_reward_current_episode)
    episode += 1


############################################################ Expected Sarsa
agent = My_Agents.my_agent(**agent_init_info)
agent.Known_State_action_table_generator() 
Total_Reward_Tracker_ES = []
converged = False
episode = 0

while  episode < num_episodes:
    state, _ = env.reset()
    done = False
    total_reward_current_episode = 0  
    action = agent.agent_start_epsilon_greedy(state)


    for step in range(max_steps_per_episode):
        state, reward, done, _, _ = env.step(action)
        total_reward_current_episode += reward
        q_before_update=agent.q
        if not done:
            action = agent.agent_step_TD0_Expected_Sarsa_greedy(state, reward)
        else:
            agent.agent_end(reward)
            break
        q_after_update = agent.q
        max_absolute_difference = np.max(np.abs(q_after_update - q_before_update))
        if max_absolute_difference <= convergence_thd:
            converged=True

    Total_Reward_Tracker_ES.append(total_reward_current_episode)

    print('************episode num= ', episode,' ***************total_reward_current_episode= ', total_reward_current_episode)
    episode += 1

env.close()

###################################
plt.figure(figsize=(12, 6))
plt.plot(Total_Reward_Tracker_ES, label='Expected Sarsa')
plt.plot(Total_Reward_Tracker_Q, label='Q Learning')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards Over Episodes')
plt.legend()
plt.grid()
plt.show()

