import gym
import numpy as np
import My_Agents
import matplotlib.pyplot as plt

env = gym.make('CliffWalking-v0')

#Hyper Parameter setting
num_states=env.observation_space.n
num_actions=env.action_space.n
agent_init_info = {"num_actions": num_actions, "num_states": num_states, "epsilon_greedy": 0.01, "step_size": 0.9, "discount": 1.0}

## agent
agent = My_Agents.my_agent(**agent_init_info)
agent.Known_State_action_table_generator()

# Training parameters
convergence_thd = 1e-5  # Convergence threshold for action value change
max_steps_per_episode = 100
num_episodes = 100  # Maximum episodes to prevent infinite loops

# Track total rewards
Total_Reward_Tracker = []

converged = False
episode = 0

while  episode < num_episodes:
    # Reset the environment for a new episode
    state, _ = env.reset()
    done = False
    total_reward_current_episode = 0  # Initialize total reward for the episode

    # Use agent's method to choose the initial action
    action = agent.agent_start_epsilon_greedy(state)


    for step in range(max_steps_per_episode):
        # Take the action in the environment
        state, reward, done, _, _ = env.step(action)
        
        # Track the total reward
        total_reward_current_episode += reward

        # Track the previous Q-value for convergence check
        q_before_update=agent.q

        if not done:
            # Choose the next action using the agent's step method
            action = agent.agent_step_TD0_Qlearning_greedy(state, reward)
        else:
            # End the episode and update the Q-table for the last state
            agent.agent_end(reward)
            break
        
        # Track the change in Q-value
        q_after_update = agent.q
        max_absolute_difference = np.max(np.abs(q_after_update - q_before_update))
        if max_absolute_difference <= convergence_thd:
            converged=True
   
    
    # Append the total reward of the episode to the list
    Total_Reward_Tracker.append(total_reward_current_episode)

    print('************episode num= ', episode,' ***************total_reward_current_episode= ', total_reward_current_episode)
    episode += 1

env.close()

# Print the learned Q-table and number of episodes until convergence
print("Learned Q-table:")
print(agent.q)
print(f"Convergence reached after {episode} episodes" if converged else "Max episodes reached without convergence")

# Plot the total rewards for each episode
plt.figure(figsize=(12, 6))
plt.plot(Total_Reward_Tracker, label='Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards Over Episodes')
plt.legend()
plt.grid()
plt.show()