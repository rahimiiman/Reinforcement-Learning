In previous folders(1 to 5) Ignite and PTAN libraries used to handle interaction between Agent+ENV and also control training loop.


However these libraries are fascinating to reduce time of coding and also offer perfect features their maintenance and upgrade process in not aligned with torch and NumPy.
Also they are not compatible with latest version of gymnasium and many times they work with older version of gym.

So all codes are rewritten and these two libraries are removed and their functionality is implemented manually.


summary of folder codes:

1- Dynamic Programming (DP) RLs : (1-state_value_iteration , 2- action_value_iteration methods)

2- Tabular methods : (1- Q_Learning )

3- Value-Based DRL ( 1-DQN , 2-DoubleDQN , 3-DuelingDQN)

4- Policy-Based DRL (1-Reinforce 2-A2C for both discrete and continuous action spaces)


Future updates :

I'm providing codes for SARSA / Expectd SARSA / SAC / PPO / DDPG