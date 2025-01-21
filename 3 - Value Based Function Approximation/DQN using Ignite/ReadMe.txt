This notebook is designed to solve CartPole environment using Deep Q learning(DQN).

Structure of code is same as basic DQN code already provided however in this code we used Ignite framework for training loops.

The reason is because from now on, it will be the foundation of codes I will share in the future. 

in HYPER_PARAMETERS cell , there are some hyper parameters if you want to try different scenarios and compare results.

Condition for stopping training loop is to reach running average reward greater than "stop_reward" (a hyperparameter you can change).


I recommend you to run this notebook on Kaggle , if so , at the end of runtime , it will save the state_dict of trained NN at Kaggle output.
if you don't want to use it in Kaggle , you may need to make a modification for output path. 


if you have any question feel free to reach out to me by Email : rahimi.iman587@gmail.com



