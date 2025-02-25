This folder includes some policy gradient RL methods. 
First method in PG family is reinforce that feeds one state to Neural Network and provide a discrete probability distribution function for each value in action space.
Reinforce usually used with Envs with discrete action space and output is a stochastic policy over actin space domain.

some Enhancement has been proposed for Reinforce method by different authors in order to improve stability of convergence.
One of drawbacks of Reinforce is the need to full episode. In other words we have to wait till end of episode in order to calculate discounted reward recursively.
This is not challenging for simple Envs like CartPle that we used in our code. However for more complex Envs it may take long time for one episode t be completed. 
to cover this issue , a version of code is also provided using u step unrolling future reward calculation.
Usually in reinforce method we need to subtract a base line from discounted reward in order to reduce variance and increase stability of convergence.
Furthermore, an entropy term is considered in objective function to prevent getting stuck in a specific action/state pair and perform exploration/exploitation dilemma.


Advantage Actor-Critic is a widely used method among PG family which use a second NN to estimate baseline . 

More PG family codes will be added gradually.
  