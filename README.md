# online learning transformer_rl-mujoco-half-cheetah-experiment-reward change(in progress)
This experiment follows three steps. collecting data from normal rl env. training custome transformer model with collected data trajectory. training new rl agent with trained transformer model in normal rl env

## 1) normal rl trained agent ##
![normal+rl+trained+half+cheetah+agent (1)](https://github.com/user-attachments/assets/ae9b8d88-bc68-4b8d-bd18-e18d51aca74d)

## 2) transformer rl trained agent ##
![transofrmer+rl+trained+half+cheetah+agent](https://github.com/user-attachments/assets/9e6832ea-58c6-407e-862e-1553d2cdbedb)

## Train loss for normal rl ##
![normal rl train result](https://github.com/user-attachments/assets/b5db2b49-9bb2-49ba-9c02-46ab311c3475)

## Train loss for transformer model ##
![transformer training result](https://github.com/user-attachments/assets/80603326-7371-4103-b01f-f7fd6ae6985d)


## Train loss for transformer + rl ##
![transformer rl](https://github.com/user-attachments/assets/b3d4777c-783c-4d93-a9fc-054f5a461843)


## difference description ##

normal rl experiment and transformer + rl experiment both uses simple actor critic method with 3 layers.
Difference is this part. if n_epi ' reward > (n_epi + 1) ' reward, then reward is changed by adding transformer model's output(=custome reward).  
![1](https://github.com/user-attachments/assets/b09a4f38-a650-4b99-895d-ec5e21e00dda)
epi_reward_sum is cumulative reward for each n_epi

## meaning of transformer model ##
from the first stage, trajectory is full of bad [state,action,reward] sequence. Because agent dose not gain good reward and show good performance.
Transformer trains with these failed dataset and gets knowledge. Then, for some trajectory that value too much on bad action sequence, transformer will signal that
these action combinations are bad. Then overestimation for q_val will be decreased. point in this project is that using failed dataset.

## online learning transformer ##
idea : for certain steps, transformer model learns from new trajectorctory.
I hope this will prevent actor network from optimistic expectation

## about files ##
first_stage , second_stage, third_stage, visualization.py files are divided version of ipynb





