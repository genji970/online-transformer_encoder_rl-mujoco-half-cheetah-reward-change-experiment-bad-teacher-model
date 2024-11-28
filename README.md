# online learning transformer_rl-mujoco-half-cheetah-experiment-reward change(in progress)
This experiment follows three steps. collecting data from normal rl env. training custome transformer model with collected data trajectory. training new rl agent with trained transformer model in normal rl env

## 1) normal rl trained agent ##
![다운로드 (2)](https://github.com/user-attachments/assets/6383787a-06f0-4bf6-9d9b-6324c7b81b18)

## 2) transformer rl trained agent ##
![다운로드+(1) (1)](https://github.com/user-attachments/assets/a43c7212-a264-4c9f-a064-86606773bf5e)


## Train loss for normal rl ##
![1](https://github.com/user-attachments/assets/6b3e29f0-c8f8-435d-b80f-76a694af885b)

## Train loss for transformer model ##
![2](https://github.com/user-attachments/assets/822b1901-2048-4745-998d-fef3138ecbbc)

## Train loss for transformer + rl ##
![3](https://github.com/user-attachments/assets/e1753690-c8a8-4ce2-9542-76717a7b386c)


## difference description ##

normal rl experiment and transformer + rl experiment both uses simple actor critic method with 3 layers.
Difference is this part. reward gained from env.step(action) and custome reward from transformer model.
![스크린샷 2024-11-29 004204](https://github.com/user-attachments/assets/29359311-49be-4187-9567-a351200b5d2c)

## meaning of transformer model ##
from the first stage, trajectory is full of bad [state,action,reward] sequence. Because agent dose not gain good reward and show good performance.
Transformer trains with these failed dataset and gets knowledge. Then, for some trajectory that value too much on bad action sequence, transformer will signal that
these action combinations are bad. Then overestimation for q_val will be decreased. point in this project is that using failed dataset.

## online learning transformer + rl ##
idea : for every steps, transformer model learns from new trajectorctory.
I hope this will prevent actor network from optimistic expectation

## result of online transformer + rl

## about files ##
first_stage , second_stage, third_stage, visualization.py files are divided version of ipynb





