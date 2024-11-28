
def transformer_rl_build():
    class New_Agent(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.tanh = nn.Tanh()
            self.relu = nn.ReLU()

            self.actor = nn.Sequential(
                nn.Linear(config.state_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.action_dim + config.action_dim))

            self.critic = nn.Sequential(
                nn.Linear(config.state_dim + config.action_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, 1)
            )

            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr_actor)

        def update(self, model):
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            states, actions, rewards, next_states, next_actions, dones = memory.sample(AC_config.batch_size)

            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            next_actions = next_actions.to(device)
            dones = dones.to(device)

            # critic network
            input = torch.cat([states, actions], axis=1)
            next_input = torch.cat([next_states, next_actions], axis=1)

            critic = self.critic(input)
            next_critic = self.critic(next_input)

            # loss 계산
            actor_loss = -critic
            critic_loss = (critic - (rewards + AC_config.gamma * (1 - dones) * (next_critic - critic))) ** 2

            actor_loss = actor_loss.mean()
            critic_loss = critic_loss.mean()

            actor_loss.backward(retain_graph=True)
            critic_loss.backward(retain_graph=True)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

    # OpenAI Gym 환경
    env = gym.make('HalfCheetah-v4', render_mode='rgb_array')

    new_agent = New_Agent(AC_config).to(device)

    num_epis, epi_rews = 200, []
    memory = ReplayBuffer(AC_config)
    tanh = nn.Tanh()
    epi_reward_sum = [0]

    for n_epi in tqdm(range(num_epis)):
        state, _ = env.reset()
        terminated, truncated = False, False
        epi_rew = 0
        cnt = 0
        reward_sum = 0
        epi_reward_sum.append(0)
        while not (terminated or truncated):
            cnt += 1
            state = torch.Tensor(state)
            state = state.to(device)

            # actor network
            a = new_agent.actor(state)

            if len(a.shape) >= 2:
                mean = a[:, :6]
                std = a[:, 6:]
            else:
                mean = a[:6]
                std = a[6:]

            std = torch.clamp(std, min=1e-10)
            log_std = torch.log(std)
            # don't use jacobian
            normal = dist.Normal(mean, std)
            z = normal.rsample()  # reparameterization trick
            z = torch.Tensor(z)

            action = tanh(z)
            # policy = log_prob

            next_state, reward, terminated, truncated, _ = env.step(action.cpu().detach().numpy())
            next_state = torch.Tensor(next_state)
            next_state = next_state.to(device)

            na = new_agent.actor(next_state)

            if len(na.shape) >= 2:
                next_mean = na[:, :6]
                next_std = na[:, 6:]
            else:
                next_mean = na[:6]
                next_std = na[6:]

            next_std = torch.clamp(next_std, min=1e-10)
            next_log_std = torch.log(next_std)
            # don't use jacobian
            next_normal = dist.Normal(next_mean, next_std)
            next_z = next_normal.rsample()  # reparameterization trick
            next_z = torch.Tensor(next_z)

            next_action = tanh(next_z)

            ### ensemble components
            model_output = model.update(state.unsqueeze(0))  # batch_num , [action , reward]
            custome_reward = model_output[:, 6]

            next_custome_reward = model.update(next_state.unsqueeze(0))[:, 6]

            if n_epi <= 10:
                if epi_reward_sum[n_epi] != 0 and epi_reward_sum[n_epi + 1] != 0:
                    if epi_reward_sum[n_epi] >= epi_reward_sum[n_epi + 1]:
                        reward = ((0.6 * reward + 0.4 * custome_reward) * 0.5)
                    else:
                        reward = reward

            reward_sum += reward
            epi_reward_sum[n_epi] = reward_sum
            ###

            memory.put([state, action, reward, next_state, next_action, terminated or truncated])

            if memory.size() > 1000:
                for i in range(5):
                    new_agent.update(model)

                    epi_rew += reward

                    state = next_state

        epi_rews += [epi_rew]

        if n_epi % 2 == 0:
            plt.figure(figsize=(20, 10))
            plt.plot(epi_rews, label='episode returns')
            plt.legend(fontsize=20)
            plt.show()
            plt.close()

    