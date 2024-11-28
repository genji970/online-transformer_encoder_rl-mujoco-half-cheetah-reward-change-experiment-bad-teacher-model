# dataset

def reinforcement_learning():
    columns = ["state", "action", "reward", "next_state", "next_action" , "done"]
    df = pd.DataFrame(columns = columns)

    # OpenAI Gym 환경
    env = gym.make('HalfCheetah-v4', render_mode = 'rgb_array')

    #config
    AC_config = OmegaConf.create({
        "buffer_limit": int(1e6),
        # RL parameter
        'gamma': 0.99,
        'batch_size': 512,

        # neural network parameters
        'device': 'cpu',
        'hidden_dim': 256,
        'state_dim': env.observation_space.shape[0],
        'action_dim': int(env.action_space.shape[0]),  # cannot use .n because not actions are continuous!

        # learning parameters
        'lr_actor': 0.0003,
    })

    #replay buffer class
    class ReplayBuffer():
        def __init__(self, config):
            self.config = config
            self.buffer = collections.deque(maxlen=self.config.buffer_limit)

        def put(self, transition):
            self.buffer.append(transition)

        def sample(self, n):
            mini_batch = random.sample(self.buffer, n)
            s_lst, a_lst, r_lst, next_s_lst, next_a_lst, done_mask_lst = [], [], [], [], [], []

            for transition in mini_batch:
                s, a, r, next_s, next_a, done = transition
                s_lst.append(s.tolist())
                a_lst.append(a.tolist())
                r_lst.append([r])
                next_s_lst.append(next_s.tolist())
                next_a_lst.append(next_a.tolist())
                done_mask = 0.0 if done else 1.0
                done_mask_lst.append([done_mask])

            return torch.Tensor(s_lst), torch.Tensor(a_lst), torch.Tensor(r_lst), \
                   torch.Tensor(next_s_lst), torch.Tensor(next_a_lst), torch.Tensor(done_mask_lst)

        def size(self):
            return len(self.buffer)

    #network build
    class Agent(nn.Module):
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

        def update(self):
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            states, actions, rewards, next_states, next_actions, dones = memory.sample(AC_config.batch_size)

            # critic network
            input = torch.cat([states, actions], axis=1)
            next_input = torch.cat([next_states, next_actions], axis=1)

            critic = self.critic(input)
            next_critic = self.critic(next_input)

            # loss 계산
            actor_loss = -self.critic(input)
            critic_loss = (self.critic(input) - (
                        rewards + AC_config.gamma * (1 - dones) * (self.critic(next_input) - self.critic(input)))) ** 2

            actor_loss = actor_loss.mean()
            critic_loss = critic_loss.mean()

            actor_loss.backward(retain_graph=True)
            critic_loss.backward(retain_graph=True)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

    # training loop
    agent = Agent(AC_config)

    num_epis, epi_rews = 100, []
    memory = ReplayBuffer(AC_config)
    tanh = nn.Tanh()

    for n_epi in tqdm(range(num_epis)):
        state, _ = env.reset()
        terminated, truncated = False, False
        epi_rew = 0
        cnt = 0

        while not (terminated or truncated):
            cnt += 1
            state = torch.Tensor(state)

            # actor network
            a = agent.actor(state)

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

            log_prob = compute_log_prob(mean, log_std, z)

            action = tanh(z)
            policy = log_prob

            # 환경에서 다음 상태 및 보상 얻기
            action = action.detach().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = torch.Tensor(next_state)

            ### 데이터 저장
            # transition 데이터 한 행으로 저장
            transition = {
                "state": torch.Tensor(state.tolist()),  # state를 리스트로 변환
                "action": torch.Tensor(action.tolist()),  # action도 리스트로 변환
                "reward": torch.Tensor([reward]),  # scalar 값
                "next_state": torch.Tensor(next_state.tolist()),
                "next_action": torch.Tensor(action.tolist()),
                "done": torch.Tensor([terminated or truncated])  # boolean 값
            }

            # DataFrame에 추가
            df = pd.concat([df, pd.DataFrame([transition])], ignore_index=True)
            ###

            if len(agent.actor(next_state).shape) >= 2:
                next_mean = agent.actor(next_state)[:, :6]
                next_std = agent.actor(next_state)[:, 6:]
            else:
                next_mean = agent.actor(next_state)[:6]
                next_std = agent.actor(next_state)[6:]

            next_std = torch.clamp(next_std, min=1e-10)
            next_log_std = torch.log(next_std)
            # don't use jacobian
            next_normal = dist.Normal(next_mean, next_std)
            next_z = next_normal.rsample()  # reparameterization trick
            next_z = torch.Tensor(next_z)

            next_log_prob = compute_log_prob(next_mean, next_log_std, next_z)

            next_action = tanh(next_z)
            next_policy = next_log_prob

            memory.put([state, action, reward, next_state, next_action, terminated or truncated])

            if memory.size() > 1000:
                for i in range(2):
                    agent.update()

                    epi_rew += reward

                    state = next_state

        epi_rews += [epi_rew]

        if n_epi % 2 == 0:
            plt.figure(figsize=(20, 10))
            plt.plot(epi_rews, label='episode returns')
            plt.legend(fontsize=20)
            plt.show()
            plt.close()

    