
def visualization():
    os.environ['MUJOCO_GL'] = 'egl'
    env = gym.wrappers.RecordVideo(env, video_folder='./videos')

    state, _ = env.reset()
    terminated, truncated = False, False
    for i in range(1):
        while not (terminated or truncated):
            state = torch.Tensor(state)
            state = state.to(device)
            ###
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

            # log_prob = compute_log_prob(mean, log_std, z)

            action = tanh(z)

            action = action.cpu().detach().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)

            state = next_state

    env.close()

    Video("./videos/rl-video-episode-0.mp4", embed=True)
