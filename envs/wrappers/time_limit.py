import gym


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['truncated'] = not done
            info['truncated_obs'] = observation
            done = True

        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()

    def reset_random(self):
        self._elapsed_steps = 0
        return self.env.reset_random()

    def reset_to_level(self, level, editing=False):
        self._elapsed_steps = 0
        return self.env.reset_to_level(level, editing)

    def reset_agent(self):
        self._elapsed_steps = 0
        return self.env.reset_agent()