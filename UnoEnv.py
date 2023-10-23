import gymnasium as gym
import numpy as np


class Uno(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(61)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1, 1), dtype=np.float32)
        self.reward_range = (-1, 1)
        self.spec = None

    def step(self, action: gym.ActType) -> tuple:
        return self.observation_space.sample(), 0, False, {}

    def reset(self) -> gym.ObsType:
        return self.observation_space.sample()

    def render(self):
        pass

    def close(self):
        pass

