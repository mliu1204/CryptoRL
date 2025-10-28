import gymnasium
from gymnasium import spaces
import numpy as np
import random
from collections import deque

# model needs to detect if there is a 1 in the past 5 digits
class BinaryHistoryEnv(gymnasium.Env):
    def __init__(self, max_steps: int = 100):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8)

        self.max_steps = max_steps
        self.cur_step = None

        # binary observation the model sees
        self.current_obs = None

        # used to calculate reward and render, never passed to model
        self.history = deque(maxlen=5)
        self.last_reward = None
        self.last_action = None

    def seed(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return [seed]

    def reset(self, *, seed: int = None):
        self.seed(seed)
        self.cur_step = 0

        self.history.clear()
        for _ in range(5):
            self.history.append(0)

        self.current_obs = np.random.randint(0, 2)
        self.history.append(int(self.current_obs))

        self.last_reward = None
        self.last_action = None

        return np.array([self.current_obs], dtype=np.int8), {}

    def step(self, action: int):
        self.last_action = action

        saw_one_in_last_five = any(self.history)
        if action == 1:
            reward = 0.5 if saw_one_in_last_five else -1
        else:  
            reward = -0.2

        self.last_reward = reward

        next_obs = 1 if np.random.rand() < 0.1 else 0
        self.current_obs = next_obs
        self.history.append(int(next_obs))

        # Increment step counter and check termination
        self.cur_step += 1
        terminated = self.cur_step >= self.max_steps
        truncated = False

        return np.array([self.current_obs], dtype=np.int8), reward, terminated, truncated, {}


    def render(self, mode="human"):
        hist_list = list(self.history)
        reward_str = (
            f"{self.last_reward:.2f}" if self.last_reward is not None else "N/A"
        )
        action_str = (
            str(self.last_action) if self.last_action is not None else "N/A"
        )

        print(
            f"Step: {self.cur_step:3d} | "
            f"History(last 5): {hist_list} | "
            f"Current Obs: {self.current_obs} | "
            f"Last Action: {action_str} | "
            f"Last Reward: {reward_str}"
        )

    def close(self):
        pass
