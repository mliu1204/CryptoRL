import gymnasium
from gymnasium import spaces
import json
import numpy as np
import time
import random

class SimpleEnv(gymnasium.Env):
  metadata = {'render.modes': ['human']}
  LEFT = 0
  RIGHT = 1
  STOP = 2

  def __init__(self, road_size = 10):
    super(SimpleEnv, self).__init__()
    self.road_size = road_size
    self.max_time = 50
    self.cur_time = 0
    self.robot_position = road_size + 1 # very right

    self.action_space = spaces.Discrete(3)

    self.observation_space = spaces.Box(low=0, high= road_size,
                                        shape=(1,), dtype=np.float32)

  def step(self, action):
    if action == self.LEFT:
      self.robot_position -= 1
    elif action == self.RIGHT:
      self.robot_position += 1
    elif action == self.STOP:
      self.robot_position += 0
    else:
      raise ValueError("Received invalid action={action} which is not part of the action space".format(action))
    self.robot_position = np.clip(self.robot_position, 0, self.road_size-1)
    reward = 1 if bool(self.robot_position == 0) else 0
    info = {}
    observation = np.array([self.robot_position], dtype=np.float32)
    terminated = bool(self.robot_position == 0)
    truncated = bool(self.cur_time == self.max_time)
    self.cur_time += 1
    return observation, reward, terminated, truncated, info
  def reset(self,seed = 1):
    np.random.seed(seed)
    self.robot_position = np.random.randint(0, self.road_size)
    observation = np.array([self.robot_position],dtype=np.float32)
    info = {}
    return (observation, info) 
  def render(self, mode='human'):
    print("." * self.robot_position, end="")
    print("x", end="")
    print("." * (self.road_size - self.robot_position))
  def close (self):
    pass
