
"""
actually trains the agent with PPO environment
"""

from stable_baselines3.common.monitor import Monitor
import time
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

import torch
import multiprocessing as mp
from pivot_ppo_no_dumb import PPOEnv
from stable_baselines3.common.vec_env import SubprocVecEnv





def make_env(dummy = True, seed = 1, n_envs = 1):
    env_id = "PPOEnv-v0" 

    def make_env_fn():
        return Monitor(PPOEnv("data/small_all_incorrect_points_removed_added_features.json"))
    if dummy:
        vec_env = DummyVecEnv([make_env_fn])
        vec_env.seed(seed=seed)
    else:
        vec_env = SubprocVecEnv([make_env_fn for _ in range(n_envs)])
        vec_env.seed(seed=seed)

    return vec_env

def train_model(vec_env, total_timesteps, name, model_load_path = None):
    if model_load_path is None:
        model = PPO(
        policy="MlpPolicy", 
        env=vec_env,
        verbose=1,
        learning_rate=3e-5,
        tensorboard_log="./tensorboard/",
        )
    else:
        print("loading_weights")
        model = PPO.load(model_load_path, env=vec_env)
        mean_reward, std_reward = evaluate_policy(model,
                                          vec_env,
                                          n_eval_episodes=10,
                                          warn=False)
        print(f"Before continuing training, reloaded policy has R={mean_reward:.1f} ± {std_reward:.1f}")

    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
    model.save(name)

import random
import numpy as np

def evaluate_with_random_seeds(model, vec_env, n_eval_episodes=1000):
    all_returns = []
    for _ in range(n_eval_episodes):

        seed = random.randint(0, 2**32 - 1)

        vec_env.seed(seed)
        obs = vec_env.reset()
        
        done = False
        total_reward = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = vec_env.step(action)
            total_reward += float(reward[0] if isinstance(reward, np.ndarray) else reward)
        
        all_returns.append(total_reward)
    
    mean_return = np.mean(all_returns)
    std_return  = np.std(all_returns)
    print(f"Mean reward: {mean_return:.2f} ± {std_return:.2f}")
    return mean_return, std_return


def evaluate_model(model_path, vec_env):
    model = PPO.load(model_path, env=vec_env)
    print(f"Model was trained for {model.num_timesteps} timesteps.")
    evaluate_with_random_seeds(model, vec_env, n_eval_episodes=100)


def live_inference_model(model_path, vec_env, seed = 69):
    model = PPO.load(model_path, env=vec_env)
    vec_env.seed(seed)  
    obs = vec_env.reset()
    num_envs = vec_env.num_envs
    done = False
    total_reward = 0
    while not done:

        action, _ = model.predict(
            obs, deterministic=False
        )

        obs, rewards, done, infos = vec_env.step(action)
        
        total_reward += rewards[0]
        raw_env = vec_env.envs[0]
        if not done:
            raw_env.render()
        else:
            print("Now done.")
        time.sleep(0.1)  

    
    print(f'We achieved a total reward of {total_reward}')
    vec_env.close()




def main():
    model_name = "testingMLP_fixed_reward_V2.zip"


    train_vec_env = make_env(dummy=True)
    train_model(train_vec_env, total_timesteps=10000000, name=model_name)
    

    eval_vec_env = make_env(dummy=True)
    evaluate_model(model_path=model_name, vec_env=eval_vec_env)
    live_inference_model(model_path=model_name, vec_env=eval_vec_env, seed=100)


if __name__ == "__main__":
    # mp.set_start_method('fork', force=True)
    main()
