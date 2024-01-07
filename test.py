import gymnasium as gym
import torch
from gymnasium.spaces import flatdim

from network import Network

if __name__ == "__main__":
    env_name = "BipedalWalker-v3"
    extras = "hardcore"
    model_type = "ppo"
    model_name = "ppo_actor"
    model_layers = [256, 256, 256]
    allowed_time_steps = 2000

    env = eval(f"gym.make(env_name, {extras + '=True,' if extras else ''} render_mode='human')")

    obs_dims = flatdim(env.observation_space)
    act_dims = flatdim(env.action_space)

    model_path = f"models/{model_type}/{env_name}/{model_name}.pth"

    policy = Network(obs_dims, act_dims, model_layers)
    policy.load_state_dict(torch.load(model_path))

    while True:
        obs, _ = env.reset()

        for _ in range(allowed_time_steps):
            action = policy(obs).detach().numpy()
            obs, reward, terminated, _, _ = env.step(action)

            if terminated:
                break
