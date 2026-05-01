from __future__ import annotations

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete


def evaluate_policy(
    env,
    policy,
    device: torch.device,
    num_episodes: int = 1,
    max_steps: int = 1000,
) -> np.ndarray:
    returns = []

    is_continuous = isinstance(env.action_space, Box)
    is_discrete = isinstance(env.action_space, Discrete)

    if not (is_continuous or is_discrete):
        raise ValueError(f"Action space no soportado: {env.action_space}")

    action_low = env.action_space.low if is_continuous else None
    action_high = env.action_space.high if is_continuous else None

    policy.to(device)
    policy.eval()

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_return = None

        for _step in range(max_steps):
            action = policy.act(
                obs=obs,
                device=device,
                action_low=action_low,
                action_high=action_high,
            )

            obs, reward, terminated, truncated, info = env.step(action)

            reward = np.asarray(reward, dtype=np.float32)

            if episode_return is None:
                episode_return = np.zeros_like(reward, dtype=np.float32)

            episode_return += reward

            if terminated or truncated:
                break

        returns.append(episode_return)

    return np.mean(np.stack(returns), axis=0)