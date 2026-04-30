from __future__ import annotations

import numpy as np
import torch


def evaluate_policy(
    env,
    policy,
    device: torch.device,
    num_episodes: int = 1,
    max_steps: int = 1000,
) -> np.ndarray:
    """
    Evalúa una política en un ambiente multiobjetivo.

    Regresa:
        Vector promedio de retornos multiobjetivo.
        Ejemplo para HalfCheetah:
            [retorno_objetivo_1, retorno_objetivo_2]
    """
    returns = []

    action_low = env.action_space.low
    action_high = env.action_space.high

    policy.to(device)
    policy.eval()

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_return = None

        for _step in range(max_steps):
            action = policy.act(
                obs=obs,
                action_low=action_low,
                action_high=action_high,
                device=device,
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