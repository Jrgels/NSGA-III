from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from envs.make_env import make_env
from policies.mlp_policy import MLPPolicy
from utils.device import get_device, print_device_info
from utils.seed import set_seed


def load_policy(checkpoint_path: str, device: torch.device) -> MLPPolicy:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    policy = MLPPolicy(
        obs_dim=checkpoint["obs_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )

    policy.load_state_dict(checkpoint["state_dict"])
    policy.to(device)
    policy.eval()

    return policy


def render_policy(
    env_id: str,
    checkpoint_path: str,
    output_path: str,
    seed: int = 0,
    max_steps: int = 1000,
    fps: int = 30,
) -> None:
    set_seed(seed)

    device = get_device()
    print_device_info(device)

    env = make_env(env_id, render_mode="rgb_array")
    policy = load_policy(checkpoint_path, device)

    obs, info = env.reset(seed=seed)

    action_low = env.action_space.low
    action_high = env.action_space.high

    frames = []
    episode_return = None

    for step in range(max_steps):
        frame = env.render()
        frames.append(frame)

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

    env.close()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimsave(output_path, frames, fps=fps)

    print(f"[Done] Video guardado en: {output_path}")
    print(f"[Done] Return vector: {episode_return.tolist()}")
    print(f"[Done] Steps: {len(frames)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)

    args = parser.parse_args()

    render_policy(
        env_id=args.env,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        seed=args.seed,
        max_steps=args.max_steps,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()