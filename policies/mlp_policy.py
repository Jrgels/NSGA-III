from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        action_type: str = "continuous",
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_type = action_type

        final_activation = nn.Tanh() if action_type == "continuous" else nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            final_activation,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        device: torch.device,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ):
        self.eval()

        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        raw_action = self.forward(obs_tensor).squeeze(0)

        if self.action_type == "discrete":
            return int(torch.argmax(raw_action).item())

        action = raw_action.cpu().numpy()

        scaled_action = action_low + (action + 1.0) * 0.5 * (
            action_high - action_low
        )

        return np.clip(scaled_action, action_low, action_high).astype(np.float32)

    def clone(self) -> "MLPPolicy":
        cloned = MLPPolicy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            action_type=self.action_type,
        )
        cloned.load_state_dict(self.state_dict())
        return cloned