from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """
    Política MLP determinista para espacios de acción continuos.

    Entrada:
        observación del ambiente

    Salida:
        acción continua en el rango real del ambiente
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        """
        Produce una acción en el rango válido del ambiente.

        La red produce valores en [-1, 1].
        Luego se reescalan al rango [action_low, action_high].
        """
        self.eval()

        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        action = self.forward(obs_tensor).squeeze(0).cpu().numpy()

        scaled_action = action_low + (action + 1.0) * 0.5 * (
            action_high - action_low
        )

        return np.clip(scaled_action, action_low, action_high).astype(np.float32)

    def clone(self) -> "MLPPolicy":
        """
        Crea una copia independiente de la política.
        """
        cloned = MLPPolicy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
        )
        cloned.load_state_dict(self.state_dict())
        return cloned