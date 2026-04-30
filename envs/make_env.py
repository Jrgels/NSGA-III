from __future__ import annotations

import mo_gymnasium as mo_gym


MUJOCO_ENVS = [
    "mo-halfcheetah-v5",
    "mo-hopper-v5",
    "mo-walker2d-v5",
    "mo-ant-v5",
    "mo-swimmer-v5",
    "mo-humanoid-v5",
    "mo-reacher-v5",
]


def make_env(env_id: str, render_mode: str | None = None):
    """
    Crea un ambiente de MO-Gymnasium.

    render_mode=None       -> entrenamiento/evaluación normal
    render_mode='rgb_array' -> render para guardar video
    render_mode='human'     -> visualización en pantalla
    """
    if env_id not in MUJOCO_ENVS:
        raise ValueError(
            f"Ambiente no soportado: {env_id}. "
            f"Usa uno de: {MUJOCO_ENVS}"
        )

    return mo_gym.make(env_id, render_mode=render_mode)