from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from algorithms.nondominated_sort import get_nondominated_indices
from algorithms.nsga3 import nsga3_select
from algorithms.reference_points import generate_reference_points
from algorithms.variation import make_child
from envs.make_env import make_env
from metrics.hypervolume import compute_hypervolume
from policies.mlp_policy import MLPPolicy
from training.evaluate_policy import evaluate_policy
from utils.device import get_device, print_device_info
from utils.seed import set_seed


def save_policy(policy: MLPPolicy, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": policy.state_dict(),
            "obs_dim": policy.obs_dim,
            "action_dim": policy.action_dim,
            "hidden_dim": policy.hidden_dim,
        },
        path,
    )


def select_representative_indices(objectives: np.ndarray) -> dict[str, int]:
    """
    Selecciona 3 políticas representativas del frente:
    - mejor objetivo 1
    - mejor objetivo 2
    - solución balanceada
    """
    normalized = (objectives - objectives.min(axis=0)) / (
        objectives.max(axis=0) - objectives.min(axis=0) + 1e-8
    )

    best_obj1 = int(np.argmax(objectives[:, 0]))
    best_obj2 = int(np.argmax(objectives[:, 1]))

    target = np.ones(objectives.shape[1], dtype=np.float32) * 0.5
    balanced = int(np.argmin(np.linalg.norm(normalized - target, axis=1)))

    return {
        "best_obj1": best_obj1,
        "best_obj2": best_obj2,
        "balanced": balanced,
    }


def run_nsga3(
    env_id: str = "mo-halfcheetah-v5",
    seed: int = 0,
    population_size: int = 16,
    generations: int = 10,
    eval_episodes: int = 1,
    max_steps: int = 300,
    hidden_dim: int = 64,
    reference_divisions: int | None = None,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.2,
    mutation_sigma: float = 0.05,
    output_dir: str = "outputs",
) -> None:
    set_seed(seed)

    device = get_device()
    print_device_info(device)

    print(f"[Run] env_id={env_id}")
    print(f"[Run] seed={seed}")
    print(f"[Run] population_size={population_size}")
    print(f"[Run] generations={generations}")
    print(f"[Run] eval_episodes={eval_episodes}")
    print(f"[Run] max_steps={max_steps}")

    env = make_env(env_id)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    obs, info = env.reset(seed=seed)
    _, reward, _, _, _ = env.step(env.action_space.sample())
    reward_dim = len(np.asarray(reward))

    print(f"[Env] obs_dim={obs_dim}")
    print(f"[Env] action_dim={action_dim}")
    print(f"[Env] reward_dim={reward_dim}")

    if reference_divisions is None:
        reference_divisions = max(population_size - 1, 1)

    reference_points = generate_reference_points(
        num_objectives=reward_dim,
        divisions=reference_divisions,
    )

    print(f"[NSGA-III] reference_points={len(reference_points)}")

    base_output = Path(output_dir) / env_id / f"seed_{seed}"
    metrics_dir = base_output / "metrics"
    checkpoints_dir = base_output / "checkpoints"
    pareto_dir = checkpoints_dir / "pareto_front"
    representative_dir = checkpoints_dir / "representative"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    pareto_dir.mkdir(parents=True, exist_ok=True)
    representative_dir.mkdir(parents=True, exist_ok=True)

    population = [
        MLPPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        for _ in range(population_size)
    ]

    objectives = np.stack(
        [
            evaluate_policy(
                env=env,
                policy=policy,
                device=device,
                num_episodes=eval_episodes,
                max_steps=max_steps,
            )
            for policy in population
        ]
    )

    history = []
    reference_point = None

    for generation in range(generations):
        children = []

        while len(children) < population_size:
            parent_a, parent_b = random.sample(population, 2)

            child = make_child(
                parent_a=parent_a,
                parent_b=parent_b,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                sigma=mutation_sigma,
            ).to(device)

            children.append(child)

        children_objectives = np.stack(
            [
                evaluate_policy(
                    env=env,
                    policy=child,
                    device=device,
                    num_episodes=eval_episodes,
                    max_steps=max_steps,
                )
                for child in children
            ]
        )

        combined_population = population + children
        combined_objectives = np.vstack([objectives, children_objectives])

        population, objectives = nsga3_select(
            population=combined_population,
            objectives=combined_objectives,
            population_size=population_size,
            reference_points=reference_points,
        )

        hv, reference_point = compute_hypervolume(
            objectives=objectives,
            reference_point=reference_point,
            margin=1.0,
            mc_samples=50_000,
            seed=seed + generation,
        )

        nondominated_indices = get_nondominated_indices(objectives)

        row = {
            "generation": generation,
            "hypervolume": hv,
            "nondominated_count": len(nondominated_indices),
        }

        for i in range(reward_dim):
            row[f"best_obj_{i}"] = float(np.max(objectives[:, i]))
            row[f"mean_obj_{i}"] = float(np.mean(objectives[:, i]))
            row[f"std_obj_{i}"] = float(np.std(objectives[:, i]))

        history.append(row)

        print(
            f"[Gen {generation:03d}] "
            f"HV={hv:.4f} | "
            f"ND={len(nondominated_indices)} | "
            f"best={np.max(objectives, axis=0)}"
        )

        pd.DataFrame(history).to_csv(
            metrics_dir / "hypervolume_by_generation.csv",
            index=False,
        )

    final_nondominated_indices = get_nondominated_indices(objectives)

    pareto_records = []

    for local_id, idx in enumerate(final_nondominated_indices):
        policy = population[idx]
        policy_objectives = objectives[idx]

        checkpoint_path = pareto_dir / f"policy_{local_id:03d}.pt"
        save_policy(policy, checkpoint_path)

        record = {
            "policy_id": local_id,
            "checkpoint_path": str(checkpoint_path),
        }

        for j in range(reward_dim):
            record[f"objective_{j}"] = float(policy_objectives[j])

        pareto_records.append(record)

    pareto_df = pd.DataFrame(pareto_records)
    pareto_df.to_csv(metrics_dir / "pareto_front_metrics.csv", index=False)

    pareto_objectives = objectives[final_nondominated_indices]

    representatives = select_representative_indices(pareto_objectives)

    representative_metrics = {}

    for name, pareto_local_idx in representatives.items():
        global_idx = final_nondominated_indices[pareto_local_idx]
        policy = population[global_idx]
        policy_objectives = objectives[global_idx]

        checkpoint_path = representative_dir / f"{name}.pt"
        save_policy(policy, checkpoint_path)

        representative_metrics[name] = {
            "checkpoint_path": str(checkpoint_path),
            "pareto_policy_id": int(pareto_local_idx),
            "objectives": policy_objectives.tolist(),
        }

    final_hv, reference_point = compute_hypervolume(
        objectives=objectives,
        reference_point=reference_point,
        margin=1.0,
        mc_samples=100_000,
        seed=seed,
    )

    summary = {
        "env_id": env_id,
        "seed": seed,
        "population_size": population_size,
        "generations": generations,
        "eval_episodes": eval_episodes,
        "max_steps": max_steps,
        "reward_dim": reward_dim,
        "final_hypervolume": final_hv,
        "reference_point": reference_point.tolist(),
        "nondominated_count": len(final_nondominated_indices),
        "representatives": representative_metrics,
    }

    with open(metrics_dir / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    env.close()

    print("[Done] Entrenamiento terminado.")
    print(f"[Done] Métricas guardadas en: {metrics_dir}")
    print(f"[Done] Frente Pareto guardado en: {pareto_dir}")
    print(f"[Done] Representantes guardados en: {representative_dir}")