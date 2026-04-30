from __future__ import annotations

import argparse

from envs.make_env import MUJOCO_ENVS
from training.run_nsga3 import run_nsga3


def main():
    parser = argparse.ArgumentParser(
        description="NSGA-III para Reinforcement Learning Multiobjetivo en MUJOCO."
    )

    parser.add_argument(
        "--env",
        type=str,
        default="mo-halfcheetah-v5",
        choices=MUJOCO_ENVS,
        help="Ambiente de MO-Gymnasium.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--population_size", type=int, default=16)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--reference_divisions", type=int, default=None)

    parser.add_argument("--crossover_rate", type=float, default=0.9)
    parser.add_argument("--mutation_rate", type=float, default=0.2)
    parser.add_argument("--mutation_sigma", type=float, default=0.05)

    parser.add_argument("--output_dir", type=str, default="outputs")

    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Ejecuta los 7 ambientes MUJOCO.",
    )

    args = parser.parse_args()

    if args.run_all:
        for env_id in MUJOCO_ENVS:
            print("=" * 80)
            print(f"[Run All] Ejecutando ambiente: {env_id}")
            print("=" * 80)

            run_nsga3(
                env_id=env_id,
                seed=args.seed,
                population_size=args.population_size,
                generations=args.generations,
                eval_episodes=args.eval_episodes,
                max_steps=args.max_steps,
                hidden_dim=args.hidden_dim,
                reference_divisions=args.reference_divisions,
                crossover_rate=args.crossover_rate,
                mutation_rate=args.mutation_rate,
                mutation_sigma=args.mutation_sigma,
                output_dir=args.output_dir,
            )
    else:
        run_nsga3(
            env_id=args.env,
            seed=args.seed,
            population_size=args.population_size,
            generations=args.generations,
            eval_episodes=args.eval_episodes,
            max_steps=args.max_steps,
            hidden_dim=args.hidden_dim,
            reference_divisions=args.reference_divisions,
            crossover_rate=args.crossover_rate,
            mutation_rate=args.mutation_rate,
            mutation_sigma=args.mutation_sigma,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()