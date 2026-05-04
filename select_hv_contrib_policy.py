from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from metrics.hypervolume import compute_hypervolume


def select_hv_contrib_policy(
    metrics_csv: str,
    output_txt: str | None = None,
    seed: int = 0,
) -> str:
    df = pd.read_csv(metrics_csv)

    objective_cols = [col for col in df.columns if col.startswith("objective_")]

    if not objective_cols:
        raise ValueError("No encontré columnas objective_ en el CSV.")

    objectives = df[objective_cols].to_numpy(dtype=np.float32)

    full_hv, reference_point = compute_hypervolume(
        objectives=objectives,
        reference_point=None,
        margin=1.0,
        mc_samples=100_000,
        seed=seed,
    )

    contributions = []

    for i in range(len(objectives)):
        reduced = np.delete(objectives, i, axis=0)

        reduced_hv, _ = compute_hypervolume(
            objectives=reduced,
            reference_point=reference_point,
            margin=1.0,
            mc_samples=100_000,
            seed=seed,
        )

        contributions.append(full_hv - reduced_hv)

    df["hv_contribution"] = contributions

    best_idx = int(np.argmax(contributions))
    best_row = df.iloc[best_idx]

    checkpoint_path = str(best_row["checkpoint_path"])

    print("[HV Contribution] Política con mayor contribución al hypervolume")
    print(f"policy_id: {best_row['policy_id']}")
    print(f"checkpoint_path: {checkpoint_path}")
    print(f"hv_contribution: {best_row['hv_contribution']}")
    print(f"objectives: {best_row[objective_cols].to_dict()}")

    if output_txt is not None:
        output_path = Path(output_txt)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(checkpoint_path)

        print(f"[Done] Ruta guardada en: {output_path}")

    output_csv = Path(metrics_csv).with_name("pareto_front_metrics_with_hv_contrib.csv")
    df.to_csv(output_csv, index=False)
    print(f"[Done] CSV con contribuciones guardado en: {output_csv}")

    return checkpoint_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metrics_csv", type=str, required=True)
    parser.add_argument("--output_txt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    select_hv_contrib_policy(
        metrics_csv=args.metrics_csv,
        output_txt=args.output_txt,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()