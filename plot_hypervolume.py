from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_hypervolume(
    csv_path: str,
    output_path: str,
    title: str | None = None,
) -> None:
    df = pd.read_csv(csv_path)

    if "generation" not in df.columns or "hypervolume" not in df.columns:
        raise ValueError("El CSV debe tener columnas: generation, hypervolume")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(df["generation"], df["hypervolume"], marker="o")
    plt.xlabel("Generación")
    plt.ylabel("Hypervolume")
    plt.title(title or "Convergencia del Hypervolume")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[Done] Gráfica guardada en: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--title", type=str, default=None)

    args = parser.parse_args()

    plot_hypervolume(
        csv_path=args.csv,
        output_path=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()