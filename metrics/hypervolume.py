from __future__ import annotations

import numpy as np


def get_reference_point(
    objectives: np.ndarray,
    margin: float = 1.0,
) -> np.ndarray:
    """
    Calcula un punto de referencia automático para hypervolume.

    Como estamos MAXIMIZANDO, el reference point debe estar peor
    que todos los puntos observados.

    reference_point = mínimo observado por objetivo - margin
    """
    objectives = np.asarray(objectives, dtype=np.float32)
    return objectives.min(axis=0) - margin


def filter_nondominated(points: np.ndarray) -> np.ndarray:
    """
    Filtra puntos no dominados para maximización.
    """
    points = np.asarray(points, dtype=np.float32)
    n = points.shape[0]

    is_nondominated = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_nondominated[i]:
            continue

        for j in range(n):
            if i == j:
                continue

            dominates_j_i = np.all(points[j] >= points[i]) and np.any(points[j] > points[i])

            if dominates_j_i:
                is_nondominated[i] = False
                break

    return points[is_nondominated]


def hypervolume_2d(
    points: np.ndarray,
    reference_point: np.ndarray,
) -> float:
    """
    Hypervolume exacto para 2 objetivos en maximización.

    Asume que reference_point es peor que todos los puntos.
    """
    points = filter_nondominated(points)

    # Solo considerar puntos que dominan al reference point
    points = points[np.all(points > reference_point, axis=1)]

    if len(points) == 0:
        return 0.0

    # Ordenar por objetivo 1 ascendente
    points = points[np.argsort(points[:, 0])]

    hv = 0.0
    current_y = reference_point[1]

    for x, y in points:
        width = x - reference_point[0]
        height = y - current_y

        if height > 0:
            hv += width * height
            current_y = y

    return float(hv)


def monte_carlo_hypervolume(
    points: np.ndarray,
    reference_point: np.ndarray,
    samples: int = 100_000,
    seed: int = 0,
) -> float:
    """
    Aproximación Monte Carlo del hypervolume para 3+ objetivos.

    Útil para Hopper, Ant, Reacher, etc.
    """
    points = filter_nondominated(points)

    points = points[np.all(points > reference_point, axis=1)]

    if len(points) == 0:
        return 0.0

    upper_bound = points.max(axis=0)

    box_volume = np.prod(upper_bound - reference_point)

    if box_volume <= 0:
        return 0.0

    rng = np.random.default_rng(seed)

    samples_points = rng.uniform(
        low=reference_point,
        high=upper_bound,
        size=(samples, points.shape[1]),
    )

    dominated = np.zeros(samples, dtype=bool)

    for p in points:
        dominated |= np.all(p >= samples_points, axis=1)

    hv = box_volume * dominated.mean()

    return float(hv)


def compute_hypervolume(
    objectives: np.ndarray,
    reference_point: np.ndarray | None = None,
    margin: float = 1.0,
    mc_samples: int = 100_000,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """
    Calcula hypervolume para maximización.

    - Si hay 2 objetivos: cálculo exacto.
    - Si hay 3+ objetivos: aproximación Monte Carlo.

    Returns:
        hv, reference_point
    """
    objectives = np.asarray(objectives, dtype=np.float32)

    if reference_point is None:
        reference_point = get_reference_point(objectives, margin=margin)
    else:
        reference_point = np.asarray(reference_point, dtype=np.float32)

    num_objectives = objectives.shape[1]

    if num_objectives == 2:
        hv = hypervolume_2d(objectives, reference_point)
    else:
        hv = monte_carlo_hypervolume(
            objectives,
            reference_point,
            samples=mc_samples,
            seed=seed,
        )

    return hv, reference_point