from __future__ import annotations

import itertools
import numpy as np


def generate_reference_points(num_objectives: int, divisions: int) -> np.ndarray:
    """
    Genera puntos de referencia tipo Das-Dennis sobre el simplex.

    Para 2 objetivos y divisions=4:
        [0.0, 1.0]
        [0.25, 0.75]
        [0.5, 0.5]
        [0.75, 0.25]
        [1.0, 0.0]
    """
    if num_objectives < 2:
        raise ValueError("num_objectives debe ser >= 2.")

    if divisions < 1:
        raise ValueError("divisions debe ser >= 1.")

    points = []

    for comb in itertools.combinations_with_replacement(
        range(num_objectives), divisions
    ):
        counts = np.zeros(num_objectives, dtype=np.float32)

        for idx in comb:
            counts[idx] += 1.0

        points.append(counts / divisions)

    reference_points = np.asarray(points, dtype=np.float32)

    # Quitar duplicados por seguridad
    reference_points = np.unique(reference_points, axis=0)

    return reference_points


def normalize_objectives(objectives: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normaliza objetivos para maximización al rango [0, 1].

    Args:
        objectives: matriz (n_individuos, n_objetivos)

    Returns:
        matriz normalizada del mismo tamaño.
    """
    objectives = np.asarray(objectives, dtype=np.float32)

    min_values = objectives.min(axis=0)
    max_values = objectives.max(axis=0)

    return (objectives - min_values) / (max_values - min_values + eps)


def perpendicular_distance(point: np.ndarray, direction: np.ndarray) -> float:
    """
    Distancia perpendicular de un punto a una línea definida por direction.

    La línea pasa por el origen y apunta hacia direction.
    """
    point = np.asarray(point, dtype=np.float32)
    direction = np.asarray(direction, dtype=np.float32)

    norm = np.linalg.norm(direction)

    if norm < 1e-8:
        return float(np.linalg.norm(point))

    direction_unit = direction / norm
    projection = np.dot(point, direction_unit) * direction_unit

    return float(np.linalg.norm(point - projection))


def associate_to_reference_points(
    normalized_objectives: np.ndarray,
    reference_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Asocia cada individuo al punto de referencia más cercano.

    Returns:
        associations:
            índice del punto de referencia asociado a cada individuo.

        distances:
            distancia perpendicular hacia su punto de referencia asociado.
    """
    n = normalized_objectives.shape[0]

    associations = np.zeros(n, dtype=int)
    distances = np.zeros(n, dtype=np.float32)

    for i in range(n):
        dists = [
            perpendicular_distance(normalized_objectives[i], ref)
            for ref in reference_points
        ]

        best_ref = int(np.argmin(dists))
        associations[i] = best_ref
        distances[i] = dists[best_ref]

    return associations, distances