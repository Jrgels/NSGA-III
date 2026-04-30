from __future__ import annotations

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Dominancia de Pareto para MAXIMIZACIÓN.

    a domina a b si:
    - a es >= b en todos los objetivos
    - a es > b en al menos un objetivo
    """
    return np.all(a >= b) and np.any(a > b)


def nondominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """
    Ordenamiento no dominado.

    Args:
        objectives:
            Matriz de tamaño (n_individuos, n_objetivos).
            Se asume maximización.

    Returns:
        Lista de frentes.
        Cada frente es una lista de índices.
    """
    n = objectives.shape[0]

    domination_counts = np.zeros(n, dtype=int)
    dominated_sets = [[] for _ in range(n)]

    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue

            if dominates(objectives[p], objectives[q]):
                dominated_sets[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                domination_counts[p] += 1

        if domination_counts[p] == 0:
            fronts[0].append(p)

    current_front = 0

    while fronts[current_front]:
        next_front = []

        for p in fronts[current_front]:
            for q in dominated_sets[p]:
                domination_counts[q] -= 1

                if domination_counts[q] == 0:
                    next_front.append(q)

        current_front += 1
        fronts.append(next_front)

    if not fronts[-1]:
        fronts.pop()

    return fronts


def get_nondominated_indices(objectives: np.ndarray) -> list[int]:
    """
    Devuelve solamente el primer frente no dominado.
    """
    fronts = nondominated_sort(objectives)
    return fronts[0] if fronts else []