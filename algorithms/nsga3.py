from __future__ import annotations

import random
import numpy as np

from algorithms.nondominated_sort import nondominated_sort
from algorithms.reference_points import (
    normalize_objectives,
    associate_to_reference_points,
)


def nsga3_select(
    population: list,
    objectives: np.ndarray,
    population_size: int,
    reference_points: np.ndarray,
) -> tuple[list, np.ndarray]:
    """
    Selección NSGA-III para maximización.

    Args:
        population:
            Lista de políticas.

        objectives:
            Matriz (n_individuos, n_objetivos).
            Cada fila es el vector de retorno de una política.

        population_size:
            Tamaño deseado de la siguiente generación.

        reference_points:
            Puntos de referencia sobre el simplex.

    Returns:
        selected_population:
            Lista de políticas seleccionadas.

        selected_objectives:
            Objetivos correspondientes a las políticas seleccionadas.
    """
    objectives = np.asarray(objectives, dtype=np.float32)

    fronts = nondominated_sort(objectives)

    selected_indices: list[int] = []
    last_front: list[int] | None = None

    for front in fronts:
        if len(selected_indices) + len(front) <= population_size:
            selected_indices.extend(front)
        else:
            last_front = front
            break

    if len(selected_indices) == population_size:
        return (
            [population[i] for i in selected_indices],
            objectives[selected_indices],
        )

    if last_front is None:
        return (
            [population[i] for i in selected_indices],
            objectives[selected_indices],
        )

    remaining_slots = population_size - len(selected_indices)

    all_candidate_indices = selected_indices + last_front
    candidate_objectives = objectives[all_candidate_indices]

    normalized_candidate_objectives = normalize_objectives(candidate_objectives)

    associations, distances = associate_to_reference_points(
        normalized_candidate_objectives,
        reference_points,
    )

    selected_set = set(selected_indices)

    niche_counts = np.zeros(len(reference_points), dtype=int)

    # Contar nichos usando los individuos ya seleccionados
    for local_idx, global_idx in enumerate(all_candidate_indices):
        if global_idx in selected_set:
            ref_idx = associations[local_idx]
            niche_counts[ref_idx] += 1

    last_front_set = set(last_front)

    # Mapear global_idx -> local_idx
    global_to_local = {
        global_idx: local_idx
        for local_idx, global_idx in enumerate(all_candidate_indices)
    }

    while remaining_slots > 0 and last_front_set:
        min_niche_count = niche_counts.min()
        candidate_refs = np.where(niche_counts == min_niche_count)[0].tolist()
        random.shuffle(candidate_refs)

        chosen = False

        for ref_idx in candidate_refs:
            associated_last_front = [
                idx
                for idx in last_front_set
                if associations[global_to_local[idx]] == ref_idx
            ]

            if not associated_last_front:
                continue

            if niche_counts[ref_idx] == 0:
                # Si el nicho está vacío, elegir el más cercano al punto de referencia
                chosen_idx = min(
                    associated_last_front,
                    key=lambda idx: distances[global_to_local[idx]],
                )
            else:
                # Si ya hay individuos en ese nicho, elegir uno al azar
                chosen_idx = random.choice(associated_last_front)

            selected_indices.append(chosen_idx)
            selected_set.add(chosen_idx)
            last_front_set.remove(chosen_idx)

            niche_counts[ref_idx] += 1
            remaining_slots -= 1
            chosen = True
            break

        if not chosen:
            # Si ninguna referencia tiene individuos disponibles, rellena al azar
            chosen_idx = random.choice(list(last_front_set))
            selected_indices.append(chosen_idx)
            selected_set.add(chosen_idx)
            last_front_set.remove(chosen_idx)
            remaining_slots -= 1

    return (
        [population[i] for i in selected_indices],
        objectives[selected_indices],
    )