from __future__ import annotations

import random
import torch


def crossover(parent_a, parent_b, alpha: float | None = None):
    """
    Cruce aritmético entre dos políticas.

    child = alpha * parent_a + (1 - alpha) * parent_b
    """
    if alpha is None:
        alpha = random.random()

    child = parent_a.clone()

    state_a = parent_a.state_dict()
    state_b = parent_b.state_dict()
    state_child = child.state_dict()

    for key in state_child:
        state_child[key] = alpha * state_a[key] + (1.0 - alpha) * state_b[key]

    child.load_state_dict(state_child)
    return child


def mutate(policy, sigma: float = 0.05, mutation_rate: float = 0.2):
    """
    Mutación gaussiana sobre los parámetros de la política.
    """
    mutated = policy.clone()

    with torch.no_grad():
        for param in mutated.parameters():
            if random.random() < mutation_rate:
                param.add_(torch.randn_like(param) * sigma)

    return mutated


def make_child(
    parent_a,
    parent_b,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.2,
    sigma: float = 0.05,
):
    """
    Genera un hijo usando cruce + mutación.
    """
    if random.random() < crossover_rate:
        child = crossover(parent_a, parent_b)
    else:
        child = parent_a.clone()

    child = mutate(child, sigma=sigma, mutation_rate=mutation_rate)
    return child