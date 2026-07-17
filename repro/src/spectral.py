"""Finite spectral-risk classification controls for Theorems 4.4--4.5."""

from __future__ import annotations

from itertools import product

import numpy as np

from src.mdp import FiniteAmbiguityMDP


def integer_simplex(total: int, width: int) -> list[tuple[int, ...]]:
    """All non-negative integer vectors of fixed sum, deterministically."""
    if width == 1:
        return [(total,)]
    result = []
    for first in range(total + 1):
        for rest in integer_simplex(total - first, width - 1):
            result.append((first,) + rest)
    return result


def ordered_risk(values: np.ndarray, weights: np.ndarray) -> float:
    """A law-invariant monotone ordered weighted average on uniform atoms."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.shape != weights.shape:
        raise ValueError("aligned atom values and weights required")
    if not np.isclose(weights.sum(), 1.0) or np.any(weights < 0.0):
        raise ValueError("weights must define a probability vector")
    return float(np.dot(np.sort(values), weights))


def static_value(mdp: FiniteAmbiguityMDP, policy: np.ndarray, weights: np.ndarray) -> np.ndarray:
    nominal = mdp.nominal_values(policy)
    return np.asarray([ordered_risk(nominal[:, state], weights) for state in range(mdp.states)])


def bellman_policy(mdp: FiniteAmbiguityMDP, value: np.ndarray, policy: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.empty(mdp.states)
    for state, action in enumerate(policy):
        atoms = np.einsum(
            "kn,n->k",
            mdp.kernels[:, state, action, :],
            mdp.rewards[state, action, :] + mdp.gamma * value,
        )
        result[state] = ordered_risk(atoms, weights)
    return result


def static_bellman_residual(mdp: FiniteAmbiguityMDP, weights: np.ndarray) -> float:
    """Maximum exact Condition-1 residual over all stationary policies."""
    residual = 0.0
    for policy_tuple in product(range(mdp.actions), repeat=mdp.states):
        policy = np.asarray(policy_tuple, dtype=int)
        value = static_value(mdp, policy, weights)
        residual = max(residual, float(np.max(np.abs(value - bellman_policy(mdp, value, policy, weights)))))
    return residual


def independent_additivity_residual(weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Exact required Proposition-4.3 identity on two independent variables."""
    joint = (np.asarray(x)[:, None] + np.asarray(y)[None, :]).reshape(-1)
    return ordered_risk(joint, weights) - ordered_risk(np.repeat(x, len(y)), weights) - ordered_risk(np.tile(y, len(x)), weights)


def independent_multiplicativity_residual(weights: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
    """The non-negative independent product identity used in the proof."""
    joint = (np.asarray(x)[:, None] * np.asarray(z)[None, :]).reshape(-1)
    return ordered_risk(joint, weights) - ordered_risk(np.repeat(x, len(z)), weights) * ordered_risk(np.tile(z, len(x)), weights)
