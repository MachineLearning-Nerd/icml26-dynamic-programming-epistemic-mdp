from __future__ import annotations

import numpy as np

from src.mdp import correlated_action_rows_control, rectangular_binary_mdp, risk, static_expectation_counterexample
from src.spectral import independent_multiplicativity_residual, static_bellman_residual


def test_discrete_risk_basics() -> None:
    values = np.array([-2.0, 1.0, 5.0])
    probabilities = np.array([0.2, 0.3, 0.5])
    assert risk(values, probabilities, "min") == -2.0
    assert risk(values, probabilities, "max") == 5.0
    assert np.isclose(risk(values, probabilities, "mean"), 2.4)
    assert risk(values, probabilities, "cvar") < risk(values, probabilities, "mean")


def test_rectangular_support_has_all_statewise_combinations() -> None:
    mdp = rectangular_binary_mdp(3, states=3)
    assert mdp.kernels.shape[0] == 2 ** (3 * 2)
    assert np.allclose(mdp.kernels.sum(axis=-1), 1.0)


def test_robust_fixed_point_matches_static_value() -> None:
    mdp = rectangular_binary_mdp(4)
    value, policy, _ = mdp.optimal_value("min")
    assert np.max(np.abs(value - mdp.static_value(policy, "min"))) < 1e-10


def test_optimistic_fixed_point_matches_static_value() -> None:
    mdp = rectangular_binary_mdp(5)
    value, policy, _ = mdp.optimal_value("max")
    assert np.max(np.abs(value - mdp.static_value(policy, "max"))) < 1e-10


def test_bellman_policy_is_exhaustive_stationary_optimum() -> None:
    mdp = rectangular_binary_mdp(6)
    for name in ("min", "max"):
        value, policy, _ = mdp.optimal_value(name)
        _, all_values = mdp.exhaustive_stationary(name)
        assert np.max(np.abs(value - all_values.max(axis=0))) < 1e-10
        assert np.max(np.abs(value - mdp.static_value(policy, name))) < 1e-10


def test_history_dependent_prefixes_do_not_improve_robust_or_optimistic_value() -> None:
    mdp = rectangular_binary_mdp(8, states=2)
    for name in ("min", "max"):
        value, _, _ = mdp.optimal_value(name)
        for start in range(mdp.states):
            history_value, count = mdp.exhaustive_history_prefix(start, horizon=3, name=name)
            assert count == 512
            assert abs(history_value - value[start]) < 1e-10


def test_non_product_support_control_exposes_history_advantage() -> None:
    mdp = correlated_action_rows_control(1)
    advantages = []
    for name in ("min", "max"):
        value, _, _ = mdp.optimal_value(name)
        for start in range(mdp.states):
            history_value, _ = mdp.exhaustive_history_prefix(start, horizon=3, name=name)
            advantages.append(history_value - value[start])
    assert max(advantages) > 1e-3


def test_static_expectation_breaks_the_bellman_equation() -> None:
    mdp = static_expectation_counterexample()
    policy = np.zeros(mdp.states, dtype=int)
    value = mdp.static_value(policy, "mean")
    residual = np.max(np.abs(value - mdp.bellman_policy(value, policy, "mean")))
    assert residual > 1e-3


def test_resampled_expectation_matches_expected_kernel_solution() -> None:
    mdp = static_expectation_counterexample()
    policy = np.zeros(mdp.states, dtype=int)
    expected_kernel = np.einsum("k,ksan->san", mdp.probabilities, mdp.kernels)
    transition = expected_kernel[:, 0, :]
    reward = np.sum(transition * mdp.rewards[:, 0, :], axis=1)
    reference = np.linalg.solve(np.eye(mdp.states) - mdp.gamma * transition, reward)
    result = mdp.fixed_policy_value(policy, "mean")
    assert np.max(np.abs(reference - result)) < 1e-10


def test_cvar_fails_independent_additivity() -> None:
    x = np.array([-2.0, 4.0])
    y = np.array([-1.0, 5.0])
    p = np.array([0.5, 0.5])
    xy = (x[:, None] + y[None, :]).reshape(-1)
    pxy = (p[:, None] * p[None, :]).reshape(-1)
    residual = risk(xy, pxy, "cvar") - risk(x, p, "cvar") - risk(y, p, "cvar")
    assert abs(residual) > 0.1


def test_spectral_static_control_keeps_only_essential_extrema() -> None:
    mdp = rectangular_binary_mdp(4, states=2, actions=1)
    assert static_bellman_residual(mdp, np.array([1.0, 0.0, 0.0, 0.0])) < 1e-10
    assert static_bellman_residual(mdp, np.array([0.0, 0.0, 0.0, 1.0])) < 1e-10
    assert static_bellman_residual(mdp, np.full(4, 0.25)) > 1e-3


def test_spectral_multiplicativity_removes_nontrivial_extrema_mixtures() -> None:
    x = np.array([0.2, 3.0])
    z = np.array([0.4, 5.0])
    assert abs(independent_multiplicativity_residual(np.full(4, 0.25), x, z)) < 1e-10
    assert abs(independent_multiplicativity_residual(np.array([0.5, 0.0, 0.0, 0.5]), x, z)) > 1e-3
