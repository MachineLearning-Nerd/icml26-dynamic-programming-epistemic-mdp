"""Execute the full finite-state claim audit and write portable JSON evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.mdp import correlated_action_rows_control, rectangular_binary_mdp, risk, static_expectation_counterexample
from src.spectral import (
    independent_additivity_residual,
    independent_multiplicativity_residual,
    integer_simplex,
    static_bellman_residual,
)


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def main() -> None:
    rows: list[dict[str, object]] = []
    stationary_errors: list[float] = []
    history_errors: list[float] = []
    history_policies = 0
    vi_bounds: list[float] = []
    pi_monotone = True

    for risk_name in ("min", "max"):
        for states, seeds in ((2, range(6)), (3, range(8)), (4, range(4))):
            for seed in seeds:
                mdp = rectangular_binary_mdp(seed, states=states)
                optimal_value, bellman_policy, vi_errors = mdp.optimal_value(risk_name)
                policies, exhaustive = mdp.exhaustive_stationary(risk_name)
                bellman_static = mdp.static_value(bellman_policy, risk_name)
                stationary_errors.append(max_abs(optimal_value, bellman_static))
                stationary_errors.append(max_abs(bellman_static, exhaustive.max(axis=0)))

            # Proposition 4.2: the post-update greedy-policy error obeys its
            # stated 2 gamma/(1-gamma) value-iteration bound.
                initial_policy = np.zeros(mdp.states, dtype=int)
                initial_value = mdp.fixed_policy_value(initial_policy, risk_name)
                prior = initial_value.copy()
                factor = 2.0 * mdp.gamma / (1.0 - mdp.gamma)
                for _ in range(8):
                    next_value, greedy = mdp.bellman_optimal(prior, risk_name)
                    greedy_value = mdp.fixed_policy_value(greedy, risk_name)
                    vi_bounds.append(float(np.max(np.abs(greedy_value - optimal_value)) - factor * np.max(np.abs(prior - optimal_value))))
                    prior = next_value

            # Exact policy iteration with independently solved policy fixed points.
                policy = initial_policy.copy()
                previous = mdp.fixed_policy_value(policy, risk_name)
                for _ in range(12):
                    _, improved = mdp.bellman_optimal(previous, risk_name)
                    candidate = mdp.fixed_policy_value(improved, risk_name)
                    pi_monotone &= bool(np.all(candidate >= previous - 1e-11))
                    if np.array_equal(improved, policy):
                        break
                    policy, previous = improved, candidate
                rows.append({
                    "risk": risk_name,
                    "seed": seed,
                    "states": states,
                    "scenarios": int(len(mdp.kernels)),
                    "stationary_policies": int(len(policies)),
                    "value_iteration_steps": int(len(vi_errors)),
                    "optimal_policy": bellman_policy.tolist(),
                })

    # Directly challenge the stationary-policy conclusion with policy trees
    # that condition on every observed state for three steps, then attach every
    # possible stationary infinite-horizon tail.  This is not a time-dependent
    # proxy: each evaluated tree has an exact discounted tail value.
    for risk_name in ("min", "max"):
        for seed in range(2):
            mdp = rectangular_binary_mdp(seed, states=2)
            optimal_value, _, _ = mdp.optimal_value(risk_name)
            for start in range(mdp.states):
                history_value, count = mdp.exhaustive_history_prefix(start, horizon=3, name=risk_name)
                history_errors.append(abs(history_value - optimal_value[start]))
                history_policies += count

    invalid_support_advantages: list[float] = []
    for risk_name in ("min", "max"):
        invalid = correlated_action_rows_control(1)
        optimal_value, _, _ = invalid.optimal_value(risk_name)
        for start in range(invalid.states):
            history_value, _ = invalid.exhaustive_history_prefix(start, horizon=3, name=risk_name)
            invalid_support_advantages.append(history_value - optimal_value[start])

    # Proposition 4.3 algebra: min/max/expectation pass independent additivity;
    # CVaR does not, excluding it from either DP condition by the paper's
    # necessary implication.
    x = np.array([-2.0, 4.0])
    y = np.array([-1.0, 5.0])
    p = np.array([0.5, 0.5])
    xy = (x[:, None] + y[None, :]).reshape(-1)
    pxy = (p[:, None] * p[None, :]).reshape(-1)
    additivity = {}
    for risk_name in ("min", "max", "mean", "cvar"):
        additivity[risk_name] = risk(xy, pxy, risk_name) - risk(x, p, risk_name) - risk(y, p, risk_name)

    # The paper's static-kernel warning has an exact finite-support witness.
    static = static_expectation_counterexample()
    policy = np.zeros(static.states, dtype=int)
    static_mean = static.static_value(policy, "mean")
    static_bellman = static.bellman_policy(static_mean, policy, "mean")
    static_expectation_residual = max_abs(static_mean, static_bellman)

    # In contrast, resampled expectation is exactly the fixed point of its
    # Bellman operator.  This is also an independent matrix calculation.
    expected_kernel = np.einsum("k,ksan->san", static.probabilities, static.kernels)
    expected_transition = expected_kernel[:, 0, :]
    expected_reward = np.sum(expected_transition * static.rewards[:, 0, :], axis=1)
    independent_resampled = np.linalg.solve(np.eye(static.states) - static.gamma * expected_transition, expected_reward)
    bellman_resampled = static.fixed_policy_value(policy, "mean")

    # Theorem 4.4's universal statement is additionally stress-tested on the
    # complete 35-member 4-atom family of monotone, law-invariant ordered
    # weighted risks with quarter-grid weights.  On six exact rectangular MDP
    # instances, only min/max obey static Bellman evaluation.  Proposition 4.3
    # independently leaves only min/max/mean after additivity on three pairs.
    spectral_weights = [np.asarray(parts, dtype=float) / 4.0 for parts in integer_simplex(4, 4)]
    static_survivors = []
    resampled_survivors = []
    spectral_static_max = {}
    spectral_additivity_max = {}
    spectral_multiplicativity_max = {}
    pairs = [
        (np.array([-2.0, 4.0]), np.array([-1.0, 5.0])),
        (np.array([-3.0, 2.0]), np.array([0.0, 7.0])),
        (np.array([-5.0, 1.0]), np.array([-4.0, 8.0])),
    ]
    positive_pairs = [
        (np.array([0.2, 3.0]), np.array([0.4, 5.0])),
        (np.array([0.1, 2.0]), np.array([0.7, 7.0])),
        (np.array([0.3, 4.0]), np.array([0.5, 6.0])),
    ]
    for weights in spectral_weights:
        label = ",".join(f"{weight:.2g}" for weight in weights)
        static_residual = max(static_bellman_residual(rectangular_binary_mdp(seed, states=2, actions=1), weights) for seed in range(6))
        additivity_residual = max(abs(independent_additivity_residual(weights, x, y)) for x, y in pairs)
        multiplicativity_residual = max(abs(independent_multiplicativity_residual(weights, x, z)) for x, z in positive_pairs)
        spectral_static_max[label] = static_residual
        spectral_additivity_max[label] = additivity_residual
        spectral_multiplicativity_max[label] = multiplicativity_residual
        if static_residual < 1e-10:
            static_survivors.append(label)
        if additivity_residual < 1e-10 and multiplicativity_residual < 1e-10:
            resampled_survivors.append(label)

    output = {
        "paper": "Dynamic Programming for Epistemic Uncertainty in Markov Decision Processes",
        "openreview_id": "oUv02QKUxG",
        "arxiv": "2602.03381",
        "claim_1_stationary_optimality": {
            "instances": len(rows),
            "max_error_bellman_vs_static": max(stationary_errors),
            "stationary_policies_exhausted_per_instance": [4, 8, 16],
            "history_dependent_prefix_policies_exhausted": history_policies,
            "max_history_prefix_advantage": max(history_errors),
            "invalid_non_product_control_history_advantage": max(invalid_support_advantages),
        },
        "claim_2_iteration_convergence": {
            "value_iteration_bound_max_slack": max(vi_bounds),
            "policy_iteration_monotone": pi_monotone,
        },
        "claim_3_theorem_4_4_classification": {
            "static_expectation_bellman_residual": static_expectation_residual,
            "robust_and_optimistic_positive_controls": "covered by 36 exact rectangular instances",
            "four_atom_spectral_risks_exhausted": len(spectral_weights),
            "static_bellman_survivors": static_survivors,
            "spectral_static_residuals": spectral_static_max,
            "resampled_expectation_independent_matrix_error": max_abs(independent_resampled, bellman_resampled),
        },
        "claim_4_theorem_4_5_w1_classification": {
            "resampled_expectation_independent_matrix_error": max_abs(independent_resampled, bellman_resampled),
            "static_expectation_witness_residual": static_expectation_residual,
            "static_spectral_survivors": static_survivors,
        },
        "claim_5_common_risk_incompatibility": {
            "independent_additivity_residuals": additivity,
            "proposition_4_3_algebra_survivors": resampled_survivors,
            "spectral_additivity_residuals": spectral_additivity_max,
            "spectral_multiplicativity_residuals": spectral_multiplicativity_max,
        },
        "rows": rows,
    }
    destination = Path("outputs/summary.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
