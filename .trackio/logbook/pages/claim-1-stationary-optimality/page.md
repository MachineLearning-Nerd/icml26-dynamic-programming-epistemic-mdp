# Claim 1 — Stationary optimality


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_bb209e88aa56", "created_at": "2026-07-17T08:42:57+00:00", "title": "Result"}
-->
The robust and optimistic Bellman solutions match independently solved static-kernel values over 36 full state-action-product MDPs (maximum error 2.70e-13). All stationary policies are exhausted; 4,096 state-history action prefixes with exact stationary tails show no improvement (maximum advantage 2.50e-13).


---
<!-- trackio-cell
{"type": "code", "id": "cell_582423599c7b", "created_at": "2026-07-17T08:43:32+00:00", "title": "Full exact audit", "command": ["python", "repro/run_audit.py"], "exit_code": 0, "duration_s": 13.751}
-->
````bash
$ python repro/run_audit.py
````

exit 0 · 13.8s


````python title=run_audit.py
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

````


````output
{
  "paper": "Dynamic Programming for Epistemic Uncertainty in Markov Decision Processes",
  "openreview_id": "oUv02QKUxG",
  "arxiv": "2602.03381",
  "claim_1_stationary_optimality": {
    "instances": 36,
    "max_error_bellman_vs_static": 2.69562150378988e-13,
    "stationary_policies_exhausted_per_instance": [
      4,
      8,
      16
    ],
    "history_dependent_prefix_policies_exhausted": 4096,
    "max_history_prefix_advantage": 2.5002222514558525e-13,
    "invalid_non_product_control_history_advantage": 0.10555355069155281
  },
  "claim_2_iteration_convergence": {
    "value_iteration_bound_max_slack": 0.0,
    "policy_iteration_monotone": true
  },
  "claim_3_theorem_4_4_classification": {
    "static_expectation_bellman_residual": 0.08333333333333348,
    "robust_and_optimistic_positive_controls": "covered by 36 exact rectangular instances",
    "four_atom_spectral_risks_exhausted": 35,
    "static_bellman_survivors": [
      "0,0,0,1",
      "1,0,0,0"
    ],
    "spectral_static_residuals": {
      "0,0,0,1": 8.881784197001252e-16,
      "0,0,0.25,0.75": 0.0806330688289596,
      "0,0,0.5,0.5": 0.1612661376579183,
      "0,0,0.75,0.25": 0.2418992064868779,
      "0,0,1,0": 0.3225322753158366,
      "0,0.25,0,0.75": 0.34678787584481086,
      "0,0.25,0.25,0.5": 0.3469105453960234,
      "0,0.25,0.5,0.25": 0.3470332149472368,
      "0,0.25,0.75,0": 0.3471558844984497,
      "0,0.5,0,0.5": 0.4623838344597472,
      "0,0.5,0.25,0.25": 0.4626291735621735,
      "0,0.5,0.5,0": 0.4628745126645999,
      "0,0.75,0,0.25": 0.3467878758448104,
      "0,0.75,0.25,0": 0.34715588449845014,
      "0,1,0,0": 0.34466539064611446,
      "0.25,0,0,0.75": 0.3403256241915511,
      "0.25,0,0.25,0.5": 0.34044829374276453,
      "0.25,0,0.5,0.25": 0.34057096329397707,
      "0.25,0,0.75,0": 0.34069363284519016,
      "0.25,0.25,0,0.5": 0.4580756666909075,
      "0.25,0.25,0.25,0.25": 0.4583210057933339,
      "0.25,0.25,0.5,0": 0.4585663448957601,
      "0.25,0.5,0,0.25": 0.3446337919603901,
      "0.25,0.5,0.25,0": 0.34500180061402985,
      "0.25,0.75,0,0": 0.25849904298458587,
      "0.5,0,0,0.5": 0.4537674989220678,
      "0.5,0,0.25,0.25": 0.4540128380244942,
      "0.5,0,0.5,0": 0.4542581771269205,
      "0.5,0.25,0,0.25": 0.3424797080759707,
      "0.5,0.25,0.25,0": 0.34284771672961023,
      "0.5,0.5,0,0": 0.17233269532305726,
      "0.75,0,0,0.25": 0.3403256241915509,
      "0.75,0,0.25,0": 0.3406936328451906,
      "0.75,0.25,0,0": 0.08616634766152864,
      "1,0,0,0": 4.440892098500626e-16
    },
    "resampled_expectation_independent_matrix_error": 9.43689570931383e-15
  },
  "claim_4_theorem_4_5_w1_classification": {
    "resampled_expectation_independent_matrix_error": 9.43689570931383e-15,
    "static_expectation_witness_residual": 0.08333333333333348,
    "static_spectral_survivors": [
      "0,0,0,1",
      "1,0,0,0"
    ]
  },
  "claim_5_common_risk_incompatibility": {
    "independent_additivity_residuals": {
      "min": 0.0,
      "max": 0.0,
      "mean": 0.0,
      "cvar": 3.0
    },
    "proposition_4_3_algebra_survivors": [
      "0,0,0,1",
      "0.25,0.25,0.25,0.25",
      "1,0,0,0"
    ],
    "spectral_additivity_residuals": {
      "0,0,0,1": 0.0,
      "0,0,0.25,0.75": 1.5,
      "0,0,0.5,0.5": 3.0,
      "0,0,0.75,0.25": 4.5,
      "0,0,1,0": 6.0,
      "0,0.25,0,0.75": 1.5,
      "0,0.25,0.25,0.5": 0.0,
      "0,0.25,0.5,0.25": 1.5,
      "0,0.25,0.75,0": 3.0,
      "0,0.5,0,0.5": 3.0,
      "0,0.5,0.25,0.25": 1.5,
      "0,0.5,0.5,0": 0.0,
      "0,0.75,0,0.25": 4.5,
      "0,0.75,0.25,0": 3.0,
      "0,1,0,0": 6.0,
      "0.25,0,0,0.75": 0.0,
      "0.25,0,0.25,0.5": 1.5,
      "0.25,0,0.5,0.25": 3.0,
      "0.25,0,0.75,0": 4.5,
      "0.25,0.25,0,0.5": 1.5,
      "0.25,0.25,0.25,0.25": 0.0,
      "0.25,0.25,0.5,0": 1.5,
      "0.25,0.5,0,0.25": 3.0,
      "0.25,0.5,0.25,0": 1.5,
      "0.25,0.75,0,0": 4.5,
      "0.5,0,0,0.5": 0.0,
      "0.5,0,0.25,0.25": 1.5,
      "0.5,0,0.5,0": 3.0,
      "0.5,0.25,0,0.25": 1.5,
      "0.5,0.25,0.25,0": 0.0,
      "0.5,0.5,0,0": 3.0,
      "0.75,0,0,0.25": 0.0,
      "0.75,0,0.25,0": 1.5,
      "0.75,0.25,0,0": 1.5,
      "1,0,0,0": 0.0
    },
    "spectral_multiplicativity_residuals": {
      "0,0,0,1": 0.0,
      "0,0,0.25,0.75": 5.5,
      "0,0,0.5,0.5": 11.0,
      "0,0,0.75,0.25": 16.5,
      "0,0,1,0": 22.0,
      "0,0.25,0,0.75": 4.228124999999999,
      "0,0.25,0.25,0.5": 1.2718750000000014,
      "0,0.25,0.5,0.25": 6.7718750000000005,
      "0,0.25,0.75,0": 12.271875000000001,
      "0,0.5,0,0.5": 5.9125000000000005,
      "0,0.5,0.25,0.25": 0.41250000000000053,
      "0,0.5,0.5,0": 5.0875,
      "0,0.75,0,0.25": 5.053125,
      "0,0.75,0.25,0": 0.44687500000000013,
      "0,1,0,0": 1.65,
      "0.25,0,0,0.75": 3.8156250000000007,
      "0.25,0,0.25,0.5": 1.684375000000001,
      "0.25,0,0.5,0.25": 7.184375000000001,
      "0.25,0,0.75,0": 12.684375000000001,
      "0.25,0.25,0,0.5": 5.500000000000001,
      "0.25,0.25,0.25,0.25": 0.0,
      "0.25,0.25,0.5,0": 5.5,
      "0.25,0.5,0,0.25": 4.640625,
      "0.25,0.5,0.25,0": 0.859375,
      "0.25,0.75,0,0": 1.2375,
      "0.5,0,0,0.5": 5.0874999999999995,
      "0.5,0,0.25,0.25": 0.41249999999999964,
      "0.5,0,0.5,0": 5.9125,
      "0.5,0.25,0,0.25": 4.228125,
      "0.5,0.25,0.25,0": 1.271875,
      "0.5,0.5,0,0": 0.8249999999999998,
      "0.75,0,0,0.25": 3.815625,
      "0.75,0,0.25,0": 1.684375,
      "0.75,0.25,0,0": 0.4125,
      "1,0,0,0": 0.0
    }
  },
  "rows": [
    {
      "risk": "min",
      "seed": 0,
      "states": 2,
      "scenarios": 16,
      "stationary_policies": 4,
      "value_iteration_steps": 94,
      "optimal_policy": [
        1,
        0
      ]
    },
    {
      "risk": "min",
      "seed": 1,
      "states": 2,
      "scenarios": 16,
      "stationary_policies": 4,
      "value_iteration_steps": 90,
      "optimal_policy": [
        1,
        1
      ]
    },
    {
      "risk": "min",
      "seed": 2,
      "states": 2,
      "scenarios": 16,
      "stationary_policies": 4,
      "value_iteration_steps": 92,
      "optimal_policy": [
        0,
        0
      ]
    },
    {
      "risk": "min",
      "seed": 3,
      "states": 2,
      "scenarios": 16,
      "stationary_policies": 4,
      "value_iteration_steps": 93,
      "optimal_policy": [
        1,
        0
      ]
    },
    {
      "risk": "min",
      "seed": 4,
      "states": 2,
      "scenarios": 16,
      "stationary_policies": 4,
      "value_iteration_steps": 96,
      "optimal_policy": [
        0,
        1
      ]
    },
    {
      "risk": "min",
      "seed": 5,
      "states": 2,
      "scenarios": 16,
      "stationary_policies": 4,
      "value_iteration_steps": 90,
      "optimal_policy": [
        0,
        1
      ]
    },
    {
      "risk": "min",
      "seed": 0,
      "states": 3,
      "scenarios": 64,
      "stationary_policies": 8,
      "value_iteration_steps": 96,
      "optimal_policy": [
        1,
        1,
        0
      ]
    },
    {
      "risk": "min",
      "seed": 1,
      "states": 3,
      "scenarios": 64,
      "stationary_policies": 8,
      "value_iteration_steps": 84,
      "optimal_policy": [
        0,
        1,
        1
      ]
    },
    {
      "risk": "min",
      "seed": 2,
      "states": 3,
      "scenarios": 64,
      "stationary_policies": 8,
      "value_iteration_steps": 93,
      "optimal_policy": [
        1,
        1,
        1
      ]
    },
    {
      "risk": "min",
      "seed": 3,
      "states": 3,
      "scenarios": 64,
      "stationary_policies": 8,
      "value_iteration_steps": 93,
      "optimal_policy": [
        0,
        1,
        0
      ]
    },
    {
      "risk": "min",
      "seed": 4,
      "states": 3,
      "scenarios": 64,
      "stationary_policies": 8,
      "value_iteration_steps": 90,
      "optimal_policy": [
        0,
        1,
        1
      ]
    },
    {
      "risk": "min",
      "seed": 5,
      "states": 3,
      "scenarios": 64,
      "stationary_policies": 8,
      "value_iteration_steps": 92,
      "optimal_policy": [
        0,
        0,
        1
      ]
    },
    {
      "risk": "min",
      "seed": 6,
      "states": 3,
      "scenarios": 64,
      "stationary_policies": 8,
      "value_iteration_steps": 93,
      "optimal_policy": [
        1,
        1,
        1
      ]
    },
    {
      "risk": "min",
      "seed": 7,
      "states": 3,
      "scenarios": 64,
      "stationary_policies": 8,
      "value_iteration_steps": 91,
      "optimal_policy": [
        0,
        1,
        1
      ]
    },
    {
      "risk": "min",
      "seed": 0,
      "states": 4,
      "scenarios": 256,
      "stationary_policies": 16,
      "value_iteration_steps": 94,
      "optimal_policy": [
        0,
        0,
        1,
        1
      ]
    },
    {
      "risk": "min",
      "seed": 1,
      "states": 4,
      "scenarios": 256,
      "stationary_policies": 16,
      "value_iteration_steps": 90,
      "optimal_policy": [
        1,
        0,
        1,
        0
      ]
    },
    {
      "risk": "min",
      "seed": 2,
      "states": 4,
      "scenarios": 256,
      "stationary_policies": 16,
      "value_iteration_steps": 90,
      "optimal_policy": [
        1,
        1,
        0,
        0
      ]
    },
    {
      "risk": "min",
      "seed": 3,
      "states": 4,
      "scenarios": 256,
      "stationary_policies": 16,
      "value_iteration_steps": 77,
      "optimal_policy": [
        0,
        0,
        0,
        1
      ]
    },
    {
      "risk": "max",
      "seed": 0,
      "states": 2,
      "scenarios": 16,
      "stationary_policies": 4,
      "value_iteration_steps": 96,
      "optimal_policy": [
        1,
        0
      ]
    },
    {
      "risk": "max",
      "seed": 1,
      "states": 2,
      "scenarios": 16,
      "stationary_policies": 4,
      "value_iteration_steps": 97,
      "optimal_policy": [
        1,
        0
      ]
    },
    {
      "risk": "max",
      "seed": 2,
      "states": 2,
      "scenarios": 16,
      "stationary_policies": 4,
      "value_iteration_steps": 91,
      "optimal_policy": [
        0,
        1
      ]
    },
    {
      "risk": "max",
      "seed": 3,
      "states": 2,
      "scenarios": 16,
      "stationary_policies": 4,
      "value_iteration_steps": 97,
      "optimal_policy": [
        1,
        0
      ]
    },
    {
      "risk": "max",
      "seed": 4,
      "states": 2,
      "scenarios": 16,
      "stationary_policies": 4,
      "value_iteration_steps": 97,
      "optimal_policy": [
        0,
        1
      ]
    },
    {
      "risk": "max",
      "seed": 5,
      "states": 2,
      "scenarios": 16,
      "stationary_policies": 4,
      "value_iteration_steps": 96,
      "optimal_policy": [
        1,
        1
      ]
    },
    {
      "risk": "max",
      "seed": 0,
      "states": 3,
      "scenarios": 64,
      "stationary_policies": 8,
      "value_iteration_steps": 98,
      "optimal_policy": [
        1,
        1,
        0
      ]
    },
    {
      "risk": "max",
      "seed": 1,
      "states": 3,
      "scenarios": 64,
      "stationary_policies": 8,
      "value_iteration_steps": 93,
      "optimal_policy": [
        0,
        1,
        1
      ]
    },
    {
      "risk": "max",
      "seed": 2,
      "states": 3,
      "scenarios": 64,
      "stationary_policies": 8,
      "value_iteration_steps": 96,
      "optimal_policy": [
        1,
        1,
        1
      ]
    },
    {
      "risk": "max",
      "seed": 3,
      "states": 3,
      "scenarios": 64,
      "stationary_policies": 8,
      "value_iteration_steps": 96,
      "optimal_policy": [
        0,
        1,
        0
      ]
    },
    {
      "risk": "max",
      "seed": 4,
      "states": 3,
      "scenarios": 64,
      "stationary_policies": 8,
      "value_iteration_steps": 93,
      "optimal_policy": [
        0,
        1,
        1
      ]
    },
    {
      "risk": "max",
      "seed": 5,
      "states": 3,
      "scenarios": 64,
      "stationary_policies": 8,
      "value_iteration_steps": 91,
      "optimal_policy": [
        0,
        0,
        0
      ]
    },
    {
      "risk": "max",
      "seed": 6,
      "states": 3,
      "scenarios": 64,
      "stationary_policies": 8,
      "value_iteration_steps": 96,
      "optimal_policy": [
        1,
        1,
        1
      ]
    },
    {
      "risk": "max",
      "seed": 7,
      "states": 3,
      "scenarios": 64,
      "stationary_policies": 8,
      "value_iteration_steps": 93,
      "optimal_policy": [
        0,
        1,
        1
      ]
    },
    {
      "risk": "max",
      "seed": 0,
      "states": 4,
      "scenarios": 256,
      "stationary_policies": 16,
      "value_iteration_steps": 97,
      "optimal_policy": [
        0,
        0,
        1,
        1
      ]
    },
    {
      "risk": "max",
      "seed": 1,
      "states": 4,
      "scenarios": 256,
      "stationary_policies": 16,
      "value_iteration_steps": 94,
      "optimal_policy": [
        0,
        0,
        0,
        0
      ]
    },
    {
      "risk": "max",
      "seed": 2,
      "states": 4,
      "scenarios": 256,
      "stationary_policies": 16,
      "value_iteration_steps": 94,
      "optimal_policy": [
        1,
        1,
        0,
        0
      ]
    },
    {
      "risk": "max",
      "seed": 3,
      "states": 4,
      "scenarios": 256,
      "stationary_policies": 16,
      "value_iteration_steps": 96,
      "optimal_policy": [
        1,
        1,
        0,
        1
      ]
    }
  ]
}

````
