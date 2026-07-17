# Primary-source evidence map

- Paper: *Dynamic Programming for Epistemic Uncertainty in Markov Decision
  Processes*, Axel Benyamine, Julien Grand-Clément, Marek Petrik, Michael I.
  Jordan, and Alain Durmus.
- Primary source: [arXiv:2602.03381](https://arxiv.org/abs/2602.03381), PDF
  SHA-256 `53fe7f223e13a26dea976e8addf6d478f85830c2ff7a272abff5908755e20ee3`
  (retrieved 2026-07-17).
- No author implementation was located during the source audit; this project is
  a clean-room implementation of equations (6)–(8) and Algorithms 1–2.

## Scored-claim interpretation

| Challenge claim | Source location | Independent evidence in this repository |
|---|---|---|
| Stationary optimal policy | Theorem 4.1 | `FiniteAmbiguityMDP` implements equations (7)–(8). `nominal_values` solves each kernel's linear system independently; `exhaustive_stationary` checks all stationary policies; `exhaustive_history_prefix` checks state-history trees followed by exact infinite-horizon tails. |
| Value and policy iteration | Proposition 4.2, Algorithms 1–2 | `run_audit.py` evaluates the stated `2γ/(1−γ)` greedy-policy bound and componentwise policy-improvement condition. |
| Monotone law-invariant classification | Theorem 4.4 | `spectral.py` exhausts the 35 four-atom quarter-grid ordered weighted risks. Static Bellman evaluation retains only essential minimum and maximum; resampled expectation is independently solved from the expected kernel. |
| W1-continuous classification | Theorem 4.5; Example 3.8 | A finite-support static-expectation analogue is a direct Bellman counterexample. Resampled expectation agrees with an independent expected-transition matrix solve. |
| Incompatibility of CVaR/VaR/entropic risks | Discussion after Theorems 4.4–4.5; Proposition 4.3 | Exact discrete independent sums test the required additivity and exclude CVaR; the spectral algebra control retains exactly min/max/expectation. |

## Assumption audit

Theorems 4.1–4.2 require monotonicity, translation invariance, and Conditions
1–2. Theorems 4.4–4.5 add law invariance and, respectively, monotonicity or
W1 continuity. The positive constructions use a full Cartesian product over
every state-action transition row. The negative correlated-row construction
breaks that product condition on purpose and is never used as positive evidence.
