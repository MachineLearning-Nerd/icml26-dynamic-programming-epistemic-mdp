# Primary-source evidence map

## Paper record

- Title: *Dynamic Programming for Epistemic Uncertainty in Markov Decision
  Processes*.
- Authors: Axel Benyamine, Julien Grand-Clément, Marek Petrik, Michael I.
  Jordan, and Alain Durmus.
- Primary record: [arXiv:2602.03381](https://arxiv.org/abs/2602.03381).
- Audited version: [arXiv v1](https://arxiv.org/abs/2602.03381v1), retrieved
  2026-07-17.
- OpenReview: [oUv02QKUxG](https://openreview.net/forum?id=oUv02QKUxG).
- Audited PDF SHA-256:
  53fe7f223e13a26dea976e8addf6d478f85830c2ff7a272abff5908755e20ee3.

The live arXiv record now advertises v2, revised 2026-08-07. The finite
producer and theorem map in this repository are explicitly tied to the v1
snapshot above; this prevents a later paper revision from silently changing
the reproduction target.

No author implementation was located during the source audit. The repository
is a clean-room implementation of equations 6–8, Algorithms 1–2, the
Theorem 4.4/4.5 risk-measure mechanisms, and the Proposition 4.3 controls.

## Claim mapping

| Claim | Paper location | Independent repository evidence |
| --- | --- | --- |
| C1 — stationary optimal policy | Theorem 4.1 | mdp.py implements the ambiguity-averse Bellman model. nominal_values independently solves each kernel's linear system; exhaustive_stationary checks all stationary policies; exhaustive_history_prefix checks finite history trees followed by exact infinite-horizon stationary tails. |
| C2 — value and policy iteration | Proposition 4.2, Algorithms 1–2, Appendix G | verify_prop_4_2.py records the complete 11-step proof dependency chain and exact rate algebra. audit_prop_4_2.py independently validates dependency order and mutation rejection. run_audit.py supplies finite algorithm corroboration. |
| C3 — monotone law-invariant classification | Theorem 4.4 | spectral.py exhausts all 35 four-atom quarter-grid ordered weighted risks on six rectangular one-action instances; static Bellman evaluation retains only minimum and maximum, while resampled expectation is checked by an independent expected-kernel solve. |
| C4 — W1-continuous classification | Theorem 4.5 and Example 3.8 | The finite-support static-expectation analogue is a direct Bellman counterexample. Resampled expectation agrees with an independent expected-transition matrix solve. |
| C5 — common-risk incompatibility | Proposition 4.3 and discussion after Theorems 4.4–4.5 | Exact finite independent sums and positive products test additivity and multiplicativity. Minimum, maximum, and uniform expectation survive the control; CVaR and nontrivial mixtures fail. |

## Assumptions and controls

Theorems 4.1–4.2 require a finite MDP, discount factor below one,
monotonicity, translation invariance, and the paper's Conditions 1–2.
Theorems 4.4–4.5 add law invariance and their respective monotonicity or W1
continuity assumptions. Positive finite experiments use a full Cartesian
product over every state-action transition row.

The correlated-action-row experiment deliberately breaks that product
condition and produces a history-dependent advantage. The static expectation
counterexample deliberately distinguishes applying a risk measure to complete
returns from resampling a transition kernel at each Bellman step. Both are
negative controls and are never counted as positive theorem evidence.

## Evidence files

- outputs/summary.json is the complete producer output.
- outputs/claim_summary.json is a compact synchronized record.
- outputs/proposition_4_2_audit.json is the durable Proposition 4.2 proof and
  mutation-control result.
- CLAIM_EVIDENCE.md explains the scope and limits claim by claim.
