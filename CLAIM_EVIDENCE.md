# Claim-to-evidence ledger

This ledger separates finite computational corroboration from the
Proposition 4.2 proof-chain audit. VERIFIED_SCOPED means that the listed
finite checks pass; it does not promote a finite check into a universal proof.

## C1 — stationary optimality

- **Paper connection:** Theorem 4.1.
- **Model construction:** rectangular_binary_mdp creates a Cartesian product of
  two transition-row choices for every state-action row. The audit covers
  36 instances: 2-state seeds 0–5, 3-state seeds 0–7, and 4-state seeds 0–3,
  each under minimum and maximum risk.
- **Independent checks:** Bellman optimal values are compared with static
  values and exhaustive deterministic stationary policies. The policy counts
  are 4, 8, and 16 for the three state sizes. A separate enumeration checks
  4,096 history-dependent prefixes followed by exact stationary tails.
- **Results:** Maximum Bellman/static error is 2.69562150378988e-13;
  maximum history-prefix advantage is 2.5002222514558525e-13.
- **Negative control:** correlated_action_rows_control intentionally violates
  state-action product support and yields a 0.10555355069155281
  history-dependent advantage. It is a control, not positive evidence.
- **Boundary:** This is finite corroboration under the paper's rectangular
  assumptions, not an independent proof of the universal theorem.

## C2 — value and policy iteration

- **Paper connection:** Proposition 4.2 and Algorithms 1–2.
- **Finite corroboration:** repro/run_audit.py records zero maximum
  value-iteration bound slack and monotone policy iteration on the finite
  rectangular controls.
- **Proof-chain certificate:** repro/src/verify_prop_4_2.py transcribes the
  11-step dependency chain from the stated assumptions through Bellman
  structure, fixed points, both rate derivations, and completion. It checks the
  rate algebra with exact Fraction arithmetic over 100 rational gamma values
  and 13 iteration counts, for 3,900 exact checks.
- **Independent audit:** repro/src/audit_prop_4_2.py does not use the primary
  DAG validator. It independently checks dependency order and rejects three
  mutations: missing policy-rate step, duplicate step, and forward dependency.
  outputs/proposition_4_2_audit.json records all controls as passing.
- **Boundary:** The certificate audits a faithful proof transcription and its
  closure; it is not a formal proof assistant or a claim that all assumptions
  hold for arbitrary risk measures.

## C3 — monotone law-invariant classification

- **Paper connection:** Theorem 4.4.
- **Implementation:** repro/src/spectral.py enumerates every nonnegative
  quarter-grid weight vector summing to one over four equally likely atoms:
  35 ordered weighted risks. static_bellman_residual evaluates each risk on
  six one-action rectangular instances.
- **Results:** Only the two essential extrema survive static Bellman
  evaluation: weights 0,0,0,1 and 1,0,0,0. The producer also covers 36 exact
  rectangular robust/optimistic controls. Resampled expectation agrees with an
  independently solved expected transition matrix to 9.43689570931383e-15.
- **Boundary:** The 35-point spectral family and finite controls corroborate
  the classification; they do not enumerate every law-invariant risk measure.

## C4 — W1-continuous classification

- **Paper connection:** Theorem 4.5 and Example 3.8.
- **Implementation:** The finite-support static expectation witness compares
  static risk applied to complete-return values with the one-step Bellman
  operator. A separate expected-kernel linear solve checks resampled
  expectation.
- **Results:** Static expectation has residual
  0.08333333333333348. Resampled expectation's independent matrix error is
  9.43689570931383e-15. Static spectral controls again retain only the
  essential extrema in the audited instances.
- **Boundary:** This is a finite-support analogue and classification control,
  not a universal W1 continuity proof.

## C5 — incompatibility of common risk measures

- **Paper connection:** Proposition 4.3 and the discussion following
  Theorems 4.4–4.5.
- **Implementation:** independent_additivity_residual and
  independent_multiplicativity_residual evaluate the required identities on
  exact finite independent sums and positive products.
- **Results:** Additivity residuals are 0 for minimum, maximum, and mean, and
  3.0 for CVaR. The quarter-grid spectral algebra control retains minimum,
  uniform expectation, and maximum.
- **Boundary:** The controls demonstrate the paper's incompatibility
  mechanisms on finite supports; they do not claim to exhaust every risk
  measure outside the theorem's assumptions.

## Re-run

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=repro .venv/bin/python repro/run_audit.py
.venv/bin/python repro/src/verify_prop_4_2.py
.venv/bin/python repro/src/audit_prop_4_2.py
PYTHONPATH=repro .venv/bin/python -m pytest -q repro/tests
python3 verify_final.py
~~~
