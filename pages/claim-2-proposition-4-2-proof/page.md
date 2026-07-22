# Claim 2 — Proposition 4.2 complete convergence certificate

## Scored statement and source scope

The official Claim 2 says that value iteration and policy iteration converge
under the paper's ambiguity-averse framework, with respective error bounds

```text
||V^{pi_n}-V*||_infinity <= [2 gamma^n/(1-gamma)] ||V^{pi_0}-V*||_infinity
||V^{pi_n}-V*||_infinity <= gamma^n ||V^{pi_0}-V*||_infinity.
```

The claim was checked against the paper's Proposition 4.2 and its complete
proof in Appendix G using ar5iv HTML:

- source: `https://ar5iv.labs.arxiv.org/html/2602.03381`;
- scope: Section 4.1, Proposition 4.2, Algorithms 1–2, and Appendix G;
- fetched HTML SHA-256:
  `839c9812631750eec0bf1362596fa94629d8195316c30cee37b8ed9c8b8d13f0`.

The assumptions remain exactly those in the paper: a finite MDP in the stated
class, `0 <= gamma < 1`, a monotone translation-invariant risk measure, and
Conditions 1 and 2 for the selected static or resampled kernel model.

## Proof-level certificate

`repro/src/verify_prop_4_2.py` records the Appendix-G argument as a focused,
standard-library-only fail-closed dependency graph. The broader
`theorem_certificates.py` audit uses a legacy internal label `C1` that groups
Theorem 4.1 and Proposition 4.2; this focused verifier isolates the official
leaderboard Claim 2.

For value iteration the certificate checks the following complete chain:

1. `T` is a `gamma`-contraction and `V*=T V*`, hence
   `||V^n-V*|| <= gamma^n ||V^0-V*||`.
2. The policy `pi_n` is greedy for `V^{n-1}` and `V^{pi_n}` is the fixed point
   of `T_{pi_n}`.
3. The two contraction terms and triangle inequality give
   `(1-gamma)||V^{pi_n}-V*|| <= 2 gamma ||V^{n-1}-V*||`.
4. Substitution of the iterate bound gives the exact
   `2 gamma^n/(1-gamma)` policy bound.

For policy iteration it checks the half of the claim missing from the prior
judged output:

1. Greedy improvement gives
   `T_{pi_{n+1}}V^{pi_n}=T V^{pi_n} >= V^{pi_n}`.
2. Monotonicity and policy evaluation give
   `V^{pi_{n+1}} >= V^{pi_n}`.
3. Comparison with a stationary optimal greedy policy plus contraction gives
   `||V*-V^{pi_{n+1}}|| <= gamma ||V*-V^{pi_n}||`.
4. Induction yields the claimed `gamma^n` rate. The sticky tie rule and the
   finite stationary-policy set also give termination.

Imported mathematical ingredients are disclosed, not presented as newly
proved: Banach fixed-point uniqueness and the standard contraction induction.
The finite MDP runs below are corroboration rather than an extrapolation to the
universal quantifier.

## Deterministic execution and independent audit

```bash
PYTHONPATH=repro/src .venv/bin/python repro/src/verify_prop_4_2.py
PYTHONPATH=repro/src .venv/bin/python repro/src/audit_prop_4_2.py
```

Those two focused text files are self-contained and use only the Python
standard library. Additional local cross-checks used the broader repository:

```bash
PYTHONPATH=repro/src .venv/bin/python repro/src/run_theorem_audit.py
PYTHONPATH=repro .venv/bin/python -m pytest -q \
  repro/tests/test_theorem_certificates.py \
  repro/tests/test_mdp.py \
  repro/tests/test_scale_audit_c1.py
```

Fresh bounded-CPU result:

```text
primary theorem certificates: all_valid=true
focused Proposition 4.2 certificate: all_passed=true
independent DAG audit: true
exact Fraction rate checks: 3,900
mutation controls: missing policy-rate rejected; duplicate rejected;
                   forward dependency rejected
focused tests: 23 passed in 2.28s
```

The independent auditor also validates the separately generated deterministic
scale output:

```text
value-iteration bound checks: 2,520; violations: 0
policy-iteration gamma-rate checks: 1,066; violations: 0
policy-iteration runs terminated: 360/360
maximum policy updates: 3
```

Artifact hashes from this run:

```text
repro/src/verify_prop_4_2.py
  43c5ffb8bb5e08cd71b53493a23597367deadde87e774ae43ed3240d71af30f4
repro/src/audit_prop_4_2.py
  afbf55f0e8eaa8fb909698415131cc4af2680da37366bf44d72310cec2188cf2
repro/outputs/theorem_certificates.json
  ca940faa0b9874039f8f7352c4675e515758ceef4a322eb9ef11fcc5a3b4eec7
outputs/scale_audit_c1.json
  d06a6693a8aafefa897a5016f44dc6d3edef0f246f00171898474320447f20fb
```

## Fail-sensitive boundary

The independent auditor intentionally removes the policy-rate step, duplicates
a proof step, and moves the conclusion before its dependencies. All three
mutations must be rejected. It also fails if either VI or PI has no checks, any
violation is nonzero, or any policy-iteration run fails to terminate.

This evidence supports Proposition 4.2 specifically. It does not weaken or
replace the previously judged pages, all of which remain reachable in the
additive logbook.
