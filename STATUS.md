# Reproduction status

## Current state

VERIFIED_SCOPED — the five paper claims have independent finite evidence, and
Claim 2 also has a separate VERIFIED_PROOF_AUDIT. The finite evidence is
bounded by the instances, risk weights, rectangularity assumptions, and
tolerances described in [CLAIM_EVIDENCE.md](CLAIM_EVIDENCE.md); it does not
replace the paper's universal arguments.

The public repository is now named
[icml26-dynamic-programming-epistemic-mdp](https://github.com/MachineLearning-Nerd/icml26-dynamic-programming-epistemic-mdp).
It has one published branch, main, and all reachable commits use the canonical
MachineLearning-Nerd identity. See [BRANCH_AUDIT.md](BRANCH_AUDIT.md).

## Verification record

- Paper: [arXiv:2602.03381](https://arxiv.org/abs/2602.03381), audited snapshot
  v1 from 2026-07-17; OpenReview oUv02QKUxG.
- Finite producer: 36 rectangular MDP audits, 35 spectral risk weights, 4,096
  history-dependent-prefix controls, and explicit non-product/static-risk
  negative controls.
- Proposition 4.2: 11 proof-chain steps, 3,900 exact fraction checks, valid
  independent DAG audit, and three passing mutation controls.
- Regression suite: 12 focused tests in repro/tests/test_mdp.py.
- Execution: CPU-only; no network, training, or model calls.

The earlier campaign queue mentioned a possible Hugging Face publication, but
Space creation was quota-limited. No external publication is treated as
evidence here; the public GitHub source and checked-in outputs are authoritative.
