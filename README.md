# ICML 2026 — Dynamic Programming for Epistemic MDPs

Independent, CPU-only reproduction audit for **Dynamic Programming for
Epistemic Uncertainty in Markov Decision Processes** by Axel Benyamine, Julien
Grand-Clément, Marek Petrik, Michael I. Jordan, and Alain Durmus.

| Field | Value |
| --- | --- |
| Paper | [arXiv:2602.03381](https://arxiv.org/abs/2602.03381) |
| Audited snapshot | [arXiv v1](https://arxiv.org/abs/2602.03381v1), retrieved 2026-07-17 |
| OpenReview | [oUv02QKUxG](https://openreview.net/forum?id=oUv02QKUxG) |
| Repository | [MachineLearning-Nerd/icml26-dynamic-programming-epistemic-mdp](https://github.com/MachineLearning-Nerd/icml26-dynamic-programming-epistemic-mdp) |
| Former campaign name | icml26-repro-oUv02QKUxG-epistemic-mdp |
| Reproduction status | VERIFIED_SCOPED |
| Runtime | CPU only; no network or model calls |
| Branches | main only; see [BRANCH_AUDIT.md](BRANCH_AUDIT.md) |

## What the paper does

The paper develops a theory of ambiguity-averse MDPs in which uncertain
transition probabilities are random variables and a risk measure is applied to
the random return. This framework connects several models of epistemic
uncertainty, extends value functions and Bellman operators, studies stationary
policies plus value and policy iteration, and characterizes which law-invariant
risk measures are compatible with dynamic programming.

This repository implements the paper's finite mechanisms independently. It
checks rectangular uncertainty sets, stationary and history-dependent controls,
Bellman iteration, finite-support risk-measure identities, and the assumptions
behind the classification results. The evidence is deliberately separated into
finite computational corroboration and a standard-library proof-chain audit.
It does not claim that finite experiments replace the paper's general proofs.

## Claim results and scope

VERIFIED_SCOPED means that the claim is independently checked on the listed
finite instances, risk families, assumptions, and numerical tolerances.
Claim 2 additionally has a VERIFIED_PROOF_AUDIT: the paper's Proposition 4.2
dependency chain and rate algebra are transcribed into a fail-closed,
standard-library certificate. That certificate audits the proof structure; it
is not a formal proof assistant.

| Claim | Paper connection | Evidence | Result |
| --- | --- | --- | --- |
| C1 — stationary optimality | Theorem 4.1 | 36 full-product finite MDPs; Bellman values agree with independently solved static values to 2.70e-13. Every stationary policy is exhausted (4, 8, or 16 per instance), and 4,096 finite history-dependent prefixes with exact stationary tails show no advantage above 2.50e-13. A deliberately non-product control produces a 0.10555 advantage. | VERIFIED_SCOPED |
| C2 — value and policy iteration | Proposition 4.2 and Algorithms 1–2 | Finite runs have zero maximum value-iteration bound slack and monotone policy iteration. The proof-chain certificate contains 11 dependency steps and 3,900 exact Fraction checks; the independent DAG audit rejects missing, duplicate, and forward dependencies. | VERIFIED_PROOF_AUDIT |
| C3 — monotone law-invariant classification | Theorem 4.4 | All 35 quarter-grid four-atom ordered risks are exhausted over the supplied spectral family. Only essential minimum and maximum pass static Bellman evaluation; resampled expectation agrees with an independent expected-kernel solve to 9.44e-15. | VERIFIED_SCOPED |
| C4 — W1-continuous classification | Theorem 4.5 and Example 3.8 | The finite-support static-expectation witness has Bellman residual 0.08333, while resampled expectation matches an independent matrix solve to 9.44e-15. Static spectral controls retain only the essential extrema in the audited instances. | VERIFIED_SCOPED |
| C5 — incompatibility of common risk measures | Proposition 4.3 and the discussion after Theorems 4.4–4.5 | Independent additivity residuals are zero for minimum, maximum, and expectation, but 3.0 for CVaR. The finite spectral algebra control retains minimum, uniform expectation, and maximum. | VERIFIED_SCOPED |

## How each claim is produced

| Claim | Production path |
| --- | --- |
| C1 | repro/src/mdp.py::rectangular_binary_mdp constructs full Cartesian products of state-action transition rows; FiniteAmbiguityMDP.optimal_value supplies the Bellman result; exhaustive_stationary and exhaustive_history_prefix independently enumerate controls. correlated_action_rows_control is a negative control that violates the product assumption. |
| C2 | repro/src/verify_prop_4_2.py transcribes the Proposition 4.2 dependency chain, validates dependency order, and checks rate identities over exact rational gamma values. repro/src/audit_prop_4_2.py independently audits the DAG and three mutation controls. |
| C3 | repro/src/spectral.py::static_bellman_residual exhausts the 35 nonnegative quarter-grid weight vectors; repro/run_audit.py evaluates six one-action rectangular instances and 36 robust/optimistic finite controls. |
| C4 | repro/src/spectral.py compares static risk evaluation with one-step Bellman evaluation, then separately computes the expected transition-kernel fixed point for resampled expectation. |
| C5 | repro/src/spectral.py::independent_additivity_residual and independent_multiplicativity_residual test exact finite independent-sum/product identities for the common risk controls. |

The full finite producer writes outputs/summary.json. The compact
outputs/claim_summary.json is kept in sync with it. The durable
outputs/proposition_4_2_audit.json records the independent proof audit.

## Reproduce

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=repro .venv/bin/python repro/run_audit.py
.venv/bin/python repro/src/verify_prop_4_2.py
.venv/bin/python repro/src/audit_prop_4_2.py
PYTHONPATH=repro .venv/bin/python -m pytest -q repro/tests
python3 verify_final.py
~~~

The repository has no dataset, pretrained model, author package, or network
runtime dependency. Requirements are recorded in
[requirements.txt](requirements.txt). The static Trackio/logbook files are
retained as historical experiment notes; the runnable source of truth is under
repro/ and the checked-in JSON evidence.

## Source and citation

The paper-version mapping, source hashes, theorem locations, assumptions, and
author-release search result are recorded in
[docs/paper_evidence.md](docs/paper_evidence.md) and
[SOURCE_MANIFEST.md](SOURCE_MANIFEST.md). No author implementation was located
during the source audit.

BibTeX for the paper:

~~~bibtex
@inproceedings{benyamine2026dynamic,
  title     = {Dynamic Programming for Epistemic Uncertainty in Markov Decision Processes},
  author    = {Benyamine, Axel and Grand-Cl{\'e}ment, Julien and Petrik, Marek and Jordan, Michael I. and Durmus, Alain},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
  eprint    = {2602.03381},
  archivePrefix = {arXiv}
}
~~~

Please also see [CITATION.cff](CITATION.cff) for the software citation.

## Thank you

Thank you to Axel Benyamine, Julien Grand-Clément, Marek Petrik, Michael I.
Jordan, and Alain Durmus for the careful theoretical framework, explicit
assumptions, and examples that make this independent audit possible. No
author code was located, so the implementation here is clean-room and does not
depend on an author release.

## Attribution

This reproduction and its audit trail are maintained by **MachineLearning-Nerd**.
All published repository commits use the canonical
MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com> identity.
