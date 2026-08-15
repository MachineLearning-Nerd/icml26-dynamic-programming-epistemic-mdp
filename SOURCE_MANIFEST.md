# Source manifest

## Primary paper

- Title: *Dynamic Programming for Epistemic Uncertainty in Markov Decision
  Processes*.
- Authors: Axel Benyamine, Julien Grand-Clément, Marek Petrik, Michael I.
  Jordan, and Alain Durmus.
- Current record: [arXiv:2602.03381](https://arxiv.org/abs/2602.03381).
- OpenReview: [oUv02QKUxG](https://openreview.net/forum?id=oUv02QKUxG).
- Audited snapshot: arXiv v1, [2602.03381v1](https://arxiv.org/abs/2602.03381v1),
  retrieved 2026-07-17.
- Audited PDF SHA-256:
  53fe7f223e13a26dea976e8addf6d478f85830c2ff7a272abff5908755e20ee3.
- The live arXiv record has a later v2 revision dated 2026-08-07. This
  repository's numerical scope and theorem mapping are explicitly tied to the
  v1 snapshot above.

## Independent implementation inputs

- repro/src/mdp.py: finite ambiguity-averse MDPs, Bellman operators, policy
  enumeration, rectangular support, history-prefix controls, and finite
  counterexamples.
- repro/src/spectral.py: ordered weighted risks and independent additivity/
  multiplicativity controls.
- repro/run_audit.py: five-claim finite producer and JSON writer.
- repro/src/verify_prop_4_2.py: standard-library Proposition 4.2 certificate.
- repro/src/audit_prop_4_2.py: independent DAG and mutation audit.
- repro/tests/test_mdp.py: 12 focused regression tests.
- requirements.txt: NumPy minimum version and pytest 8.x range.

No dataset, pretrained model, network service, or author implementation is
imported at runtime. No author implementation was located during the source
audit.

## Proof-certificate source snapshot

The Proposition 4.2 certificate records the source URL
https://ar5iv.labs.arxiv.org/html/2602.03381 and source snapshot SHA-256
839c9812631750eec0bf1362596fa94629d8195316c30cee37b8ed9c8b8d13f0.
This hash belongs to the source snapshot recorded inside the certificate; the
paper PDF hash above is the primary reproducibility artifact.

## Generated evidence

- outputs/summary.json: full finite producer output.
- outputs/claim_summary.json: compact synchronized summary.
- outputs/audit_stdout.json: captured full producer output.
- outputs/proposition_4_2_audit.json: durable proof-chain and mutation result.
- .trackio/logbook/ and pages/: historical static experiment notes retained
  for provenance; they are not runtime dependencies.
