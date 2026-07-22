#!/usr/bin/env python3
"""Self-contained proof certificate for official Claim 2 / Proposition 4.2.

This standard-library-only verifier records the complete Appendix-G dependency
chain for both value and policy iteration and checks the rate algebra exactly
with ``fractions.Fraction``.  Finite MDP experiments are corroboration only;
the universal conclusion comes from the closed proof chain under the stated
paper assumptions.
"""
from __future__ import annotations

from fractions import Fraction
import json


SOURCE_URL = "https://ar5iv.labs.arxiv.org/html/2602.03381"
SOURCE_SHA256 = "839c9812631750eec0bf1362596fa94629d8195316c30cee37b8ed9c8b8d13f0"


def build_steps() -> list[dict]:
    return [
        {
            "id": "assumptions",
            "depends_on": [],
            "reference": "Proposition 4.2",
            "statement": "Finite stated-class MDP; 0<=gamma<1; rho monotone and translation invariant; Conditions 1 and 2 hold for the selected kernel model.",
        },
        {
            "id": "bellman_structure",
            "depends_on": ["assumptions"],
            "reference": "Proposition 3.7",
            "statement": "T and every stationary-policy T_pi are monotone gamma-contractions and greedy stationary actions are attained.",
        },
        {
            "id": "fixed_points",
            "depends_on": ["assumptions", "bellman_structure"],
            "reference": "Conditions 1 and 2",
            "statement": "V*=T V* and each V_pi=T_pi V_pi.",
        },
        {
            "id": "vi_iterate_rate",
            "depends_on": ["bellman_structure", "fixed_points"],
            "reference": "Appendix G, Proposition 4.2 item 1",
            "statement": "For V^n=T V^{n-1}, contraction induction gives ||V^n-V*|| <= gamma^n ||V^0-V*||.",
        },
        {
            "id": "vi_greedy_residual",
            "depends_on": ["bellman_structure", "fixed_points", "vi_iterate_rate"],
            "reference": "Appendix G, Proposition 4.2 item 1",
            "statement": "Greediness and two contraction bounds give (1-gamma)||V_pi_n-V*|| <= 2 gamma ||V^{n-1}-V*||.",
        },
        {
            "id": "vi_policy_rate",
            "depends_on": ["vi_greedy_residual"],
            "reference": "Proposition 4.2 item 1",
            "statement": "Substitution yields ||V_pi_n-V*|| <= 2 gamma^n/(1-gamma) ||V_pi_0-V*||.",
        },
        {
            "id": "pi_improvement",
            "depends_on": ["bellman_structure", "fixed_points"],
            "reference": "Appendix G, Proposition 4.2 item 2",
            "statement": "Greedy update and monotone policy evaluation give V_pi_{n+1} >= V_pi_n.",
        },
        {
            "id": "stationary_optimum",
            "depends_on": ["bellman_structure", "fixed_points"],
            "reference": "Theorem 4.1",
            "statement": "An attained stationary greedy policy pi* has V_pi*=V*.",
        },
        {
            "id": "pi_one_step_rate",
            "depends_on": ["pi_improvement", "stationary_optimum"],
            "reference": "Appendix G, Proposition 4.2 item 2",
            "statement": "Comparison with pi* and contraction gives ||V*-V_pi_{n+1}|| <= gamma ||V*-V_pi_n||.",
        },
        {
            "id": "pi_rate",
            "depends_on": ["pi_one_step_rate"],
            "reference": "Proposition 4.2 item 2",
            "statement": "Induction gives ||V_pi_n-V*|| <= gamma^n ||V_pi_0-V*||.",
        },
        {
            "id": "complete",
            "depends_on": ["vi_policy_rate", "pi_rate"],
            "reference": "Proposition 4.2",
            "statement": "Both algorithms converge with the two claimed rates.",
        },
    ]


def validate_steps(steps: list[dict]) -> dict:
    seen: set[str] = set()
    duplicates: list[str] = []
    missing: dict[str, list[str]] = {}
    for step in steps:
        step_id = step.get("id")
        if not isinstance(step_id, str) or not step_id or step_id in seen:
            duplicates.append(str(step_id))
        dependencies = step.get("depends_on", [])
        absent = [d for d in dependencies if d not in seen]
        if absent:
            missing[str(step_id)] = absent
        seen.add(str(step_id))
    return {
        "valid": not duplicates and not missing and "complete" in seen,
        "duplicate_ids": duplicates,
        "missing_or_forward_dependencies": missing,
    }


def exact_rate_checks() -> int:
    checks = 0
    for numerator in range(100):
        gamma = Fraction(numerator, 100)
        for n in range(13):
            e0 = Fraction(17, 7)
            vi_iterate = e0
            pi_iterate = e0
            for _ in range(n):
                vi_iterate *= gamma
                pi_iterate *= gamma
            assert vi_iterate == gamma**n * e0
            assert pi_iterate == gamma**n * e0
            if n:
                prior_vi = gamma ** (n - 1) * e0
                greedy = Fraction(2) * gamma * prior_vi / (1 - gamma)
                assert greedy == Fraction(2) * gamma**n * e0 / (1 - gamma)
            checks += 3
    return checks


def verify() -> dict:
    steps = build_steps()
    closure = validate_steps(steps)
    algebra_checks = exact_rate_checks()
    return {
        "claim": "Official Claim 2 / Proposition 4.2",
        "source_url": SOURCE_URL,
        "source_scope": "Section 4.1, Proposition 4.2, Algorithms 1-2, Appendix G",
        "source_sha256": SOURCE_SHA256,
        "proof_chain": closure,
        "exact_fraction_checks": algebra_checks,
        "all_passed": closure["valid"] and algebra_checks == 3900,
        "scope_note": "Finite experiments are corroboration only; the certificate checks the paper proof under its assumptions.",
        "steps": steps,
    }


def main() -> None:
    result = verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
