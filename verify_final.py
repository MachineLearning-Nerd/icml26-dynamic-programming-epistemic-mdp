#!/usr/bin/env python3
"""Fail-closed final-state checks for the epistemic-MDP audit repository."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CANONICAL = (
    "MachineLearning-Nerd",
    "MachineLearning-Nerd@users.noreply.github.com",
)
FINAL_REMOTE = "MachineLearning-Nerd/icml26-dynamic-programming-epistemic-mdp"


def run(*args: str) -> str:
    result = subprocess.run(args, cwd=ROOT, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def as_float(value: object) -> float:
    return float(value)


def main() -> None:
    required = [
        "README.md",
        "STATUS.md",
        "CLAIM_EVIDENCE.md",
        "SOURCE_MANIFEST.md",
        "BRANCH_AUDIT.md",
        "CITATION.cff",
        "docs/paper_evidence.md",
        "outputs/summary.json",
        "outputs/claim_summary.json",
        "outputs/audit_stdout.json",
        "outputs/proposition_4_2_audit.json",
        "repro/run_audit.py",
        "repro/src/mdp.py",
        "repro/src/spectral.py",
        "repro/src/verify_prop_4_2.py",
        "repro/src/audit_prop_4_2.py",
        "repro/tests/test_mdp.py",
    ]
    for relative in required:
        require((ROOT / relative).is_file(), f"missing required file: {relative}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in (
        "Dynamic Programming for Epistemic Uncertainty in Markov Decision Processes",
        "Axel Benyamine",
        "Julien Grand-Clément",
        "Marek Petrik",
        "Michael I. Jordan",
        "Alain Durmus",
        "VERIFIED_SCOPED",
        "VERIFIED_PROOF_AUDIT",
        "How each claim is produced",
        "Thank you",
        "MachineLearning-Nerd@users.noreply.github.com",
        "icml26-dynamic-programming-epistemic-mdp",
    ):
        require(phrase in readme, f"README missing required phrase: {phrase}")

    summary = json.loads((ROOT / "outputs/summary.json").read_text(encoding="utf-8"))
    compact = json.loads((ROOT / "outputs/claim_summary.json").read_text(encoding="utf-8"))
    proof = json.loads((ROOT / "outputs/proposition_4_2_audit.json").read_text(encoding="utf-8"))
    require(summary["openreview_id"] == "oUv02QKUxG", "OpenReview identifier changed")
    require(summary["arxiv"] == "2602.03381", "arXiv identifier changed")

    c1 = summary["claim_1_stationary_optimality"]
    require(c1["instances"] == 36, "C1 instance count changed")
    require(c1["stationary_policies_exhausted_per_instance"] == [4, 8, 16], "C1 policy scope changed")
    require(c1["history_dependent_prefix_policies_exhausted"] == 4096, "C1 history scope changed")
    require(as_float(c1["max_error_bellman_vs_static"]) <= 1e-10, "C1 Bellman/static error is too large")
    require(as_float(c1["max_history_prefix_advantage"]) <= 1e-10, "C1 history-prefix advantage is too large")
    require(as_float(c1["invalid_non_product_control_history_advantage"]) > 1e-3, "C1 negative control disappeared")

    c2 = summary["claim_2_iteration_convergence"]
    require(c2["value_iteration_bound_max_slack"] <= 1e-12, "C2 value-iteration bound failed")
    require(c2["policy_iteration_monotone"] is True, "C2 policy iteration is not monotone")
    require(proof["all_passed"] is True, "C2 proof audit did not pass")
    require(proof["primary_all_passed"] is True, "C2 primary certificate did not pass")
    require(proof["independent_dag_valid"] is True, "C2 independent DAG audit did not pass")
    require(proof["exact_fraction_checks"] == 3900, "C2 exact-check count changed")
    require(all(proof["mutation_controls"].values()), "C2 mutation control failed")

    c3 = summary["claim_3_theorem_4_4_classification"]
    require(c3["four_atom_spectral_risks_exhausted"] == 35, "C3 spectral scope changed")
    require(c3["static_bellman_survivors"] == ["0,0,0,1", "1,0,0,0"], "C3 survivor set changed")
    require("36 exact rectangular instances" in c3["robust_and_optimistic_positive_controls"], "C3 compact scope is stale")
    require(as_float(c3["static_expectation_bellman_residual"]) > 1e-3, "C3 expectation control disappeared")
    require(as_float(c3["resampled_expectation_independent_matrix_error"]) <= 1e-10, "C3 matrix cross-check failed")

    c4 = summary["claim_4_theorem_4_5_w1_classification"]
    require(as_float(c4["static_expectation_witness_residual"]) > 1e-3, "C4 static expectation witness disappeared")
    require(as_float(c4["resampled_expectation_independent_matrix_error"]) <= 1e-10, "C4 matrix cross-check failed")
    require(c4["static_spectral_survivors"] == ["0,0,0,1", "1,0,0,0"], "C4 survivor set changed")

    c5 = summary["claim_5_common_risk_incompatibility"]
    require(all(as_float(c5["independent_additivity_residuals"][name]) <= 1e-12 for name in ("min", "max", "mean")), "C5 additive controls failed")
    require(as_float(c5["independent_additivity_residuals"]["cvar"]) > 1.0, "C5 CVaR negative control disappeared")
    require(c5["proposition_4_3_algebra_survivors"] == ["0,0,0,1", "0.25,0.25,0.25,0.25", "1,0,0,0"], "C5 algebra survivors changed")
    require(compact["claim_3_theorem_4_4_classification"]["robust_and_optimistic_positive_controls"] == "covered by 36 exact rectangular instances", "compact summary is not synchronized")

    remote = run("git", "remote", "get-url", "origin")
    require(FINAL_REMOTE in remote, f"origin does not point to final repository: {remote}")
    require(run("git", "branch", "--show-current") == "main", "working branch is not main")
    refs = run("git", "for-each-ref", "--format=%(refname)").splitlines()
    require(not any("orx/" in ref for ref in refs), "an orx branch/ref remains")
    require(
        all(ref in {"refs/heads/main", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"} for ref in refs),
        f"unexpected Git refs remain: {refs}",
    )

    identities = run("git", "log", "--all", "--format=%an%x00%ae%x00%cn%x00%ce").splitlines()
    for identity in identities:
        fields = identity.split("\x00")
        require(tuple(fields) == (*CANONICAL, *CANONICAL), f"non-canonical reachable identity: {identity}")
    require(not run("git", "status", "--porcelain"), "working tree is not clean")

    print("PASS: required documentation and durable evidence")
    print("PASS: paper identifiers, claim scopes, metrics, and proof certificate")
    print("PASS: final repository, branch, refs, and canonical identities")
    print("PASS: clean working tree")


if __name__ == "__main__":
    main()
