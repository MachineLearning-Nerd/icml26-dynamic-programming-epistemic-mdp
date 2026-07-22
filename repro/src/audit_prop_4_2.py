#!/usr/bin/env python3
"""Independent, fail-closed audit of the Proposition 4.2 evidence.

The primary focused proof certificate is ``verify_prop_4_2.py``. This auditor
deliberately does not use its DAG validator: it independently checks that every
dependency precedes its consumer, runs three mutation controls, and, when the
separate scaled value/policy-iteration result is locally available, validates
that corroborating output too. The proof audit remains self-contained when the
optional scale JSON is absent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).absolute().parent
ROOT = SCRIPT_DIR.parents[1]
SCALE_RESULT = ROOT / "outputs" / "scale_audit_c1.json"
sys.path.insert(0, str(SCRIPT_DIR))
import verify_prop_4_2 as primary  # noqa: E402

REQUIRED_STEPS = {
    "assumptions",
    "bellman_structure",
    "fixed_points",
    "vi_iterate_rate",
    "vi_greedy_residual",
    "vi_policy_rate",
    "pi_improvement",
    "stationary_optimum",
    "pi_one_step_rate",
    "pi_rate",
    "complete",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_dag_check(steps: list[dict], conclusion: str) -> bool:
    """Validate order and closure without using the primary validator."""
    seen: set[str] = set()
    for step in steps:
        step_id = step.get("id")
        dependencies = step.get("depends_on", [])
        if not isinstance(step_id, str) or not step_id or step_id in seen:
            return False
        if not isinstance(dependencies, list) or any(d not in seen for d in dependencies):
            return False
        seen.add(step_id)
    return REQUIRED_STEPS <= seen and conclusion in seen


def main() -> None:
    primary_result = primary.verify()
    steps = primary_result["steps"]
    primary_valid = independent_dag_check(steps, "complete")

    missing_rate = [s for s in steps if s["id"] != "pi_rate"]
    duplicate = steps + [dict(steps[-1])]
    reordered = [steps[-1], *steps[:-1]]
    mutation_controls = {
        "missing_policy_rate_rejected": not independent_dag_check(missing_rate, "complete"),
        "duplicate_step_rejected": not independent_dag_check(duplicate, "complete"),
        "forward_dependency_rejected": not independent_dag_check(reordered, "complete"),
    }

    scale_checks = {"available": SCALE_RESULT.is_file()}
    if SCALE_RESULT.is_file():
        scale = json.loads(SCALE_RESULT.read_text())
        vi_pi = scale["S1_S2_vi_pi"]
        scale_checks.update({
            "algorithm1_bound_checks": vi_pi["algorithm1_bound_checks"],
            "algorithm1_bound_violations": vi_pi["algorithm1_bound_violations"],
            "algorithm2_rate_checks": vi_pi["algorithm2_rate_checks"],
            "algorithm2_violations": vi_pi["algorithm2_violations"],
            "algorithm2_terminated_runs": vi_pi["algorithm2_terminated_runs"],
            "algorithm2_max_policy_updates": vi_pi["algorithm2_max_policy_updates"],
            "sha256": sha256(SCALE_RESULT),
        })
        assert scale_checks["algorithm1_bound_checks"] > 0
        assert scale_checks["algorithm1_bound_violations"] == 0
        assert scale_checks["algorithm2_rate_checks"] > 0
        assert scale_checks["algorithm2_violations"] == 0
        assert scale_checks["algorithm2_terminated_runs"] == vi_pi["attitude_runs"]

    all_passed = primary_result["all_passed"] and primary_valid and all(mutation_controls.values())
    result = {
        "claim": "Official Claim 2 / Proposition 4.2",
        "all_passed": all_passed,
        "primary_all_passed": primary_result["all_passed"],
        "independent_dag_valid": primary_valid,
        "mutation_controls": mutation_controls,
        "exact_fraction_checks": primary_result["exact_fraction_checks"],
        "scale_checks": scale_checks,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
