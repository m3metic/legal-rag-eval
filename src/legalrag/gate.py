"""Evaluation gate: fail a change if retrieval quality drops.

The gate file (configs/gate.yaml) lists experiment configs and the scores each one
is expected to reach. `python -m legalrag gate` reruns them, compares every metric
with its baseline and exits with status 1 if any falls by more than the tolerance.
In CI this blocks a pull request that makes retrieval worse.

When a change is meant to move the numbers (a better chunker, a fixed bug), rerun
with --update to write the new scores into the gate file and commit it, so the
improvement becomes the new floor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from .experiment import run


@dataclass
class Check:
    experiment: str
    metric: str
    baseline: float
    current: float
    tolerance: float

    @property
    def delta(self) -> float:
        return self.current - self.baseline

    @property
    def passed(self) -> bool:
        return self.delta >= -self.tolerance


def compare_to_baseline(
    experiment: str, metrics: dict[str, float], baseline: dict[str, float], tolerance: float
) -> list[Check]:
    """One Check per baseline metric. A metric missing from the run counts as 0, so it fails."""
    return [
        Check(experiment, name, float(base), float(metrics.get(name, 0.0)), tolerance)
        for name, base in baseline.items()
    ]


def report(checks: list[Check]) -> str:
    """Markdown table of baseline vs current, one row per experiment and metric."""
    lines = [
        "| experiment | metric | baseline | current | change | |",
        "|---|---|---|---|---|---|",
    ]
    for c in checks:
        mark = "✅" if c.passed else "❌"
        lines.append(
            f"| {c.experiment} | {c.metric} | {c.baseline:.4f} | {c.current:.4f} | {round(c.delta, 4) + 0.0:+.4f} | {mark} |"
        )
    failed = [c for c in checks if not c.passed]
    verdict = (
        f"**Gate failed:** {len(failed)} of {len(checks)} checks dropped by more than the tolerance."
        if failed
        else f"**Gate passed:** all {len(checks)} checks are at or above baseline (within tolerance)."
    )
    return "\n".join(lines) + "\n\n" + verdict


def gate(gate_path: str | Path, out_root: str | Path = "results/gate", update: bool = False) -> bool:
    """Run every experiment in the gate file. Returns True if all checks pass."""
    gate_path = Path(gate_path)
    spec = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    tolerance = float(spec.get("tolerance", 0.005))

    checks: list[Check] = []
    for exp in spec["experiments"]:
        result = run(exp["config"], limit=exp.get("limit"), out_root=out_root)
        if update:
            exp["baseline"] = {m: round(result["metrics"][m], 4) for m in exp["baseline"]}
        checks += compare_to_baseline(result["name"], result["metrics"], exp["baseline"], tolerance)

    text = report(checks)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:  # shows the table on the GitHub Actions run page
        with open(summary, "a", encoding="utf-8") as f:
            f.write("## Retrieval evaluation gate\n\n" + text + "\n")

    if update:
        gate_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        print(f"\nBaselines updated in {gate_path}")
        return True
    return all(c.passed for c in checks)
