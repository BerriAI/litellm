import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


BUDGET_PATH = Path("ci_cd/root-size-budget.json")


def _git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _tracked_files() -> list[str]:
    result = _git(["ls-files"])
    return [line for line in result.stdout.splitlines() if line]


def current_root_metrics() -> dict[str, int]:
    root_files: list[str] = []
    root_dirs: set[str] = set()

    for file_path in _tracked_files():
        parts = Path(file_path).parts
        if len(parts) == 1:
            root_files.append(file_path)
        else:
            root_dirs.add(parts[0])

    return {
        "max_tracked_root_dirs": len(root_dirs),
        "max_tracked_root_entries": len(root_files) + len(root_dirs),
        "max_tracked_root_files": len(root_files),
    }


def load_budget() -> dict[str, int]:
    with BUDGET_PATH.open() as budget_file:
        budget = json.load(budget_file)
    return {key: int(value) for key, value in budget.items()}


def load_base_budget(base_ref: str | None) -> dict[str, int] | None:
    if not base_ref:
        return None

    result = _git(["show", f"{base_ref}:{BUDGET_PATH}"], check=False)
    if result.returncode != 0:
        return None
    parsed: dict[str, Any] = json.loads(result.stdout)
    return {key: int(value) for key, value in parsed.items()}


def check_current_budget(metrics: dict[str, int], budget: dict[str, int]) -> list[str]:
    errors: list[str] = []
    for key, actual_value in metrics.items():
        budget_value = budget.get(key)
        if budget_value is None:
            errors.append(f"{key}: missing from {BUDGET_PATH}")
            continue
        if actual_value > budget_value:
            errors.append(f"{key}: actual {actual_value} exceeds budget {budget_value}")
    return errors


def check_budget_ratchet(
    budget: dict[str, int], base_budget: dict[str, int] | None
) -> list[str]:
    if base_budget is None:
        return []

    errors: list[str] = []
    for key, budget_value in budget.items():
        base_value = base_budget.get(key)
        if base_value is None:
            continue
        if budget_value > base_value:
            errors.append(
                f"{key}: budget increased from {base_value} to {budget_value}"
            )
    return errors


def emit(message: str = "") -> None:
    sys.stdout.write(f"{message}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensure tracked repo-root file/dir counts do not grow."
    )
    parser.add_argument("--base", help="Optional base ref for budget ratcheting")
    args = parser.parse_args()

    metrics = current_root_metrics()
    budget = load_budget()
    base_budget = load_base_budget(args.base)

    errors = check_current_budget(metrics, budget)
    errors.extend(check_budget_ratchet(budget, base_budget))

    emit("Current root metrics:")
    for key in sorted(metrics):
        emit(f"  {key}: {metrics[key]} / {budget.get(key, 'missing')}")

    if errors:
        emit("\nRoot budget check failed:")
        for error in errors:
            emit(f"  - {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
