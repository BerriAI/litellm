from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Final

GROUPS: Final = MappingProxyType(json.loads(Path(__file__).with_name("contracts.json").read_text())["groups"])


def main() -> int:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("group", choices=tuple(GROUPS))
    parser.add_argument("--results", type=Path, default=Path("test-results/integration"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("INTEGRATION_SEED", "4106601")))
    parser.add_argument("--order-seed", type=int, default=int(os.environ.get("INTEGRATION_ORDER_SEED", "0")))
    options: Final = parser.parse_args()
    root: Final = Path(__file__).resolve().parents[2]
    selected: Final = tuple(
        str(path.relative_to(root))
        for folder in GROUPS[options.group]
        for path in sorted((root / "tests/integration" / folder).glob("test_*.py"))
    )
    if not selected:
        parser.error(f"No integration contracts selected for {options.group}")
    output: Final = options.results.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest: Final = json.loads((root / "tests/integration/contracts.json").read_text())["tests"]
    expected: Final = sorted(node for node in manifest if node.split("::", 1)[0] in selected)
    if not expected or set(selected) != {node.split("::", 1)[0] for node in expected}:
        parser.error("Every selected file must have canonical manifest nodes")
    environment: Final = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(root), str(root / "tests"), str(root / "tests/e2e"))),
        "INTEGRATION_RESULTS_DIR": str(output),
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    result: Final = subprocess.call(
        [
            sys.executable,
            "-m",
            "pytest",
            *selected,
            "-vv",
            "--strict-markers",
            "-p",
            "no:pytest-retry",
            "-p",
            "no:rerunfailures",
            "--timeout=90",
            "--durations=15",
            f"--hypothesis-seed={options.seed}",
            f"--integration-order-seed={options.order_seed}",
            f"--junitxml={output / 'junit.xml'}",
        ],
        cwd=root,
        env=environment,
    )
    if result != 0:
        return result
    evidence: Final = json.loads((output / "execution.json").read_text())
    if not evidence["complete"] or sorted(evidence["passed"]) != expected or sorted(evidence["collected"]) != expected:
        print("Executed integration nodes differ from the canonical manifest", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
