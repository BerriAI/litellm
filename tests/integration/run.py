from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

GROUPS: Final = MappingProxyType(
    {
        "management": ("management", "authorization", "configuration"),
        "accounting": ("pricing", "spend"),
        "database": ("database",),
        "providers": ("providers", "routing", "streaming", "messages_endpoint", "translation"),
        "extensions": ("observability", "compatibility"),
        "mcp": ("mcp",),
        "sdk": ("sdk",),
        "cost": ("cost_calculation",),
        "security": ("security",),
    }
)
GITHUB_FILES: Final = frozenset({"tests/integration/database/test_roi_observed.py"})


@dataclass(frozen=True, slots=True)
class Selection:
    nodes: tuple[str, ...]
    foreign: tuple[str, ...]


def file_of(node: str) -> str:
    return node.split("::", 1)[0]


def select(requested: tuple[str, ...], group_files: tuple[str, ...]) -> Selection:
    members: Final = frozenset(group_files)
    return Selection(
        nodes=requested or group_files,
        foreign=tuple(sorted({node for node in requested if file_of(node) not in members})),
    )


def uncollected(nodes: tuple[str, ...], collected: frozenset[str]) -> tuple[str, ...]:
    collected_files: Final = frozenset(file_of(node) for node in collected)
    return tuple(node for node in nodes if file_of(node) not in collected_files)


def main() -> int:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("group", choices=tuple(GROUPS))
    parser.add_argument("--results", type=Path, default=Path("test-results/integration"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("INTEGRATION_SEED", "4106601")))
    parser.add_argument("--order-seed", type=int, default=int(os.environ.get("INTEGRATION_ORDER_SEED", "0")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("INTEGRATION_WORKERS", "1")))
    parser.add_argument("--list", action="store_true", help="print the group's test files and exit")
    parser.add_argument("files", nargs="*", help="run only these files, or pytest node ids inside them, of the group")
    options: Final = parser.parse_intermixed_args()
    root: Final = Path(__file__).resolve().parents[2]
    group_files: Final = tuple(
        str(path.relative_to(root))
        for folder in GROUPS[options.group]
        for path in sorted((root / "tests/integration" / folder).rglob("test_*.py"))
        if str(path.relative_to(root)) not in GITHUB_FILES
    )
    if options.list:
        print("\n".join(group_files))
        return 0
    selection: Final = select(tuple(options.files), group_files)
    if selection.foreign:
        parser.error(f"Not in the {options.group} group: {', '.join(selection.foreign)}")
    if not selection.nodes:
        parser.error(f"No integration test files selected for {options.group}")
    output: Final = options.results.resolve()
    output.mkdir(parents=True, exist_ok=True)
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
            *selection.nodes,
            "-vv",
            "-rs",
            "--strict-markers",
            "-p",
            "no:pytest-retry",
            "-p",
            "no:rerunfailures",
            "--timeout=90",
            "--durations=15",
            "--tb=short",
            f"--hypothesis-seed={options.seed}",
            f"--integration-order-seed={options.order_seed}",
            f"--junitxml={output / 'junit.xml'}",
            "-o",
            "junit_family=xunit1",
            *(("-n", str(options.workers)) if options.workers > 1 else ()),
        ],
        cwd=root,
        env=environment,
    )
    if result != 0:
        return result
    evidence: Final = json.loads((output / "execution.json").read_text())
    empty: Final = uncollected(selection.nodes, frozenset(evidence["collected"]))
    if empty:
        sys.stderr.write(f"Selected integration files collected zero tests: {', '.join(empty)}\n")
        return 1
    if not evidence["complete"]:
        sys.stderr.write("Integration run did not complete: a collected node neither passed nor skipped\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
