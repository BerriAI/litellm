#!/usr/bin/env python3
"""Coverage gate for the proxy_server.py behavior-pinning project.

Reads a coverage XML report (produced by ``pytest --cov-branch
--cov-report=xml:<path>``) and asserts that line + branch coverage on
``litellm/proxy/proxy_server.py`` meets the per-PR target.

Target selection:
    --pr-target {1|2|3}   explicit target
    (none)                self-selected by inspecting which placeholder
                          test files have been filled (PR1 fills before
                          PR2, PR2 before PR3). With nothing filled, the
                          target is "PR0" (baseline, no minimum).

Exits 0 on PASS, non-zero on FAIL.
"""

from __future__ import annotations

import argparse
import ast
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
SOURCE_FILE = "litellm/proxy/proxy_server.py"

# PR target gates: (line%, branch%)
TARGETS: Dict[str, Tuple[float, float]] = {
    "PR0": (0.0, 0.0),
    "PR1": (25.0, 18.0),
    "PR2": (50.0, 38.0),
    "PR3": (70.0, 55.0),
}

# Which placeholder files each PR is expected to fill (see Notion plan).
PR1_FILES: List[str] = [
    "test_lifecycle.py",
    "test_proxy_config.py",
    "test_spend_counters.py",
    "test_background_health.py",
    "test_openapi_customization.py",
    "test_exception_handlers.py",
    "test_streaming_helpers.py",
]
PR2_FILES: List[str] = [
    "test_routes_models.py",
    "test_routes_chat_completions.py",
    "test_routes_completions.py",
    "test_routes_embeddings.py",
    "test_routes_moderations.py",
    "test_routes_audio.py",
    "test_routes_assistants.py",
    "test_routes_threads.py",
    "test_routes_utils.py",
    "test_routes_model_info.py",
    "test_routes_model_metrics.py",
    "test_routes_queue.py",
]
PR3_FILES: List[str] = [
    "test_routes_login_sso.py",
    "test_routes_onboarding.py",
    "test_routes_invitation.py",
    "test_routes_config.py",
    "test_routes_model_cost_map.py",
    "test_routes_anthropic_beta.py",
    "test_routes_misc.py",
]


def file_has_tests(path: Path) -> bool:
    """A test file is considered filled if it defines at least one ``test_*``."""
    if not path.is_file():
        return False
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name.startswith("test_"):
            return True
    return False


def detect_pr_target(dir_path: Path) -> str:
    """Pick the strictest PR whose files are fully filled in this directory."""
    pr3_filled = all(file_has_tests(dir_path / f) for f in PR3_FILES)
    pr2_filled = all(file_has_tests(dir_path / f) for f in PR2_FILES)
    pr1_filled = all(file_has_tests(dir_path / f) for f in PR1_FILES)
    if pr3_filled and pr2_filled and pr1_filled:
        return "PR3"
    if pr2_filled and pr1_filled:
        return "PR2"
    if pr1_filled:
        return "PR1"
    return "PR0"


def parse_coverage_xml(xml_path: Path) -> Tuple[float, float]:
    """Extract (line%, branch%) for proxy_server.py from a coverage XML report.

    Returns (0.0, 0.0) if the file isn't found in the report.
    """
    if not xml_path.is_file():
        raise FileNotFoundError(f"Coverage XML not found at {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for class_elem in root.iter("class"):
        filename = class_elem.get("filename", "")
        # Coverage tools emit either a repo-relative path or just the basename
        # depending on configuration. Match by suffix.
        if filename.endswith("proxy/proxy_server.py") or filename.endswith(
            "proxy_server.py"
        ):
            line_rate = float(class_elem.get("line-rate", "0"))
            branch_rate = float(class_elem.get("branch-rate", "0"))
            return line_rate * 100.0, branch_rate * 100.0
    return 0.0, 0.0


def parse_baseline(baseline_path: Path) -> Tuple[float, float]:
    """Parse ``line:<float> branch:<float>`` baseline; missing file -> (0, 0)."""
    if not baseline_path.is_file():
        return 0.0, 0.0
    line_pct = 0.0
    branch_pct = 0.0
    for token in baseline_path.read_text().split():
        if ":" not in token:
            continue
        key, _, value = token.partition(":")
        try:
            num = float(value)
        except ValueError:
            continue
        if key == "line":
            line_pct = num
        elif key == "branch":
            branch_pct = num
    return line_pct, branch_pct


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pr-target",
        choices=["1", "2", "3"],
        default=None,
        help="Explicit PR target (1, 2, or 3). If omitted, self-selected.",
    )
    parser.add_argument(
        "--coverage-xml",
        default=str(HERE.parent.parent.parent.parent / ".cov_new.xml"),
        help="Path to coverage XML (default: <repo>/.cov_new.xml)",
    )
    args = parser.parse_args()

    if args.pr_target:
        target = f"PR{args.pr_target}"
    else:
        target = detect_pr_target(HERE)
    target_line, target_branch = TARGETS[target]

    # The effective floor is the max of the PR target and the committed
    # baseline. The baseline is updated as each PR lands so a future
    # regression (e.g. a test deletion) trips this gate even if the
    # static PR target is already met.
    baseline_line, baseline_branch = parse_baseline(HERE / ".coverage_baseline")
    line_min = max(target_line, baseline_line)
    branch_min = max(target_branch, baseline_branch)

    xml_path = Path(args.coverage_xml)
    try:
        line_pct, branch_pct = parse_coverage_xml(xml_path)
    except FileNotFoundError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    line_ok = line_pct >= line_min
    branch_ok = branch_pct >= branch_min
    status = "PASS" if (line_ok and branch_ok) else "FAIL"

    print(
        f"target={target} baseline=(line:{baseline_line:.2f} branch:{baseline_branch:.2f})"
    )
    print(
        f"line:   {line_pct:6.2f}% / {line_min:6.2f}% " f"{'OK' if line_ok else 'MISS'}"
    )
    print(
        f"branch: {branch_pct:6.2f}% / {branch_min:6.2f}% "
        f"{'OK' if branch_ok else 'MISS'}"
    )
    print(status)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
