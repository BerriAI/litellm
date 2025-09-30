"""Lean4 + LiteLLM integration recipe.

This sample shows how to invoke the Lean4 prover CLI (`cli_mini`) as an
external process, then optionally use LiteLLM to summarise the batch results.
It treats Lean4 as a companion sidecar service—perfect for pairing with
mini-agent or other LiteLLM providers.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from dotenv import find_dotenv, load_dotenv

try:
    import litellm
except ImportError:
    litellm = None  # type: ignore

load_dotenv(find_dotenv())

LEAN4_REPO = Path(os.getenv("LEAN4_REPO", "/home/graham/workspace/experiments/lean4"))
if not LEAN4_REPO.exists():
    print(f"LEAN4_REPO not found at {LEAN4_REPO}. Set the env var before running.")
    sys.exit(1)

DEFAULT_CMD = "python -m lean4_prover.cli_mini batch"
CLI_CMD = os.getenv("LEAN4_CLI_CMD", DEFAULT_CMD)
SUMMARY_MODEL = os.getenv("LEAN4_SUMMARY_MODEL") or os.getenv("LITELLM_DEFAULT_MODEL")
EXTRA_FLAGS = os.getenv("LEAN4_EXTRA_FLAGS", "--deterministic")

# Minimal example requirement; override by pointing LEAN4_INPUT_JSON to your own file.
DEFAULT_ITEMS: List[Dict[str, Any]] = [
    {
        "requirement_text": "Prove that for all natural numbers n, (n + 0) = n.",
        "context": {"section_id": "demo_nat_add_zero"},
    }
]

INPUT_OVERRIDE = os.getenv("LEAN4_INPUT_JSON")


def build_input() -> Path:
    if INPUT_OVERRIDE:
        p = Path(INPUT_OVERRIDE)
        if not p.exists():
            raise FileNotFoundError(f"LEAN4_INPUT_JSON points to missing file: {p}")
        return p
    tmp = tempfile.NamedTemporaryFile("w", suffix="_lean4_input.json", delete=False)
    json.dump(DEFAULT_ITEMS, tmp, indent=2)
    tmp.flush()
    Path(tmp.name).chmod(0o600)
    return Path(tmp.name)


def build_output_path() -> Path:
    tmp = tempfile.NamedTemporaryFile("w", suffix="_lean4_output.json", delete=False)
    tmp.close()
    return Path(tmp.name)


def run_lean4(input_path: Path, output_path: Path) -> None:
    args = shlex.split(CLI_CMD)
    flags = shlex.split(EXTRA_FLAGS) if EXTRA_FLAGS else []
    command = args + ["--input-file", str(input_path), "--output-file", str(output_path)] + flags
    print("Running:", " ".join(command))
    completed = subprocess.run(command, cwd=LEAN4_REPO, capture_output=True, text=True)
    if completed.returncode != 0:
        print("Lean4 CLI stderr:\n", completed.stderr)
        print("Lean4 CLI stdout:\n", completed.stdout)
        raise RuntimeError(f"Lean4 CLI failed with exit code {completed.returncode}")
    if completed.stderr.strip():
        print("[Lean4 stderr]", completed.stderr.strip())
    if completed.stdout.strip():
        print("[Lean4 stdout]", completed.stdout.strip())


def load_scorecard(output_path: Path) -> Dict[str, Any]:
    if not output_path.exists():
        raise FileNotFoundError(f"Lean4 output file not found: {output_path}")
    return json.loads(output_path.read_text())


def summarise_with_litellm(scorecard: Dict[str, Any]) -> str:
    if not litellm or not SUMMARY_MODEL:
        return "(LiteLLM summary skipped: install litellm and/or set LEAN4_SUMMARY_MODEL)"
    prompt = (
        "You are a formal methods engineer. Summarise the Lean4 proof batch. "
        "Mention total items, proved/unproved counts, and highlight any failures with section_id."
    )
    response = litellm.completion(
        model=SUMMARY_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(scorecard, indent=2)},
        ],
    )
    try:
        return response["choices"][0]["message"]["content"]  # type: ignore[index]
    except Exception:  # noqa: BLE001
        return str(response)


def main() -> None:
    input_path = build_input()
    output_path = build_output_path()
    try:
        run_lean4(input_path, output_path)
        scorecard = load_scorecard(output_path)
        print(json.dumps({"scorecard": scorecard}, indent=2))
        summary = summarise_with_litellm(scorecard)
        print("\n=== LiteLLM Summary ===\n")
        print(summary)
    finally:
        if not INPUT_OVERRIDE and input_path.exists():
            try:
                input_path.unlink()
            except Exception:
                pass
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()
