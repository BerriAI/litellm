"""The path the Buildkite runner images run (`python3 tests/e2e/claude_code/pr_gate_version_resolver.py`) to pick the
Claude Code release they install. They fall back to @latest when this file is missing, so keep it until they call
`cli_harness/npm_version_resolver.py` themselves."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Final

_RESOLVER: Final = Path(__file__).resolve().parent.parent / "cli_harness" / "npm_version_resolver.py"
_PACKAGE: Final = "@anthropic-ai/claude-code"

if __name__ == "__main__":
    raise SystemExit(subprocess.run((sys.executable, "-I", str(_RESOLVER), _PACKAGE), check=False).returncode)
