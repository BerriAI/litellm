"""Pin tests for the `Bash`-using compat cells.

Every cell that passes `--allowed-tools Bash` to the `claude` CLI is
giving a model-controlled response the ability to run host commands.
On the PR-gate CircleCI machine executor, those commands have access
to the Docker socket and can read `docker inspect compat-proxy` to
recover the provider credentials living inside the proxy container.

To narrow that surface, every Bash-using cell must:

1. Restrict the allow rule to the *exact* command `Bash(echo pong)` so
   a compromised provider response cannot turn `Bash` into arbitrary
   host execution by emitting a `tool_use` with a different command.

2. Pair it with `--permission-mode dontAsk` so anything not matching
   an allow rule is auto-denied instead of prompting (which would
   abort the CLI in headless mode, but auto-denial is the explicit
   contract).

These restrictions are enforced by the `claude` CLI, not by the
model — see https://code.claude.com/docs/en/permissions for the
permission-rule precedence (`deny` → `ask` → `allow`).

This test scans every cell under the three Bash-using feature
directories (`tool_use`, `tool_use_streaming`, `thinking_with_tool_use`)
and pins both requirements so a future test refactor cannot silently
revert any cell to the broad `Bash` allow that was originally
flagged by Veria.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CLAUDE_CODE_DIR = REPO_ROOT / "tests" / "claude_code"

# Feature directories whose cells drive the `Bash` built-in tool. Add
# new entries here when a new Bash-using feature is added; the test
# fails loudly for any unhandled directory so we never miss one by
# silent omission.
BASH_FEATURE_DIRS = (
    "tool_use",
    "tool_use_streaming",
    "thinking_with_tool_use",
)


def _bash_cells() -> Iterable[Path]:
    for feature in BASH_FEATURE_DIRS:
        feature_dir = CLAUDE_CODE_DIR / feature
        assert feature_dir.is_dir(), (
            f"{feature_dir} is missing — BASH_FEATURE_DIRS is out of sync "
            f"with the layout under tests/claude_code/."
        )
        for path in sorted(feature_dir.glob("test_*.py")):
            yield path


@pytest.mark.parametrize(
    "cell", list(_bash_cells()), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_bash_allow_rule_is_pinned_to_exact_echo_pong(cell: Path) -> None:
    """The cell must pass `Bash(echo pong)` as the allow rule, not the
    unrestricted `Bash` value that was originally flagged."""
    text = cell.read_text()
    assert '"Bash(echo pong)"' in text, (
        f"{cell.relative_to(REPO_ROOT)} must restrict `--allowed-tools` to "
        f'`Bash(echo pong)` (exact-match pattern). Unrestricted `"Bash"` '
        f"grants arbitrary host command execution to model-controlled "
        f"tool_use blocks, which can read `docker inspect compat-proxy` "
        f"to exfiltrate provider credentials from the proxy container."
    )
    assert '"Bash"' not in text or '"Bash(echo pong)"' in text, (
        f"{cell.relative_to(REPO_ROOT)} still references the unrestricted "
        f'`"Bash"` value somewhere — sweep it out before merging.'
    )


@pytest.mark.parametrize(
    "cell", list(_bash_cells()), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_bash_cell_uses_dontask_permission_mode(cell: Path) -> None:
    """The cell must pair the allow rule with `--permission-mode dontAsk`
    so tool calls that don't match the allow rule are auto-denied (as
    opposed to defaulting to "ask", which in headless mode would
    succeed without ever surfacing the security issue)."""
    text = cell.read_text()
    assert '"--permission-mode"' in text and '"dontAsk"' in text, (
        f"{cell.relative_to(REPO_ROOT)} must pass `--permission-mode dontAsk` "
        f"alongside the `Bash(echo pong)` allow rule. Without dontAsk, "
        f"commands outside the allow rule fall back to the default ask-"
        f"mode behavior, which in `--print` (headless) mode is non-"
        f"interactive — defeating the explicit-allow contract."
    )
