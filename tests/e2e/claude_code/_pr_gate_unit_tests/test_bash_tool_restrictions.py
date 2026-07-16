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

CLAUDE_CODE_DIR = Path(__file__).resolve().parents[1]

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
            f"with the layout under tests/e2e/claude_code/."
        )
        for path in sorted(feature_dir.glob("test_*.py")):
            yield path


def _is_exempt_stub(text: str) -> bool:
    """Return True for cells that never drive the `claude` CLI and
    never pass `--allowed-tools`.

    Such a cell (e.g. the static `not_applicable` stubs in the
    `vertex_ai_gpt` column) cannot grant Bash — or any tool — to a
    model-controlled response, so the allow-rule pins below don't
    apply to it. Both conditions are required: a file that references
    `--allowed-tools` without a visible `run_claude` entrypoint is NOT
    exempt and must still carry the pinned shape, so a cell can't dodge
    the scan by hiding its driver behind an indirection.
    """
    return "run_claude" not in text and "--allowed-tools" not in text


def _has_bare_bash_token(text: str) -> bool:
    """Return True if `text` contains a `"Bash"` token outside the
    `"Bash(echo pong)"` allow rule.

    Extracted as a pure helper so the negative path can be unit-tested
    directly. Without it, the previous structure of this assertion was
    `'"Bash"' not in text or '"Bash(echo pong)"' in text`, which
    short-circuits to True any time the allow rule is present and lets
    a stray bare `"Bash"` slip through the security pin undetected.
    """
    return '"Bash"' in text.replace('"Bash(echo pong)"', "")


@pytest.mark.parametrize(
    "cell", list(_bash_cells()), ids=lambda p: str(p.relative_to(CLAUDE_CODE_DIR))
)
def test_bash_allow_rule_is_pinned_to_exact_echo_pong(cell: Path) -> None:
    """The cell must pass `Bash(echo pong)` as the allow rule, not the
    unrestricted `Bash` value that was originally flagged."""
    text = cell.read_text()
    if _is_exempt_stub(text):
        return
    assert '"Bash(echo pong)"' in text, (
        f"{cell.relative_to(CLAUDE_CODE_DIR)} must restrict `--allowed-tools` to "
        f'`Bash(echo pong)` (exact-match pattern). Unrestricted `"Bash"` '
        f"grants arbitrary host command execution to model-controlled "
        f"tool_use blocks, which can read `docker inspect compat-proxy` "
        f"to exfiltrate provider credentials from the proxy container."
    )
    # The only place `"Bash"` (the bare token, surrounded by quotes
    # exactly as it would appear in `--allowed-tools` lists) is allowed
    # to appear is *inside* the exact-match `"Bash(echo pong)"` rule.
    # `_has_bare_bash_token` keeps that scan independent of the first
    # assertion — otherwise `'"Bash"' not in text or '"Bash(echo pong)"'
    # in text` short-circuits to True and lets a stray bare `"Bash"`
    # slip through silently.
    assert not _has_bare_bash_token(text), (
        f"{cell.relative_to(CLAUDE_CODE_DIR)} still references the unrestricted "
        f'`"Bash"` value outside the `"Bash(echo pong)"` allow rule — '
        f"sweep it out before merging."
    )


def test_has_bare_bash_token_flags_unrestricted_value():
    """A file that allows the bare `"Bash"` token alongside the
    exact-match rule must be flagged. Without this guard the security
    pin reverts to the dead-code `or` it had originally, which let
    arbitrary host commands through under the noise of a passing test.
    """
    text = '--allowed-tools "Bash" "Bash(echo pong)"'
    assert _has_bare_bash_token(text)


def test_has_bare_bash_token_accepts_only_exact_match():
    """The standard pattern — only the exact-match allow rule, no bare
    `"Bash"` — must be accepted. This is the shape every Bash-using
    cell in the suite is required to take.
    """
    text = '--allowed-tools "Bash(echo pong)" --permission-mode "dontAsk"'
    assert not _has_bare_bash_token(text)


def test_has_bare_bash_token_ignores_unrelated_substrings():
    """`Bash(echo pong)` is the only allowed shape; substrings like
    `BashTool` or `Bashing` are unrelated identifiers and must not be
    confused with the bare `"Bash"` token (i.e. the exact quoted
    string `"Bash"`)."""
    text = "BashTool helper used by the bashing harness"
    assert not _has_bare_bash_token(text)


@pytest.mark.parametrize(
    "cell", list(_bash_cells()), ids=lambda p: str(p.relative_to(CLAUDE_CODE_DIR))
)
def test_bash_cell_uses_dontask_permission_mode(cell: Path) -> None:
    """The cell must pair the allow rule with `--permission-mode dontAsk`
    so tool calls that don't match the allow rule are auto-denied (as
    opposed to defaulting to "ask", which in headless mode would
    succeed without ever surfacing the security issue)."""
    text = cell.read_text()
    if _is_exempt_stub(text):
        return
    assert '"--permission-mode"' in text and '"dontAsk"' in text, (
        f"{cell.relative_to(CLAUDE_CODE_DIR)} must pass `--permission-mode dontAsk` "
        f"alongside the `Bash(echo pong)` allow rule. Without dontAsk, "
        f"commands outside the allow rule fall back to the default ask-"
        f"mode behavior, which in `--print` (headless) mode is non-"
        f"interactive — defeating the explicit-allow contract."
    )


def test_is_exempt_stub_accepts_not_applicable_stub():
    """A static not_applicable stub (no CLI driver, no tool grants) is
    outside the Bash pin's threat model and must be exempt — this is
    the shape of the `vertex_ai_gpt` cells."""
    text = 'compat_result.set({"status": "not_applicable", "reason": REASON})'
    assert _is_exempt_stub(text)


def test_is_exempt_stub_rejects_cli_driving_cell():
    """Any cell that drives the CLI stays subject to the pins, whether
    or not it currently grants tools."""
    text = (
        "run_claude_models_parallel(models=MODELS, "
        'extra_args=["--allowed-tools", "Bash(echo pong)"])'
    )
    assert not _is_exempt_stub(text)


def test_is_exempt_stub_rejects_allowed_tools_without_visible_driver():
    """A cell that passes `--allowed-tools` while hiding its driver
    behind an indirection must not slip out of the pinned shape."""
    text = 'helper(extra_args=["--allowed-tools", "Bash"])'
    assert not _is_exempt_stub(text)


def test_claude_code_dir_anchor_is_layout_independent() -> None:
    """CLAUDE_CODE_DIR must resolve to the `claude_code/` directory that
    contains this test file, regardless of how deep the repository is
    mounted. The previous anchor `Path(__file__).resolve().parents[4]`
    baked in the host layout (repo root sits four levels up) and broke
    when the suite runs inside the stage container, where tests/e2e/ is
    mounted at /app/e2e/ so `parents[4]` resolves to filesystem root and
    the BASH_FEATURE_DIRS assertion looks for `/tests/e2e/claude_code/
    tool_use`. Anchoring at `parents[1]` (the sibling of this file's
    parent) is the same directory in both layouts.
    """
    assert CLAUDE_CODE_DIR.name == "claude_code"
    assert CLAUDE_CODE_DIR.is_dir()
    assert (CLAUDE_CODE_DIR / "_pr_gate_unit_tests" / Path(__file__).name).is_file()
