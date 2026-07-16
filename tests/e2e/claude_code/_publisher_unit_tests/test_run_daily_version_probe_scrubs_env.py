"""Pin: the cron `claude --version` probe must run under `env -i`.

The systemd service `litellm-compat-matrix.service` loads provider
credentials (`ANTHROPIC_API_KEY`, `AWS_BEARER_TOKEN_BEDROCK`,
`AZURE_FOUNDRY_API_KEY`) and the agent-shin GitHub token
(`AGENT_SHIN_GITHUB_TOKEN`) into `run_daily.sh`'s environment from
`/etc/litellm-compat-matrix.env`. Running the npm-installed `claude`
binary directly there would hand that full env to package code, so a
compromised `@anthropic-ai/claude-code` release could read those
secrets out of `os.environ` before the proxy or test harness ever
starts. The version probe must be wrapped in `env -i` with a minimal
PATH/HOME/USER/TERM/LANG/LC_ALL/TMPDIR allowlist — matching the
resolver/npm-install/pytest scrubs in the same script.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_DAILY = REPO_ROOT / "tests" / "e2e" / "claude_code" / "cron_vm" / "run_daily.sh"


def _anchor(text: str, needle: str, start: int = 0) -> int:
    found = text.find(needle, start)
    if found == -1:
        pytest.fail(f"run_daily.sh anchor not found: {needle!r}")
    return found


def _version_probe_block() -> str:
    body = RUN_DAILY.read_text()
    start = _anchor(body, "PROBED_CLAUDE_VERSION=")
    end = _anchor(body, '[[ -n "${PROBED_CLAUDE_VERSION}" ]]', start)
    return body[start:end]


def test_version_probe_wraps_claude_in_env_i() -> None:
    block = _version_probe_block()
    assert "env -i" in block, (
        "run_daily.sh: the `claude --version` probe must run under "
        "`env -i` so a compromised @anthropic-ai/claude-code package "
        "cannot read provider/GitHub credentials out of the systemd "
        "service environment."
    )
    assert block.index("env -i") < block.index("claude --version"), (
        "run_daily.sh: `env -i` must precede `claude --version`; "
        "otherwise the binary inherits the full credential-bearing env."
    )


def test_version_probe_env_i_excludes_provider_secrets() -> None:
    block = _version_probe_block()
    for forbidden in (
        "ANTHROPIC_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "VERTEXAI_CREDENTIALS",
        "AZURE_FOUNDRY_API_KEY",
        "GITHUB_TOKEN",
        "AGENT_SHIN_GITHUB_TOKEN",
    ):
        assert forbidden not in block, (
            f"run_daily.sh: the version-probe `env -i` allowlist must "
            f"not pass {forbidden} through. Found it inside the probe "
            f"block."
        )


def test_version_probe_uses_isolated_home_not_runtime_user_home() -> None:
    """Pin: the `claude --version` probe runs under a fresh empty HOME.

    `ProtectHome=read-only` in the systemd unit allows reads of the
    runtime user's real home directory. If the probe's `env -i`
    block forwards `HOME=${HOME}`, a compromised `claude` package
    can `os.path.expanduser("~/.config/gh/hosts.yml")` or
    `os.path.expanduser("~/.ssh/...")` and exfiltrate the contents
    before the proxy or test harness ever starts. The probe must
    point HOME at a per-run tmpdir under `${WORKDIR}` so the CLI
    sees an empty HOME instead.
    """
    block = _version_probe_block()
    body = RUN_DAILY.read_text()

    assert "CLAUDE_PROBE_HOME=" in body, (
        "run_daily.sh: must define a `CLAUDE_PROBE_HOME` per-run tmpdir "
        "for the `claude --version` probe so the CLI never sees the "
        "runtime user's real $HOME."
    )
    assert 'HOME="${CLAUDE_PROBE_HOME}"' in block, (
        "run_daily.sh: the probe's `env -i` block must set HOME to "
        "the per-run isolated tmpdir, not to the runtime user's $HOME."
    )
    assert 'HOME="${HOME}"' not in block, (
        "run_daily.sh: the probe's `env -i` block must not forward the "
        "runtime user's $HOME to `claude --version`. Use the isolated "
        "$CLAUDE_PROBE_HOME tmpdir instead."
    )
