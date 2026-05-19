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
PR-gate's resolver/npm-install/pytest scrubs.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DAILY = REPO_ROOT / "tests" / "claude_code" / "cron_vm" / "run_daily.sh"


def _version_probe_block() -> str:
    body = RUN_DAILY.read_text()
    start = body.index("CLAUDE_CODE_VERSION=")
    end = body.index("[[ -n \"${CLAUDE_CODE_VERSION}\" ]]", start)
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
