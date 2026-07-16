"""Pin: the cron `pytest` invocation must run under `env -i`.

The systemd service `litellm-compat-matrix.service` loads provider
credentials (`ANTHROPIC_API_KEY`, `AWS_BEARER_TOKEN_BEDROCK`,
`AZURE_FOUNDRY_API_KEY`, `VERTEXAI_*`) and the agent-shin GitHub token
(`AGENT_SHIN_GITHUB_TOKEN`) into `run_daily.sh`'s environment from
`/etc/litellm-compat-matrix.env`. Pytest only needs to talk to the
loopback proxy at `127.0.0.1:${PROXY_PORT}` and has no legitimate reason
to see provider creds in its own `os.environ`. Leaving them in would
let a test under `tests/e2e/claude_code/` read them via `os.environ` and
exfiltrate them, and would also let a model-directed `Read` tool call
during a PDF/vision cell reach `/proc/<pytest-pid>/environ`. The
PR-gate's pytest step in `.circleci/config.yml` already runs under
`env -i`; this pin enforces the same scrub on the cron path.
"""

from __future__ import annotations

from pathlib import Path

RUN_DAILY = Path(__file__).resolve().parents[1] / "cron_vm" / "run_daily.sh"


def _pytest_invocation_block() -> str:
    """Return only the executable lines around the pytest invocation.

    Comment text in run_daily.sh explains *why* certain credential
    names must not appear, so a naïve substring scan over the whole
    region would false-positive on the rationale itself. Strip lines
    whose first non-space character is `#`.
    """
    body = RUN_DAILY.read_text()
    start = body.index('log "running pytest"')
    end = body.index("PYTEST_EXIT=$?", start)
    return "\n".join(
        line for line in body[start:end].splitlines()
        if line.lstrip()[:1] != "#"
    )


def test_pytest_invocation_wraps_in_env_i() -> None:
    block = _pytest_invocation_block()
    assert "env -i" in block, (
        "run_daily.sh: the pytest invocation must run under `env -i` so "
        "PR-controlled test code under tests/e2e/claude_code/ cannot read "
        "provider/agent-shin credentials out of the systemd service "
        "environment, and so a model-directed `Read` tool call cannot "
        "reach /proc/<pytest-pid>/environ to pull them out."
    )
    assert block.index("env -i") < block.index('"${WORKTREE_UV}" run pytest'), (
        "run_daily.sh: `env -i` must precede the pytest invocation; "
        "otherwise pytest inherits the full credential-bearing env."
    )


def test_pytest_invocation_env_i_excludes_provider_secrets() -> None:
    block = _pytest_invocation_block()
    for forbidden in (
        "ANTHROPIC_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "VERTEXAI_CREDENTIALS",
        "VERTEXAI_PROJECT",
        "VERTEXAI_LOCATION",
        "AZURE_FOUNDRY_API_KEY",
        "AZURE_FOUNDRY_API_BASE",
        "GITHUB_TOKEN",
        "AGENT_SHIN_GITHUB_TOKEN",
    ):
        assert forbidden not in block, (
            f"run_daily.sh: the pytest-step `env -i` allowlist must not "
            f"pass {forbidden} through. Found it inside the pytest "
            f"invocation block."
        )


def test_pytest_invocation_passes_proxy_url_and_key_explicitly() -> None:
    block = _pytest_invocation_block()
    assert "LITELLM_PROXY_BASE_URL=" in block, (
        "run_daily.sh: the pytest `env -i` block must still pass "
        "LITELLM_PROXY_BASE_URL so the test suite knows where to find "
        "the loopback proxy."
    )
    assert "LITELLM_PROXY_API_KEY=" in block, (
        "run_daily.sh: the pytest `env -i` block must still pass "
        "LITELLM_PROXY_API_KEY so the test suite can authenticate to "
        "the loopback proxy."
    )
    assert "COMPAT_RESULTS_PATH=" in block, (
        "run_daily.sh: the pytest `env -i` block must still pass "
        "COMPAT_RESULTS_PATH so the conftest writes the per-cell "
        "tagged-union artifact to the script-managed path."
    )
