"""Pin: run_daily.sh must install the resolver-selected claude-code version.

PR #32548 deleted the CircleCI job `claude_code_compat_pr_gate`, which
was the only automated consumer of `pr_gate_version_resolver.py` (newest
`@anthropic-ai/claude-code` npm version published at least 3 days ago; a
security-review buffer, PRD #26476). The daily runner used to probe
whatever `claude` happened to be on the cron VM's PATH, so nothing
automated enforced the buffer anymore. These tests pin the replacement
flow in run_daily.sh:

  1. The target version comes from running `pr_gate_version_resolver.py`
     and exactly `@anthropic-ai/claude-code@<resolved>` is npm-installed
     into a per-run prefix under `${WORKDIR}`.
  2. Both the resolver and the npm install run under the same `env -i`
     credential scrub (with the isolated per-run HOME) as the probe and
     pytest steps: npm postinstall runs package code, which is exactly
     the supply-chain vector the 3-day buffer exists for.
  3. The `claude --version` probe verifies the binary now on PATH
     reports exactly the resolved version and dies on a mismatch.
  4. The install's bin dir reaches PATH only inside the two `env -i`
     blocks that spawn `claude` (probe and pytest), never at script
     scope where later git/gh/curl/uv steps run with the token env.
  5. The npm install and the probe additionally run inside an
     unprivileged user+pid namespace with a fresh /proc: env scrubbing
     alone is not a boundary because same-uid package code can read
     the secret-bearing parent's /proc/<pid>/environ.

The resolve/install/probe block is extracted out of run_daily.sh and
executed with a stub resolver and a fake `npm`, mirroring the fake-curl
technique in `test_run_daily_release_pagination.py`.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_DAILY = REPO_ROOT / "tests" / "e2e" / "claude_code" / "cron_vm" / "run_daily.sh"

RESOLVED_VERSION = "0.0.99"

_PREAMBLE = (
    "set -Eeuo pipefail\n"
    "log() { printf '==> %s\\n' \"$*\" >&2; }\n"
    "die() { printf 'ERROR: %s\\n' \"$*\" >&2; exit 1; }\n"
)

_SECRET_ENV = {
    "ANTHROPIC_API_KEY": "test-secret-anthropic",
    "AWS_BEARER_TOKEN_BEDROCK": "test-secret-bedrock",
    "AZURE_FOUNDRY_API_KEY": "test-secret-foundry",
    "AGENT_SHIN_GITHUB_TOKEN": "test-secret-agent-shin",
    "GITHUB_TOKEN": "test-secret-github",
}


def _anchor(text: str, needle: str, start: int = 0) -> int:
    found = text.find(needle, start)
    if found == -1:
        pytest.fail(f"run_daily.sh anchor not found: {needle!r}")
    return found


def _extract_block(text: str, start_marker: str, end_marker: str) -> str:
    start = _anchor(text, start_marker)
    end = _anchor(text, end_marker, start)
    return text[start:end]


def _extract_pin_snippet() -> str:
    return _extract_block(
        RUN_DAILY.read_text(),
        'CLAUDE_PROBE_HOME="${WORKDIR}/claude-probe-home"',
        'log "pinned claude code:',
    )


def _executable_lines(block: str) -> str:
    return "\n".join(
        line for line in block.splitlines() if line.lstrip()[:1] != "#"
    )


@dataclass(frozen=True, slots=True)
class PinHarness:
    workdir: Path
    resolver_env_json: Path
    npm_env_txt: Path
    npm_argv_txt: Path
    script: str
    env: dict[str, str]

    def run(self) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(
            ["bash", "-c", self.script],
            capture_output=True,
            text=True,
            env=self.env,
        )


def _build_harness(tmp_path: Path, installed_version: str) -> PinHarness:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    dumps = tmp_path / "dumps"
    dumps.mkdir()
    workdir = tmp_path / "work"
    workdir.mkdir()
    populator_dir = tmp_path / "claude_code" / "cron_vm"
    populator_dir.mkdir(parents=True)

    resolver_env_json = dumps / "resolver_env.json"
    resolver = tmp_path / "claude_code" / "pr_gate_version_resolver.py"
    resolver.write_text(
        textwrap.dedent(
            f"""\
            import json
            import os

            with open({str(resolver_env_json)!r}, "w") as fh:
                json.dump(dict(os.environ), fh)
            print({RESOLVED_VERSION!r})
            """
        )
    )

    npm_env_txt = dumps / "npm_env.txt"
    npm_argv_txt = dumps / "npm_argv.txt"
    fake_npm = fake_bin / "npm"
    fake_npm.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            printf '%s\\n' "$@" > "{npm_argv_txt}"
            env > "{npm_env_txt}"
            prefix=""
            pkg=""
            while [[ $# -gt 0 ]]; do
              case "$1" in
                --prefix) prefix="$2"; shift 2 ;;
                install) shift ;;
                *) pkg="$1"; shift ;;
              esac
            done
            mkdir -p "${{prefix}}/node_modules/.bin"
            cat > "${{prefix}}/node_modules/.bin/claude" <<'CLAUDE_EOF'
            #!/usr/bin/env bash
            echo "{installed_version} (Claude Code)"
            CLAUDE_EOF
            chmod +x "${{prefix}}/node_modules/.bin/claude"
            """
        )
    )
    fake_npm.chmod(0o755)

    fake_unshare = fake_bin / "unshare"
    fake_unshare.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            while [[ "${1:-}" == --* ]]; do shift; done
            exec "$@"
            """
        )
    )
    fake_unshare.chmod(0o755)

    script = (
        _PREAMBLE
        + f'WORKDIR="{workdir}"\n'
        + f'POPULATOR_DIR="{populator_dir}"\n'
        + _extract_pin_snippet()
        + 'printf "%s" "${CLAUDE_CODE_VERSION}"\n'
    )
    env = {
        **os.environ,
        **_SECRET_ENV,
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
    }
    return PinHarness(
        workdir=workdir,
        resolver_env_json=resolver_env_json,
        npm_env_txt=npm_env_txt,
        npm_argv_txt=npm_argv_txt,
        script=script,
        env=env,
    )


def test_run_daily_installs_exactly_the_resolver_selected_version(
    tmp_path: Path,
) -> None:
    harness = _build_harness(tmp_path, installed_version=RESOLVED_VERSION)
    result = harness.run()
    assert result.returncode == 0, (
        f"pin snippet failed: stderr={result.stderr!r}"
    )
    assert result.stdout == RESOLVED_VERSION, (
        f"CLAUDE_CODE_VERSION must be the resolver's output, got "
        f"{result.stdout!r}"
    )
    argv = harness.npm_argv_txt.read_text().splitlines()
    assert f"@anthropic-ai/claude-code@{RESOLVED_VERSION}" in argv, (
        f"run_daily.sh must `npm install` the exact resolver-selected "
        f"version; npm argv was {argv!r}"
    )
    assert "install" in argv
    prefix = argv[argv.index("--prefix") + 1]
    assert Path(prefix) == harness.workdir / "claude-cli", (
        "the per-run install prefix must live inside ${WORKDIR} so the "
        "cleanup trap removes it"
    )
    assert (
        harness.workdir / "claude-cli" / "node_modules" / ".bin" / "claude"
    ).exists()


def test_resolver_and_npm_install_run_under_env_scrub(tmp_path: Path) -> None:
    harness = _build_harness(tmp_path, installed_version=RESOLVED_VERSION)
    result = harness.run()
    assert result.returncode == 0, (
        f"pin snippet failed: stderr={result.stderr!r}"
    )

    resolver_env = json.loads(harness.resolver_env_json.read_text())
    npm_env = dict(
        line.split("=", 1)
        for line in harness.npm_env_txt.read_text().splitlines()
        if "=" in line
    )
    probe_home = str(harness.workdir / "claude-probe-home")
    for name, seen_env in (("resolver", resolver_env), ("npm", npm_env)):
        for secret in _SECRET_ENV:
            assert secret not in seen_env, (
                f"run_daily.sh: the {name} step leaked {secret} through "
                f"its `env -i` scrub. npm postinstall runs package code, "
                f"which is exactly the supply-chain vector the 3-day "
                f"buffer exists for."
            )
        assert seen_env.get("HOME") == probe_home, (
            f"run_daily.sh: the {name} step must run under the isolated "
            f"per-run HOME, not the runtime user's; got "
            f"{seen_env.get('HOME')!r}"
        )


def test_probe_dies_when_installed_version_mismatches_resolved(
    tmp_path: Path,
) -> None:
    harness = _build_harness(tmp_path, installed_version="9.9.9")
    result = harness.run()
    assert result.returncode != 0, (
        "run_daily.sh must die when the claude binary on PATH does not "
        "report the resolver-selected version; a silent pass here means "
        "the 3-day security buffer is not actually enforced."
    )
    assert "9.9.9" in result.stderr
    assert RESOLVED_VERSION in result.stderr


def test_static_resolver_and_install_are_env_i_wrapped() -> None:
    body = RUN_DAILY.read_text()
    lines = _executable_lines(_extract_pin_snippet())
    resolver_call = _anchor(lines, "pr_gate_version_resolver.py")
    assert "env -i" in lines[:resolver_call], (
        "run_daily.sh must run pr_gate_version_resolver.py under `env -i`"
    )
    install_call = _anchor(lines, "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}")
    assert "env -i" in lines[resolver_call:install_call], (
        "run_daily.sh must run the npm install under its own `env -i`"
    )
    assert "claude" not in _required_commands(body), (
        "run_daily.sh must not require a pre-provisioned `claude` on "
        "PATH anymore; the pinned install provides it"
    )
    for required in ("npm", "python3"):
        assert required in _required_commands(body)


def test_static_npm_bin_dir_scoped_to_probe_and_pytest_env_blocks() -> None:
    body = RUN_DAILY.read_text()
    _anchor(body, 'CLAUDE_CLI_BIN="${CLAUDE_CLI_PREFIX}/node_modules/.bin"')
    prepend = 'PATH="${CLAUDE_CLI_BIN}:${PATH}"'
    assert body.count(prepend) == 2, (
        "run_daily.sh must prepend the npm install's bin dir in exactly "
        "the two `env -i` blocks that spawn claude (version probe and "
        "pytest); every other step (git, gh, curl, uv, docs publish) "
        "must not see package-controlled binaries on PATH."
    )
    probe_block = _extract_block(
        body, "PROBED_CLAUDE_VERSION=", '[[ -n "${PROBED_CLAUDE_VERSION}" ]]'
    )
    pytest_block = _extract_block(body, 'log "running pytest"', "PYTEST_EXIT=$?")
    assert prepend in probe_block
    assert prepend in pytest_block
    for line in body.splitlines():
        assert not line.startswith(("PATH=", "export PATH=")), (
            f"run_daily.sh reassigns PATH at script scope: {line!r}. A "
            f"compromised npm dependency shipping a bin named `git` or "
            f"`gh` would then execute in later steps with the full "
            f"systemd-injected token environment."
        )


def test_static_npm_install_and_probe_run_in_user_pid_namespace() -> None:
    body = RUN_DAILY.read_text()
    unshare_cmd = "unshare --user --map-current-user --pid --fork --mount-proc"
    assert "unshare" in _required_commands(body), (
        "run_daily.sh must require unshare up front and die early on "
        "kernels that cannot create unprivileged user namespaces, "
        "instead of degrading to unsandboxed package execution."
    )
    lines = _executable_lines(_extract_pin_snippet())
    assert lines.count(unshare_cmd) == 2, (
        "both package-code execution points (npm install and the claude "
        "probe) must run inside the user+pid namespace; `env -i` alone "
        "is not a boundary because same-uid package code can read the "
        "secret-bearing parent's /proc/<pid>/environ."
    )
    resolver_call = _anchor(lines, "pr_gate_version_resolver.py")
    npm_call = _anchor(lines, "npm install --prefix")
    assert unshare_cmd in lines[resolver_call:npm_call], (
        "the npm install (whose lifecycle scripts run package code) must "
        "be unshare-wrapped"
    )
    probe_block = _executable_lines(
        _extract_block(
            body, "PROBED_CLAUDE_VERSION=", '[[ -n "${PROBED_CLAUDE_VERSION}" ]]'
        )
    )
    assert _anchor(probe_block, unshare_cmd) < _anchor(probe_block, "claude --version"), (
        "the installed claude binary must be spawned inside the "
        "user+pid namespace"
    )


def _required_commands(body: str) -> tuple[str, ...]:
    marker = "for cmd in "
    start = _anchor(body, marker) + len(marker)
    end = _anchor(body, ";", start)
    return tuple(body[start:end].split())
