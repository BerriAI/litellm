"""Client for the `client_apps` suite: runs the Claude Code CLI and the Codex CLI
installed into this folder by `npm ci`, and the Vercel AI SDK script
`vercel_ai_turn.mjs`, headlessly against the proxy on a virtual key. Each run gets
an empty HOME and only an allowlisted environment, reads nothing from stdin, and
comes back as the typed events or summary it printed, or a `CommandFailed` value.

Holds the shared ProxyClient so `resources` / `scoped_key` still clean up and the
tests can read the spend rows the proxy wrote for the key.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from claude_code.cli_driver import run_claude
from client_apps_models import CLAUDE_EVENTS, CODEX_EVENTS, ClaudeEvent, CodexEvent, VercelApi, VercelTurn
from e2e_config import PROXY_BASE_URL, SLOW_PROVIDER_TIMEOUT_SECONDS
from proxy_client import ProxyClient

SUITE_DIR: Final = Path(__file__).resolve().parent
CLAUDE_CLI: Final = SUITE_DIR / "node_modules" / ".bin" / "claude"
CODEX_CLI: Final = SUITE_DIR / "node_modules" / ".bin" / "codex"
VERCEL_AI_SCRIPT: Final = SUITE_DIR / "vercel_ai_turn.mjs"
NODE: Final = shutil.which("node") or "node"
CODEX_KEY_ENV: Final = "LITELLM_E2E_CODEX_KEY"
PASSTHROUGH_ENV: Final = ("PATH", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "TERM")


@dataclass(frozen=True, slots=True)
class CommandFailed:
    reason: Literal["nonzero_exit", "timed_out"]
    exit_code: int | None
    stdout: str
    stderr: str


def unwrap_run[T](run: T | CommandFailed) -> T:
    match run:
        case CommandFailed():
            raise AssertionError(run)
        case _:
            return run


def _decoded(chunk: str | bytes | None) -> str:
    match chunk:
        case bytes():
            return chunk.decode(errors="replace")
        case str():
            return chunk
        case None:
            return ""


def _isolated_env(home: Path) -> Mapping[str, str]:
    return {**{name: os.environ[name] for name in PASSTHROUGH_ENV if name in os.environ}, "HOME": str(home)}


def _run(args: tuple[str, ...], *, env: Mapping[str, str], cwd: Path) -> str | CommandFailed:
    try:
        completed: Final = subprocess.run(
            args,
            env=env,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=SLOW_PROVIDER_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as expired:
        return CommandFailed(
            reason="timed_out", exit_code=None, stdout=_decoded(expired.stdout), stderr=_decoded(expired.stderr)
        )
    if completed.returncode != 0:
        return CommandFailed(
            reason="nonzero_exit", exit_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr
        )
    return completed.stdout


def _jsonl_as_array(text: str) -> str:
    return "[" + ",".join(line for line in text.splitlines() if line.strip()) + "]"


def _codex_config(model: str) -> str:
    return "\n".join(
        (
            f'model = "{model}"',
            'model_provider = "litellm"',
            'sandbox_mode = "danger-full-access"',
            "",
            "[features]",
            "plugins = false",
            "",
            "[model_providers.litellm]",
            'name = "LiteLLM proxy"',
            f'base_url = "{PROXY_BASE_URL}/v1"',
            f'env_key = "{CODEX_KEY_ENV}"',
            'wire_api = "responses"',
            "requires_openai_auth = false",
            "",
        )
    )


@dataclass(frozen=True, slots=True)
class ClientAppsClient:
    proxy: ProxyClient

    def run_claude_code(
        self, *, key: str, model: str, prompt: str, allowed_tool: str
    ) -> tuple[ClaudeEvent, ...] | CommandFailed:
        """One `claude --print --output-format stream-json` turn through
        `claude_code/cli_driver.py`, which already isolates HOME and the env.
        `--include-partial-messages` surfaces the streamed deltas as events."""
        driven: Final = run_claude(
            prompt=prompt,
            model=model,
            base_url=PROXY_BASE_URL,
            api_key=key,
            extra_args=("--allowed-tools", allowed_tool, "--permission-mode", "dontAsk", "--include-partial-messages"),
            cli_path=str(CLAUDE_CLI),
            timeout=SLOW_PROVIDER_TIMEOUT_SECONDS,
        )
        if driven.exit_code != 0:
            return CommandFailed(
                reason="nonzero_exit", exit_code=driven.exit_code, stdout=driven.text, stderr=driven.stderr
            )
        return tuple(CLAUDE_EVENTS.validate_python(driven.events))

    def run_codex(self, *, key: str, model: str, prompt: str) -> tuple[CodexEvent, ...] | CommandFailed:
        """One `codex exec --json` turn from a throwaway CODEX_HOME whose
        config.toml points the `litellm` provider at the proxy's Responses API."""
        with tempfile.TemporaryDirectory(prefix="e2e-codex-") as home:
            codex_home: Final = Path(home) / "codex"
            workspace: Final = Path(home) / "workspace"
            codex_home.mkdir()
            workspace.mkdir()
            (codex_home / "config.toml").write_text(_codex_config(model))
            output: Final = _run(
                (
                    str(CODEX_CLI),
                    "exec",
                    "--json",
                    "--skip-git-repo-check",
                    "--ephemeral",
                    "-C",
                    str(workspace),
                    prompt,
                ),
                env={**_isolated_env(Path(home)), "CODEX_HOME": str(codex_home), CODEX_KEY_ENV: key},
                cwd=workspace,
            )
        match output:
            case CommandFailed():
                return output
            case str():
                return tuple(CODEX_EVENTS.validate_json(_jsonl_as_array(output)))

    def run_vercel_ai_sdk(
        self, *, key: str, model: str, api: VercelApi, secret_word: str
    ) -> VercelTurn | CommandFailed:
        """One `streamText` turn of `vercel_ai_turn.mjs` over the chosen API shape,
        with a tool whose result is `secret_word`, summarized as JSON on stdout."""
        with tempfile.TemporaryDirectory(prefix="e2e-vercel-ai-") as home:
            output: Final = _run(
                (NODE, str(VERCEL_AI_SCRIPT)),
                env={
                    **_isolated_env(Path(home)),
                    "LITELLM_PROXY_URL": PROXY_BASE_URL,
                    "LITELLM_API_KEY": key,
                    "E2E_MODEL": model,
                    "E2E_API": api,
                    "E2E_SECRET_WORD": secret_word,
                },
                cwd=SUITE_DIR,
            )
        match output:
            case CommandFailed():
                return output
            case str():
                return VercelTurn.model_validate_json(output)
