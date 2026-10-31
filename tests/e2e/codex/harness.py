from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Final

from cli_harness.lockdown import isolated_home, locked_down_env
from cli_harness.rate_limiter import RateBucket, RateLimiter
from codex.rollout import Compaction, rollout_compactions
from codex.run import CodexRun, codex_run
from e2e_metadata import step

CODEX_CLI: Final = "codex"
EXEC_TIMEOUT_SECONDS: Final = 600.0
_API_KEY_ENV: Final = "LITELLM_API_KEY"
_PROVIDER: Final = "litellm"
_CONFIG_TOML: Final = """\
model_provider = "{provider}"
approval_policy = "never"
sandbox_mode = "workspace-write"
check_for_update_on_startup = false

[analytics]
enabled = false

[model_providers.{provider}]
name = "LiteLLM"
base_url = "{gateway}/v1"
env_key = "{api_key_env}"
wire_api = "responses"
model_catalog_url = "{gateway}/v1/models"
"""


@dataclass(frozen=True, slots=True)
class CodexHarness:
    home: Path
    api_key: str = field(repr=False)
    rate_bucket: RateBucket
    limiter: RateLimiter
    cli: str = CODEX_CLI
    timeout_seconds: float = EXEC_TIMEOUT_SECONDS

    @property
    def codex_home(self) -> Path:
        return self.home / ".codex"

    def new_workspace(self, files: Mapping[str, str]) -> Path:
        workspace: Final = Path(tempfile.mkdtemp(prefix="workspace-", dir=self.home)).resolve()
        for name, content in files.items():
            _ = (workspace / name).write_text(content)
        return workspace

    @step('Run Codex on {model} through the proxy with the prompt "{prompt}"')
    def exec(self, prompt: str, *, model: str, cwd: Path, compact_after_tokens: int | None = None) -> CodexRun:
        return self._run(("exec",), (*_compaction_flags(compact_after_tokens), "--model", model, "--", prompt), cwd)

    @step('Resume Codex thread {thread_id} on {model} through the proxy with the prompt "{prompt}"')
    def resume(
        self, thread_id: str, prompt: str, *, model: str, cwd: Path, compact_after_tokens: int | None = None
    ) -> CodexRun:
        return self._run(
            ("exec", "resume"),
            (*_compaction_flags(compact_after_tokens), "--model", model, "--", thread_id, prompt),
            cwd,
        )

    def compactions(self, thread_id: str) -> tuple[Compaction, ...]:
        rollouts: Final = sorted(self.codex_home.glob(f"sessions/**/rollout-*-{thread_id}.jsonl"))
        return tuple(chain.from_iterable(rollout_compactions(rollout.read_text()) for rollout in rollouts))

    def _run(self, subcommand: tuple[str, ...], args: tuple[str, ...], cwd: Path) -> CodexRun:
        _ = self.limiter.acquire(self.rate_bucket)
        completed: Final = subprocess.run(
            (self.cli, *subcommand, "--json", "--skip-git-repo-check", *args),
            cwd=cwd,
            env=locked_down_env(
                home=self.home, overrides={"CODEX_HOME": str(self.codex_home), _API_KEY_ENV: self.api_key}
            ),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        return codex_run(exit_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


def _compaction_flags(after_tokens: int | None) -> tuple[str, ...]:
    match after_tokens:
        case None:
            return ()
        case int():
            return (
                "--config",
                f'model_providers.{_PROVIDER}.capabilities.remote_compaction="v2"',
                "--config",
                'model_auto_compact_token_limit_scope="body_after_prefix"',
                "--config",
                f"model_auto_compact_token_limit={after_tokens}",
            )


@contextmanager
def codex_harness(
    *, gateway: str, api_key: str, rate_bucket: RateBucket, limiter: RateLimiter
) -> Generator[CodexHarness]:
    with isolated_home("codex-cli-home-") as home:
        harness: Final = CodexHarness(home=home, api_key=api_key, rate_bucket=rate_bucket, limiter=limiter)
        harness.codex_home.mkdir()
        _ = (harness.codex_home / "config.toml").write_text(
            _CONFIG_TOML.format(provider=_PROVIDER, gateway=gateway, api_key_env=_API_KEY_ENV)
        )
        yield harness
