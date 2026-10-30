from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final

from e2e_metadata import step

from client_apps.codex.events import Transcript, parse_transcript

CLIENT_APPS_DIR: Final = Path(__file__).resolve().parents[1]
LOCAL_BINARY: Final = CLIENT_APPS_DIR / "node_modules" / ".bin" / "codex"
TIMEOUT_ENV: Final = "LITELLM_COMPAT_CODEX_TIMEOUT_SECONDS"
DEFAULT_TIMEOUT_SECONDS: Final = 300.0
LITELLM_PROVIDER_KEY_ENV: Final = "LITELLM_API_KEY"
OPENAI_PROVIDER_KEY_ENV: Final = "CODEX_API_KEY"
SANDBOX_MODE: Final = "danger-full-access"
STDERR_TAIL_CHARS: Final = 600


class Wiring(StrEnum):
    LITELLM_PROVIDER = "litellm_provider"
    OPENAI_BASE_URL = "openai_base_url"


@dataclass(frozen=True, slots=True)
class Gateway:
    base_url: str
    api_key: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class CodexSession:
    root: Path

    @property
    def codex_home(self) -> Path:
        return self.root / "codex_home"

    @property
    def home(self) -> Path:
        return self.root / "home"

    @property
    def workdir(self) -> Path:
        return self.root / "workdir"


def new_session(root: Path) -> CodexSession:
    session: Final = CodexSession(root=root)
    for directory in (session.codex_home, session.home, session.workdir):
        directory.mkdir(parents=True, exist_ok=True)
    return session


@dataclass(frozen=True, slots=True)
class CodexTurn:
    model: str
    prompt: str
    options: tuple[str, ...] = ()
    resume_thread: str | None = None
    wiring: Wiring = Wiring.LITELLM_PROVIDER


@dataclass(frozen=True, slots=True)
class Finished:
    exit_code: int
    transcript: Transcript
    stderr: str


@dataclass(frozen=True, slots=True)
class TimedOut:
    seconds: float
    transcript: Transcript
    stderr: str


type CodexRun = Finished | TimedOut


def resolve_binary(local: Path = LOCAL_BINARY, which: Callable[[str], str | None] = shutil.which) -> Path | None:
    if local.is_file():
        return local
    on_path: Final = which("codex")
    return Path(on_path) if on_path is not None else None


def timeout_seconds(environ: Mapping[str, str]) -> float:
    return float(environ.get(TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS))


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _overrides(settings: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(chain.from_iterable(("-c", f"{key}={_toml_string(value)}") for key, value in settings.items()))


def wiring_args(wiring: Wiring, base_url: str) -> tuple[str, ...]:
    match wiring:
        case Wiring.LITELLM_PROVIDER:
            return _overrides(
                MappingProxyType(
                    {
                        "model_provider": "litellm",
                        "model_providers.litellm.name": "LiteLLM",
                        "model_providers.litellm.base_url": f"{base_url}/v1",
                        "model_providers.litellm.env_key": LITELLM_PROVIDER_KEY_ENV,
                        "model_providers.litellm.wire_api": "responses",
                    }
                )
            )
        case Wiring.OPENAI_BASE_URL:
            return _overrides(MappingProxyType({"openai_base_url": f"{base_url}/v1"}))


def _key_env(wiring: Wiring) -> str:
    match wiring:
        case Wiring.LITELLM_PROVIDER:
            return LITELLM_PROVIDER_KEY_ENV
        case Wiring.OPENAI_BASE_URL:
            return OPENAI_PROVIDER_KEY_ENV


def command(binary: Path, gateway: Gateway, session: CodexSession, turn: CodexTurn) -> tuple[str, ...]:
    resume: Final = ("resume", turn.resume_thread) if turn.resume_thread is not None else ()
    return (
        str(binary),
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "-C",
        str(session.workdir),
        "-m",
        turn.model,
        "-s",
        SANDBOX_MODE,
        *_overrides(MappingProxyType({"approval_policy": "never"})),
        *wiring_args(turn.wiring, gateway.base_url),
        *turn.options,
        *resume,
        "--",
        turn.prompt,
    )


def environment(gateway: Gateway, session: CodexSession, wiring: Wiring) -> Mapping[str, str]:
    return MappingProxyType(
        {
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": str(session.home),
            "CODEX_HOME": str(session.codex_home),
            _key_env(wiring): gateway.api_key,
        }
    )


def _turn_files(session: CodexSession) -> tuple[Path, Path]:
    turn: Final = len(tuple(session.root.glob("turn-*.jsonl"))) + 1
    return session.root / f"turn-{turn}.jsonl", session.root / f"turn-{turn}.stderr.txt"


@step("Run codex exec against {turn.model}")
def run_codex(binary: Path, gateway: Gateway, session: CodexSession, turn: CodexTurn, seconds: float) -> CodexRun:
    stdout_path, stderr_path = _turn_files(session)
    with (
        stdout_path.open("w") as stdout,
        stderr_path.open("w") as stderr,
        subprocess.Popen(
            command(binary, gateway, session, turn),
            cwd=session.workdir,
            env=environment(gateway, session, turn.wiring),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        ) as process,
    ):
        try:
            exit_code: Final = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            _ = process.wait()
            return TimedOut(
                seconds=seconds,
                transcript=parse_transcript(stdout_path.read_text()),
                stderr=stderr_path.read_text(),
            )
    return Finished(
        exit_code=exit_code, transcript=parse_transcript(stdout_path.read_text()), stderr=stderr_path.read_text()
    )


def stderr_tail(stderr: str) -> str:
    lines: Final = tuple(
        line for line in stderr.splitlines() if line.strip() and not line.startswith("Reading additional input")
    )
    return "\n".join(lines)[-STDERR_TAIL_CHARS:]
