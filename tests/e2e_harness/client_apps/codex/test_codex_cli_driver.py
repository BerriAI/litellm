from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from client_apps.codex.cli_driver import (
    LITELLM_PROVIDER_KEY_ENV,
    OPENAI_PROVIDER_KEY_ENV,
    TIMEOUT_ENV,
    CodexTurn,
    Finished,
    Gateway,
    TimedOut,
    Wiring,
    command,
    environment,
    new_session,
    resolve_binary,
    run_codex,
    stderr_tail,
    timeout_seconds,
)

GATEWAY: Final = Gateway(base_url="http://127.0.0.1:24001", api_key="sk-scoped")
EMITTING_STUB: Final = """#!/bin/sh
printf '%s\\n' "$@" >&2
echo '{"type":"thread.started","thread_id":"t-1"}'
echo '{"type":"item.completed","item":{"id":"i","type":"agent_message","text":"hi"}}'
echo '{"type":"turn.completed","usage":{"input_tokens":3,"output_tokens":1}}'
exit 4
"""
HANGING_STUB: Final = """#!/bin/sh
echo '{"type":"thread.started","thread_id":"t-2"}'
sleep 30
"""


def _stub(directory: Path, script: str) -> Path:
    binary: Final = directory / "codex"
    _ = binary.write_text(script)
    binary.chmod(0o755)
    return binary


def _option(argv: tuple[str, ...], name: str) -> tuple[str, ...]:
    return tuple(argv[index + 1] for index, value in enumerate(argv[:-1]) if value == name)


def test_litellm_provider_command_points_codex_at_the_proxy_v1_responses(tmp_path: Path) -> None:
    session: Final = new_session(tmp_path)
    argv: Final = command(Path("/bin/codex"), GATEWAY, session, CodexTurn("gpt-6.1-sol", "-- say hi"))

    assert argv[:2] == ("/bin/codex", "exec")
    assert set(_option(argv, "-c")) >= {
        'model_provider="litellm"',
        'model_providers.litellm.base_url="http://127.0.0.1:24001/v1"',
        f'model_providers.litellm.env_key="{LITELLM_PROVIDER_KEY_ENV}"',
        'model_providers.litellm.wire_api="responses"',
        'approval_policy="never"',
    }
    assert _option(argv, "-m") == ("gpt-6.1-sol",)
    assert _option(argv, "-C") == (str(session.workdir),)
    assert argv[-2:] == ("--", "-- say hi")


def test_resume_comes_after_every_exec_option_and_before_the_prompt(tmp_path: Path) -> None:
    turn: Final = CodexTurn("gpt-6.1-sol", "again", options=("-i", "img.png"), resume_thread="t-9")
    argv: Final = command(Path("codex"), GATEWAY, new_session(tmp_path), turn)
    resume_at: Final = argv.index("resume")

    assert argv[resume_at : resume_at + 4] == ("resume", "t-9", "--", "again")
    assert argv.index("-i") < resume_at
    assert max(index for index, value in enumerate(argv) if value == "-c") < resume_at


def test_openai_base_url_wiring_keeps_the_builtin_provider(tmp_path: Path) -> None:
    turn: Final = CodexTurn("gpt-6.1-sol", "p", wiring=Wiring.OPENAI_BASE_URL)
    overrides: Final = _option(command(Path("codex"), GATEWAY, new_session(tmp_path), turn), "-c")

    assert 'openai_base_url="http://127.0.0.1:24001/v1"' in overrides
    assert not any(override.startswith("model_provider") for override in overrides)


def test_environment_carries_only_the_session_dirs_and_the_wired_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leak")
    session: Final = new_session(tmp_path)
    litellm_env: Final = environment(GATEWAY, session, Wiring.LITELLM_PROVIDER)
    openai_env: Final = environment(GATEWAY, session, Wiring.OPENAI_BASE_URL)

    assert set(litellm_env) == {"PATH", "HOME", "CODEX_HOME", LITELLM_PROVIDER_KEY_ENV}
    assert set(openai_env) == {"PATH", "HOME", "CODEX_HOME", OPENAI_PROVIDER_KEY_ENV}
    assert litellm_env[LITELLM_PROVIDER_KEY_ENV] == openai_env[OPENAI_PROVIDER_KEY_ENV] == "sk-scoped"
    assert litellm_env["CODEX_HOME"] == str(session.codex_home)
    assert litellm_env["HOME"] == str(session.home)


def test_resolve_binary_prefers_the_pinned_local_install(tmp_path: Path) -> None:
    local: Final = _stub(tmp_path, "#!/bin/sh\n")

    assert resolve_binary(local, which=lambda _: "/usr/bin/codex") == local
    assert resolve_binary(tmp_path / "missing", which=lambda _: "/usr/bin/codex") == Path("/usr/bin/codex")
    assert resolve_binary(tmp_path / "missing", which=lambda _: None) is None


def test_timeout_seconds_reads_the_env_override() -> None:
    assert timeout_seconds({TIMEOUT_ENV: "12.5"}) == 12.5
    assert timeout_seconds({}) > 0


def test_run_codex_parses_stdout_and_keeps_each_turn_log(tmp_path: Path) -> None:
    binary: Final = _stub(tmp_path, EMITTING_STUB)
    session: Final = new_session(tmp_path / "session")
    first: Final = run_codex(binary, GATEWAY, session, CodexTurn("m", "p"), 30)
    second: Final = run_codex(binary, GATEWAY, session, CodexTurn("m", "q", resume_thread="t-1"), 30)

    assert isinstance(first, Finished)
    assert first.exit_code == 4
    assert first.transcript.thread_id == "t-1"
    assert first.transcript.final_message == "hi"
    assert first.transcript.usage is not None
    assert first.transcript.usage.input_tokens == 3
    assert isinstance(second, Finished)
    assert "resume\nt-1\n--\nq" in second.stderr
    assert sorted(path.name for path in session.root.glob("turn-*")) == [
        "turn-1.jsonl",
        "turn-1.stderr.txt",
        "turn-2.jsonl",
        "turn-2.stderr.txt",
    ]


def test_run_codex_kills_a_hung_turn_and_keeps_what_it_printed(tmp_path: Path) -> None:
    run: Final = run_codex(_stub(tmp_path, HANGING_STUB), GATEWAY, new_session(tmp_path / "s"), CodexTurn("m", "p"), 1)

    assert isinstance(run, TimedOut)
    assert run.transcript.thread_id == "t-2"


def test_stderr_tail_drops_the_stdin_notice_and_keeps_the_end() -> None:
    tail: Final = stderr_tail("Reading additional input from stdin...\n\n" + "x" * 2000 + "\nboom")

    assert "Reading additional input" not in tail
    assert tail.endswith("boom")
    assert len(tail) <= 600
