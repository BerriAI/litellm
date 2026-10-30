from __future__ import annotations

import secrets
import struct
import zlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Final

import pytest
from e2e_config import unique_marker
from pydantic import BaseModel, ConfigDict, ValidationError

from client_apps.codex.cli_driver import (
    CodexRun,
    CodexSession,
    CodexTurn,
    Finished,
    Gateway,
    TimedOut,
    Wiring,
    new_session,
    run_codex,
    stderr_tail,
)
from client_apps.codex.checks import Check, all_checks, final_message_has, named_by_eye
from client_apps.codex.events import CommandExecution, FileChange, Transcript, WebSearch
from client_apps.compat_results import Failed, Passed
from client_apps.compat_collector import CompatRecorder

SWATCHES: Final = (("red", (220, 20, 20)), ("green", (20, 160, 20)), ("blue", (20, 40, 220)))
SWATCH_SIDE: Final = 64
EXPECTED_DOCS_DOMAIN: Final = "litellm.ai"
OUTPUT_SCHEMA: Final = (
    '{"type": "object", "properties": {"marker": {"type": "string"}, "word_count": {"type": "integer"}}, '
    '"required": ["marker", "word_count"], "additionalProperties": false}'
)


@dataclass(frozen=True, slots=True)
class CodexCell:
    binary: Path
    gateway: Gateway
    seconds: float
    sessions_root: Path

    def session(self, model: str) -> CodexSession:
        return new_session(self.sessions_root / model)

    def run(self, session: CodexSession, turn: CodexTurn) -> CodexRun:
        return run_codex(self.binary, self.gateway, session, turn, self.seconds)


type Verdict = Passed | Failed
type Feature = Callable[[CodexCell, str], Verdict]


def assert_cell(compat_result: CompatRecorder, cell: CodexCell, models: tuple[str, ...], feature: Feature) -> None:
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        verdicts: Final = tuple(pool.map(partial(feature, cell), models))
    for model, verdict in zip(models, verdicts, strict=True):
        match verdict:
            case Passed():
                compat_result.passed(model)
            case Failed(error=error):
                compat_result.failed(model, error)
    failures: Final = tuple(
        f"{model}: {verdict.error}"
        for model, verdict in zip(models, verdicts, strict=True)
        if isinstance(verdict, Failed)
    )
    if failures:
        pytest.fail("\n".join(failures), pytrace=False)


def run_error(run: CodexRun) -> str | None:
    match run:
        case TimedOut(seconds=seconds, stderr=stderr):
            return f"codex exec timed out after {seconds:.0f}s; stderr: {stderr_tail(stderr)}"
        case Finished(transcript=transcript) if transcript.failure is not None:
            return f"turn failed: {transcript.failure}"
        case Finished(exit_code=exit_code, stderr=stderr) if exit_code != 0:
            return f"codex exec exited {exit_code}; stderr: {stderr_tail(stderr)}"
        case Finished(transcript=transcript) if transcript.usage is None:
            return "codex exec ended without a turn.completed event"
        case Finished():
            return None


def verdict(run: CodexRun, check: Check) -> Verdict:
    error: Final = run_error(run) or check(run.transcript)
    return Failed(error=error) if error else Passed()


def basic_messaging(cell: CodexCell, model: str) -> Verdict:
    marker: Final = unique_marker()
    prompt: Final = f"Reply with exactly this token and nothing else: {marker}"
    run: Final = cell.run(cell.session(model), CodexTurn(model, prompt))
    return verdict(run, final_message_has(marker))


def _echoed_by_shell(marker: str) -> Check:
    def check(transcript: Transcript) -> str | None:
        commands: Final = tuple(item for item in transcript.completed_items if isinstance(item, CommandExecution))
        echoed: Final = any(command.exit_code == 0 and marker in command.aggregated_output for command in commands)
        return None if echoed else f"no completed command_execution printed {marker!r}: {commands!r}"

    return check


def shell_command(cell: CodexCell, model: str) -> Verdict:
    marker: Final = unique_marker()
    prompt: Final = f"Run the shell command `echo {marker}` with your shell tool, then reply with its output only."
    run: Final = cell.run(cell.session(model), CodexTurn(model, prompt))
    return verdict(run, all_checks(_echoed_by_shell(marker), final_message_has(marker)))


def _edited(session: CodexSession, marker: str) -> Check:
    target: Final = session.workdir / "notes.txt"

    def check(transcript: Transcript) -> str | None:
        changes: Final = tuple(item for item in transcript.completed_items if isinstance(item, FileChange))
        updates: Final = chain.from_iterable(item.changes for item in changes)
        touched: Final = tuple(Path(update.path).name for update in updates)
        if target.name not in touched:
            return f"no completed file_change item touched {target.name}: {changes!r}"
        content: Final = target.read_text() if target.is_file() else None
        return None if content is not None and marker in content else f"{target.name} holds {content!r}, not {marker!r}"

    return check


def file_edit(cell: CodexCell, model: str) -> Verdict:
    marker: Final = unique_marker()
    session: Final = cell.session(model)
    prompt: Final = (
        f"Create a file named notes.txt in the current directory whose only line is {marker}. "
        "Use your apply_patch tool to create it, not the shell."
    )
    return verdict(cell.run(session, CodexTurn(model, prompt)), _edited(session, marker))


def _resumed(cell: CodexCell, model: str, first_prompt: str, second_prompt: str) -> tuple[CodexRun, CodexRun | str]:
    session: Final = cell.session(model)
    first: Final = cell.run(session, CodexTurn(model, first_prompt))
    thread_id: Final = first.transcript.thread_id
    first_error: Final = run_error(first)
    if first_error is not None or thread_id is None:
        return first, f"first turn: {first_error or 'no thread.started event'}"
    return first, cell.run(session, CodexTurn(model, second_prompt, resume_thread=thread_id))


def _same_thread(thread_id: str | None) -> Check:
    def check(transcript: Transcript) -> str | None:
        if transcript.thread_id == thread_id:
            return None
        return f"resumed thread {transcript.thread_id!r} != {thread_id!r}"

    return check


def multi_turn(cell: CodexCell, model: str) -> Verdict:
    marker: Final = unique_marker()
    first, second = _resumed(
        cell,
        model,
        f"Remember this token for later: {marker}. Reply with OK only.",
        "What token did I ask you to remember? Reply with the token only.",
    )
    if isinstance(second, str):
        return Failed(error=second)
    return verdict(second, all_checks(_same_thread(first.transcript.thread_id), final_message_has(marker)))


def _cache_read(transcript: Transcript) -> str | None:
    cached: Final = transcript.usage.cached_input_tokens if transcript.usage is not None else 0
    return None if cached > 0 else f"second turn reported {cached} cached input tokens: {transcript.usage!r}"


def prompt_caching(cell: CodexCell, model: str) -> Verdict:
    first_marker: Final = unique_marker()
    second_marker: Final = unique_marker()
    _, second = _resumed(
        cell,
        model,
        f"Reply with exactly this token and nothing else: {first_marker}",
        f"Now reply with exactly this token and nothing else: {second_marker}",
    )
    if isinstance(second, str):
        return Failed(error=second)
    return verdict(second, _cache_read)


def _reasoned(transcript: Transcript) -> str | None:
    tokens: Final = transcript.usage.reasoning_output_tokens if transcript.usage is not None else 0
    return None if tokens > 0 else f"turn reported {tokens} reasoning output tokens: {transcript.usage!r}"


def reasoning_effort(cell: CodexCell, model: str) -> Verdict:
    marker: Final = unique_marker()
    prompt: Final = (
        f"How many prime numbers are there between 1 and 60? Reply with the number only. Request id {marker}."
    )
    options: Final = ("-c", 'model_reasoning_effort="high"')
    return verdict(cell.run(cell.session(model), CodexTurn(model, prompt, options=options)), _reasoned)


def solid_png(rgb: tuple[int, int, int], side: int = SWATCH_SIDE) -> bytes:
    row: Final = b"\x00" + bytes(rgb) * side
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(row * side)),
            _png_chunk(b"IEND", b""),
        )
    )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def vision(cell: CodexCell, model: str) -> Verdict:
    marker: Final = unique_marker()
    color, rgb = secrets.choice(SWATCHES)
    session: Final = cell.session(model)
    image: Final = session.root / "swatch.png"
    _ = image.write_bytes(solid_png(rgb))
    prompt: Final = f"What single color fills the attached image? Reply with one lowercase word. Request id {marker}."
    run: Final = cell.run(session, CodexTurn(model, prompt, options=("-i", str(image))))
    return verdict(run, named_by_eye(color))


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    marker: str
    word_count: int


def _structured(marker: str) -> Check:
    def check(transcript: Transcript) -> str | None:
        try:
            answer: Final = StructuredAnswer.model_validate_json(transcript.final_message or "")
        except ValidationError as error:
            return f"final message {transcript.final_message!r} does not match the output schema: {error}"
        expected: Final = StructuredAnswer(marker=marker, word_count=4)
        return None if answer == expected else f"structured answer {answer!r} != {expected!r}"

    return check


def structured_output(cell: CodexCell, model: str) -> Verdict:
    marker: Final = unique_marker()
    session: Final = cell.session(model)
    schema: Final = session.root / "schema.json"
    _ = schema.write_text(OUTPUT_SCHEMA)
    prompt: Final = (
        f"Put the token {marker} in marker and the number of words in 'the quick brown fox' in word_count."
    )
    run: Final = cell.run(session, CodexTurn(model, prompt, options=("--output-schema", str(schema))))
    return verdict(run, _structured(marker))


def _searched(transcript: Transcript) -> str | None:
    searches: Final = tuple(item for item in transcript.completed_items if isinstance(item, WebSearch))
    return None if searches else "no completed web_search item"


def web_search(cell: CodexCell, model: str) -> Verdict:
    marker: Final = unique_marker()
    prompt: Final = (
        "Use your web search tool, not the shell, to find the domain of LiteLLM's documentation site. "
        f"Reply with the domain only. Request id {marker}."
    )
    turn: Final = CodexTurn(model, prompt, options=("-c", 'web_search="live"'), wiring=Wiring.OPENAI_BASE_URL)
    return verdict(cell.run(cell.session(model), turn), all_checks(_searched, final_message_has(EXPECTED_DOCS_DOMAIN)))
