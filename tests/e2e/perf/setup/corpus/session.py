"""A coding-agent session as the gateway sees it. The system prompt and tool set come from this
directory, the turns are the assistant reading files and the files coming back as tool results, and the
files are real source files from this checkout, chosen by the request's marker. The input budget is spent
on turns, so a 300k request is a longer session over bigger files than a 100k one, in the wire format of
the endpoint it goes to."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter

Endpoint = Literal["chat", "responses", "messages"]

CORPUS_DIR: Final = Path(__file__).resolve().parent
REPO_ROOT: Final = CORPUS_DIR.parents[4]
CHARS_PER_TOKEN: Final = 4
_FILE_BYTES: Final = (2_000, 80_000)
_STRIDE: Final = 7919
_QUESTION: Final = (
    "Which of the files above raise an exception that the caller is not told about? Answer with the paths only."
)
_CACHE_CONTROL: Final = {"type": "ephemeral"}


class ToolSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    input_schema: dict[str, object]


@dataclass(frozen=True, slots=True)
class Step:
    """One tool round trip: the assistant asks to read `path`, the user turn carries `result`."""

    call_id: str
    path: str
    offset: int
    limit: int
    result: str


@dataclass(frozen=True, slots=True)
class Session:
    system: str
    tools: tuple[ToolSpec, ...]
    steps: tuple[Step, ...]
    question: str


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@cache
def system_prompt() -> str:
    return (CORPUS_DIR / "system_prompt.md").read_text()


@cache
def tool_specs() -> tuple[ToolSpec, ...]:
    return tuple(TypeAdapter(list[ToolSpec]).validate_json((CORPUS_DIR / "tools.json").read_bytes()))


@cache
def corpus_files() -> tuple[Path, ...]:
    """Source files of this checkout between 2 KB and 80 KB, in path order so every run sees the same list."""
    low, high = _FILE_BYTES
    found: Final = tuple(
        path for path in sorted((REPO_ROOT / "litellm").rglob("*.py")) if low <= path.stat().st_size <= high
    )
    assert found, f"no corpus files under {REPO_ROOT / 'litellm'}"
    return found


def _digest(seed: str) -> int:
    return int.from_bytes(hashlib.sha1(seed.encode()).digest()[:6], "big")


def _numbered(lines: tuple[str, ...], offset: int) -> str:
    return "\n".join(f"{offset + index:>6}\t{line}" for index, line in enumerate(lines, start=1))


def _step(marker: str, index: int, budget_tokens: int) -> Step:
    files: Final = corpus_files()
    path: Final = files[(_digest(marker) + index * _STRIDE) % len(files)]
    lines: Final = tuple(path.read_text(errors="replace").splitlines())
    whole: Final = _numbered(lines, 1)
    if estimate_tokens(whole) <= budget_tokens:
        return Step(_call_id(marker, index), str(path.relative_to(REPO_ROOT)), 1, len(lines), whole)
    keep: Final = max(1, len(lines) * budget_tokens * CHARS_PER_TOKEN // max(1, len(whole)))
    return Step(_call_id(marker, index), str(path.relative_to(REPO_ROOT)), 1, keep, _numbered(lines[:keep], 1))


def _call_id(marker: str, index: int) -> str:
    return hashlib.sha1(f"{marker}:call{index}".encode()).hexdigest()[:24]


def build_session(marker: str, input_tokens: int, *, tools: bool) -> Session:
    """Spend `input_tokens` on file-reading turns after the system prompt, the tool set and the question."""
    specs: Final = tool_specs() if tools else ()
    fixed: Final = estimate_tokens(system_prompt()) + sum(estimate_tokens(spec.model_dump_json()) for spec in specs)
    question: Final = f"{marker} {_QUESTION}"
    steps: list[Step] = []  # mutable-ok: each step's size decides whether another fits
    spent = fixed + estimate_tokens(question)  # rebind-ok: running total over the steps
    while spent < input_tokens:
        steps.append(step := _step(marker, len(steps), input_tokens - spent))
        spent += estimate_tokens(step.result) + 40
    return Session(system_prompt(), specs, tuple(steps), question)


def _read_arguments(step: Step) -> dict[str, object]:
    return {"file_path": f"/workspace/litellm/{step.path}", "offset": step.offset, "limit": step.limit}


def _chat_turns(session: Session) -> Sequence[Mapping[str, object]]:
    if not session.tools:
        return [
            turn
            for step in session.steps
            for turn in (
                {"role": "user", "content": f"Review this file.\n\n{step.path}\n{step.result}"},
                {"role": "assistant", "content": f"Read {step.path}, {step.limit} lines. Continue."},
            )
        ]
    return [
        turn
        for step in session.steps
        for turn in (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{step.call_id}",
                        "type": "function",
                        "function": {"name": "Read", "arguments": json.dumps(_read_arguments(step))},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": f"call_{step.call_id}", "content": step.result},
        )
    ]


def _responses_input(session: Session) -> Sequence[Mapping[str, object]]:
    if not session.tools:
        return _chat_turns(session)
    return [
        turn
        for step in session.steps
        for turn in (
            {
                "type": "function_call",
                "call_id": f"call_{step.call_id}",
                "name": "Read",
                "arguments": json.dumps(_read_arguments(step)),
            },
            {"type": "function_call_output", "call_id": f"call_{step.call_id}", "output": step.result},
        )
    ]


def _messages_turns(session: Session) -> Sequence[Mapping[str, object]]:
    if not session.tools:
        return _chat_turns(session)
    last: Final = len(session.steps) - 1
    return [
        turn
        for index, step in enumerate(session.steps)
        for turn in (
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"toolu_{step.call_id}", "name": "Read", "input": _read_arguments(step)}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"toolu_{step.call_id}",
                        "content": step.result,
                        **({"cache_control": _CACHE_CONTROL} if index == last else {}),
                    }
                ],
            },
        )
    ]


def request_body(
    *, endpoint: Endpoint, model: str, marker: str, input_tokens: int, tools: bool, output_tokens: int, stream: bool
) -> dict[str, object]:
    """The session as one request in the endpoint's wire format, the marker in the final user message."""
    session: Final = build_session(marker, input_tokens, tools=tools)
    question: Final = {"role": "user", "content": session.question}
    if endpoint == "responses":
        return {
            "model": model,
            "instructions": session.system,
            "input": [*_responses_input(session), question],
            "max_output_tokens": output_tokens,
            "stream": stream,
            **(
                {
                    "tools": [
                        {"type": "function", "name": s.name, "description": s.description, "parameters": s.input_schema}
                        for s in session.tools
                    ]
                }
                if session.tools
                else {}
            ),
        }
    if endpoint == "messages":
        return {
            "model": model,
            "system": [{"type": "text", "text": session.system, "cache_control": _CACHE_CONTROL}],
            "messages": [*_messages_turns(session), question],
            "max_tokens": output_tokens,
            "stream": stream,
            **({"tools": [spec.model_dump() for spec in session.tools]} if session.tools else {}),
        }
    return {
        "model": model,
        "messages": [{"role": "system", "content": session.system}, *_chat_turns(session), question],
        "max_completion_tokens": output_tokens,
        **({"reasoning_effort": "none"} if session.tools else {}),
        "stream": stream,
        **(
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": s.name, "description": s.description, "parameters": s.input_schema},
                    }
                    for s in session.tools
                ]
            }
            if session.tools
            else {}
        ),
    }
