import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from itertools import chain
from typing import Final

import pytest
from integration._support.timing import assert_within_factor_of_control, timed
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.types.utils import TextCompletionResponse

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PROVIDER_KEY: Final = "synthetic-provider-key"
_PIECES: Final = ("Hello", " streamed", " world")
_PROMPT_TOKENS: Final = 11
_COMPLETION_TOKENS: Final = 3
_CACHED_TOKENS: Final = 4
_USAGE: Final[dict[str, JsonValue]] = {
    "prompt_tokens": _PROMPT_TOKENS,
    "completion_tokens": _COMPLETION_TOKENS,
    "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
    "prompt_tokens_details": {"cached_tokens": _CACHED_TOKENS},
}
_CHOICE_PIECES: Final = (("Hi", " there"), ("Hey", " you"))
_GRID_CHOICES: Final = 128
_GRID_CHUNKS: Final = 8192
_GRID_ROUNDS: Final = 2
_REBUILD_FACTOR: Final = 2


@dataclass(frozen=True, slots=True)
class Member:
    model: str
    target: str


_MEMBERS: Final = {
    "text_completion_openai": Member("text-completion-openai/gpt-3.5-turbo-instruct", "/v1/completions"),
    "hosted_vllm": Member("hosted_vllm/Qwen/Qwen3-0.6B", "/v1/completions"),
    "openai_chat_backed": Member("openai/gpt-5.4-nano", "/v1/chat/completions"),
}


def _frame(body: Mapping[str, JsonValue]) -> bytes:
    return b"data: " + json.dumps(body).encode() + b"\n\n"


def _text_choice(index: int, text: str, finish_reason: str | None) -> dict[str, JsonValue]:
    return {"text": text, "index": index, "logprobs": None, "finish_reason": finish_reason}


def _chat_choice(index: int, text: str, finish_reason: str | None) -> dict[str, JsonValue]:
    return {"index": index, "delta": {"role": "assistant", "content": text}, "finish_reason": finish_reason}


def _stream(
    request: Request, identity: str, pieces: tuple[tuple[int, str, str | None], ...], include_usage: bool
) -> Reply:
    chat: Final = request.target.endswith("/chat/completions")
    head: Final[dict[str, JsonValue]] = {
        "id": identity,
        "object": "chat.completion.chunk" if chat else "text_completion",
        "created": 1,
        "model": "scripted",
    }
    choice: Final = _chat_choice if chat else _text_choice
    usage: Final = (_frame({**head, "choices": [], "usage": _USAGE}),) if include_usage else ()
    return Reply(
        content_type="text/event-stream",
        chunks=(
            *(_frame({**head, "choices": [choice(index, text, finish)]}) for index, text, finish in pieces),
            *usage,
            b"data: [DONE]\n\n",
        ),
    )


def _asks_for_usage(body: Mapping[str, JsonValue]) -> bool:
    options: Final = body.get("stream_options")
    return isinstance(options, dict) and options.get("include_usage") is True


def _upstream(identity: str, pieces: tuple[tuple[int, str, str | None], ...]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body.get("stream") is True, body
        return _stream(request, identity, pieces, _asks_for_usage(body))

    return respond


def _one_choice_pieces() -> tuple[tuple[int, str, str | None], ...]:
    return (
        *((0, piece, None) for piece in _PIECES),
        (0, "", "length"),
    )


def _two_choice_pieces() -> tuple[tuple[int, str, str | None], ...]:
    (first, second) = _CHOICE_PIECES
    return ((0, first[0], None), (1, second[0], None), (0, first[1], "stop"), (1, second[1], "length"))


@pytest.fixture
def identity() -> str:
    return "cmpl-rebuild-" + uuid.uuid4().hex


@pytest.fixture
def one_choice_wire(identity: str) -> Iterator[Wire]:
    with wire_server(_upstream(identity, _one_choice_pieces())) as served:
        yield served


@pytest.fixture
def two_choice_wire(identity: str) -> Iterator[Wire]:
    with wire_server(_upstream(identity, _two_choice_pieces())) as served:
        yield served


def _stream_options(include_usage: bool) -> dict[str, object]:
    return {"stream_options": {"include_usage": True}} if include_usage else {}


async def _async_chunks(
    member: Member, wire: Wire, include_usage: bool, extra: Mapping[str, object]
) -> tuple[TextCompletionResponse, ...]:
    stream: Final = await litellm.atext_completion(
        model=member.model,
        prompt="rebuild probe",
        api_base=wire.url + "/v1",
        api_key=_PROVIDER_KEY,
        max_tokens=16,
        stream=True,
        num_retries=0,
        **_stream_options(include_usage),
        **extra,
    )
    return tuple([chunk async for chunk in stream])


def _sync_chunks(member: Member, wire: Wire, include_usage: bool) -> tuple[TextCompletionResponse, ...]:
    stream: Final = litellm.text_completion(
        model=member.model,
        prompt="rebuild probe",
        api_base=wire.url + "/v1",
        api_key=_PROVIDER_KEY,
        max_tokens=16,
        stream=True,
        num_retries=0,
        **_stream_options(include_usage),
    )
    return tuple(stream)


def _rebuilt(chunks: tuple[TextCompletionResponse, ...]) -> TextCompletionResponse:
    rebuilt: Final = litellm.stream_chunk_builder(chunks=list(chunks))
    assert isinstance(rebuilt, TextCompletionResponse), rebuilt
    return rebuilt


def _posted_targets(wire: Wire) -> tuple[str, ...]:
    return tuple(request.target for request in wire.drain() if request.method == "POST")


@pytest.mark.parametrize(
    ("member", "mode"),
    [
        ("text_completion_openai", "async"),
        ("text_completion_openai", "sync"),
        ("hosted_vllm", "async"),
        ("openai_chat_backed", "async"),
    ],
)
async def test_rebuilt_text_completion_stream_keeps_the_reported_finish_reason_and_usage(
    one_choice_wire: Wire, identity: str, member: str, mode: str
) -> None:
    chosen: Final = _MEMBERS[member]
    chunks: Final = (
        await _async_chunks(chosen, one_choice_wire, True, {})
        if mode == "async"
        else _sync_chunks(chosen, one_choice_wire, True)
    )

    rebuilt: Final = _rebuilt(chunks)

    assert [(choice.index, choice.text) for choice in rebuilt.choices] == [(0, "".join(_PIECES))], rebuilt
    assert rebuilt.choices[0].finish_reason == "length", rebuilt
    assert (rebuilt.usage.prompt_tokens, rebuilt.usage.completion_tokens, rebuilt.usage.total_tokens) == (
        _PROMPT_TOKENS,
        _COMPLETION_TOKENS,
        _PROMPT_TOKENS + _COMPLETION_TOKENS,
    ), rebuilt.usage
    assert rebuilt.usage.prompt_tokens_details is not None, rebuilt.usage
    assert rebuilt.usage.prompt_tokens_details.cached_tokens == _CACHED_TOKENS, rebuilt.usage
    assert rebuilt.id == identity, rebuilt
    assert _posted_targets(one_choice_wire) == (chosen.target,)


@pytest.mark.parametrize("member", ["text_completion_openai", "openai_chat_backed"])
async def test_rebuilt_text_completion_stream_without_reported_usage_counts_the_streamed_text(
    one_choice_wire: Wire, member: str
) -> None:
    chosen: Final = _MEMBERS[member]
    chunks: Final = await _async_chunks(chosen, one_choice_wire, False, {})

    rebuilt: Final = _rebuilt(chunks)

    text: Final = "".join(_PIECES)
    assert rebuilt.choices[0].text == text, rebuilt
    assert rebuilt.choices[0].finish_reason == "length", rebuilt
    expected_completion: Final = litellm.token_counter(model=chosen.model, text=text, count_response_tokens=True)
    assert expected_completion > 0
    assert rebuilt.usage.completion_tokens == expected_completion, rebuilt.usage
    assert rebuilt.usage.total_tokens == rebuilt.usage.prompt_tokens + rebuilt.usage.completion_tokens, rebuilt.usage
    assert _posted_targets(one_choice_wire) == (chosen.target,)


async def test_rebuilt_chat_backed_text_completion_stream_keeps_each_choice_text_apart(two_choice_wire: Wire) -> None:
    chosen: Final = _MEMBERS["openai_chat_backed"]
    chunks: Final = await _async_chunks(chosen, two_choice_wire, True, {"n": 2})

    rebuilt: Final = _rebuilt(chunks)

    assert [(choice.index, choice.text) for choice in rebuilt.choices] == [
        (index, "".join(pieces)) for index, pieces in enumerate(_CHOICE_PIECES)
    ], rebuilt
    assert (rebuilt.usage.prompt_tokens, rebuilt.usage.completion_tokens) == (_PROMPT_TOKENS, _COMPLETION_TOKENS)
    assert _posted_targets(two_choice_wire) == (chosen.target,)


def _grid_row(step: int, last: int, choices: int) -> Iterator[tuple[int, str, str | None]]:
    return ((index, "x", "length" if step == last else None) for index in range(choices))


def _grid_pieces(choices: int) -> tuple[tuple[int, str, str | None], ...]:
    steps: Final = _GRID_CHUNKS // choices
    return tuple(chain.from_iterable(_grid_row(step, steps - 1, choices) for step in range(steps)))


def _grid_upstream(request: Request) -> Reply:
    requested: Final = _JSON_OBJECT.validate_json(request.body).get("n")
    choices: Final = requested if isinstance(requested, int) else 1
    return _stream(request, "cmpl-grid-" + uuid.uuid4().hex, _grid_pieces(choices), False)


@pytest.fixture
def grid_wire() -> Iterator[Wire]:
    with wire_server(_grid_upstream) as served:
        yield served


async def _rebuild_seconds(wire: Wire, choices: int) -> float:
    chunks: Final = await _async_chunks(_MEMBERS["openai_chat_backed"], wire, False, {"n": choices})
    run: Final = timed(lambda: _rebuilt(chunks))
    assert sum(len(choice.text) for choice in run.value.choices) == _GRID_CHUNKS, run.value.usage
    return run.seconds


@pytest.mark.timeout(600)
async def test_rebuilding_a_many_choice_text_completion_stream_costs_what_one_choice_of_the_same_size_does(
    grid_wire: Wire, record_property: Callable[[str, object], None]
) -> None:
    pairs: Final = [
        (await _rebuild_seconds(grid_wire, 1), await _rebuild_seconds(grid_wire, _GRID_CHOICES))
        for _ in range(_GRID_ROUNDS)
    ]
    record_property("one_choice_rebuild_seconds", [control for control, _ in pairs])
    record_property("many_choice_rebuild_seconds", [observed for _, observed in pairs])

    assert_within_factor_of_control(
        min(observed for _, observed in pairs),
        control=min(control for control, _ in pairs),
        factor=_REBUILD_FACTOR,
    )
    assert len(_posted_targets(grid_wire)) == 2 * _GRID_ROUNDS
