import inspect
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Final, TypeAlias, cast  # noqa: TID251  # public entries return union results

import pytest

import litellm
from litellm.messages import dispatch as messages
from litellm.rust_bridge import catalog, configuration
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.chat_completions.entrypoints import NATIVE_ACOMPLETION, NATIVE_COMPLETION
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.embeddings.entrypoints import NATIVE_AEMBEDDING, NATIVE_EMBEDDING
from litellm.rust_bridge.messages.entrypoints import NATIVE_AMESSAGES, NATIVE_MESSAGES
from litellm.rust_bridge.ocr.entrypoints import NATIVE_AOCR, NATIVE_OCR
from litellm.rust_bridge.public_call import NativeCall
from litellm.rust_bridge.responses.entrypoints import NATIVE_ARESPONSES, NATIVE_RESPONSES
from litellm.rust_bridge.runtime import NoPythonImplementationError
from litellm.types.utils import ModelResponse

MESSAGES: Final = [{"role": "user", "content": "hi"}]

Served: TypeAlias = Callable[[object], bool]


@dataclass(frozen=True, slots=True)
class Entry:
    """``python`` is ``None`` for a route with no Python implementation, which only RUST_REQUIRED rules serve."""

    route: Route
    sync: Callable[..., object]
    asynchronous: Callable[..., Awaitable[object]]
    flag: str
    sync_binding: NativeBinding[object]
    async_binding: NativeBinding[object]
    arguments: Mapping[str, object]
    positional: int
    python: Served | None

    def args(self) -> tuple[object, ...]:
        return tuple(self.arguments.values())[: self.positional]

    def kwargs(self, **extra: object) -> Mapping[str, object]:  # kwargs-ok: forwards the async flag
        return {**dict(tuple(self.arguments.items())[self.positional :]), **extra}


def _binding(value: object) -> NativeBinding[object]:
    return cast("NativeBinding[object]", value)  # cast-ok: the table only overrides and resets bindings


def _chat(result: object) -> bool:
    return isinstance(result, ModelResponse) and result.choices[0].message.content == "python"  # pyright: ignore[reportAttributeAccessIssue]  # non-streaming choice


def _embedding(result: object) -> bool:
    return getattr(result, "data")[0]["embedding"] == [0.5]  # noqa: B009  # union result has no common attribute


def _responses(result: object) -> bool:
    return getattr(result, "output_text") == "python"  # noqa: B009  # union result has no common attribute


def _messages(result: object) -> bool:
    return isinstance(result, Mapping) and result["content"][0]["text"] == "python"


ENTRIES: Final = (
    Entry(
        route=Route.CHAT_COMPLETIONS,
        sync=litellm.completion,
        asynchronous=litellm.acompletion,
        flag="acompletion",
        sync_binding=_binding(NATIVE_COMPLETION),
        async_binding=_binding(NATIVE_ACOMPLETION),
        arguments={"model": "openai/gpt-4o", "messages": MESSAGES, "mock_response": "python"},
        positional=2,
        python=_chat,
    ),
    Entry(
        route=Route.EMBEDDINGS,
        sync=litellm.embedding,
        asynchronous=litellm.aembedding,
        flag="aembedding",
        sync_binding=_binding(NATIVE_EMBEDDING),
        async_binding=_binding(NATIVE_AEMBEDDING),
        arguments={"model": "openai/text-embedding-3-small", "input": ["hi"], "mock_response": [0.5]},
        positional=2,
        python=_embedding,
    ),
    Entry(
        route=Route.RESPONSES,
        sync=litellm.responses,
        asynchronous=litellm.aresponses,
        flag="aresponses",
        sync_binding=_binding(NATIVE_RESPONSES),
        async_binding=_binding(NATIVE_ARESPONSES),
        arguments={"input": "hi", "model": "openai/gpt-4o", "mock_response": "python"},
        positional=2,
        python=_responses,
    ),
    Entry(
        route=Route.MESSAGES,
        sync=messages.anthropic_messages_handler,
        asynchronous=messages.anthropic_messages,
        flag="is_async",
        sync_binding=_binding(NATIVE_MESSAGES),
        async_binding=_binding(NATIVE_AMESSAGES),
        arguments={
            "max_tokens": 16,
            "messages": MESSAGES,
            "model": "anthropic/claude-sonnet-4-5",
            "mock_response": "python",
        },
        positional=0,
        python=_messages,
    ),
    Entry(
        route=Route.OCR,
        sync=litellm.ocr,
        asynchronous=litellm.aocr,
        flag="aocr",
        sync_binding=_binding(NATIVE_OCR),
        async_binding=_binding(NATIVE_AOCR),
        arguments={"model": "mistral/mistral-ocr-latest", "document": {"type": "file", "file": b"pdf"}},
        positional=2,
        python=None,
    ),
)


class Form(Enum):
    SYNC = auto()
    ASYNC = auto()
    SYNC_FLAGGED = auto()
    ASYNC_FLAGGED = auto()


async def call(entry: Entry, form: Form) -> object:
    match form:
        case Form.SYNC:
            return entry.sync(*entry.args(), **entry.kwargs())
        case Form.ASYNC:
            return await entry.asynchronous(*entry.args(), **entry.kwargs())
        case Form.SYNC_FLAGGED:
            return await cast(
                "Awaitable[object]", entry.sync(*entry.args(), **entry.kwargs(**{entry.flag: True}))
            )  # cast-ok: a flagged sync entry returns a coroutine
        case Form.ASYNC_FLAGGED:
            return await entry.asynchronous(*entry.args(), **entry.kwargs(**{entry.flag: True}))


@dataclass(slots=True)
class Native:
    result: ModelResponse = field(default_factory=ModelResponse)
    calls: list[tuple[str, NativeCall]] = field(
        default_factory=list
    )  # mutable-ok: records the native calls a test makes

    def install(self, entry: Entry) -> None:
        def sync(native_call: NativeCall) -> ModelResponse:
            self.calls.append(("sync", native_call))
            return self.result

        async def asynchronous(native_call: NativeCall) -> ModelResponse:
            self.calls.append(("async", native_call))
            return self.result

        entry.sync_binding.override(sync)
        entry.async_binding.override(asynchronous)


@pytest.fixture
def native(entry: Entry) -> Iterator[Native]:
    recorder: Final = Native()
    recorder.install(entry)
    yield recorder
    entry.sync_binding.reset()
    entry.async_binding.reset()


@pytest.fixture(autouse=True)
def rust_switch(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    yield
    configuration.rust(None)


class Outcome(Enum):
    PYTHON = auto()
    NATIVE = auto()
    UNAVAILABLE = auto()
    NO_PYTHON = auto()


@dataclass(frozen=True, slots=True)
class Scenario:
    rollout: Rollout | None
    switch: bool | None
    binding: bool
    outcome: Outcome


SCENARIOS: Final = {
    "no-rule": Scenario(None, None, binding=True, outcome=Outcome.PYTHON),
    "python-only": Scenario(Rollout.PYTHON_ONLY, True, binding=True, outcome=Outcome.PYTHON),
    "opt-in-switched-off": Scenario(Rollout.RUST_OPT_IN, None, binding=True, outcome=Outcome.PYTHON),
    "opt-in": Scenario(Rollout.RUST_OPT_IN, True, binding=True, outcome=Outcome.NATIVE),
    "opt-in-without-extension": Scenario(Rollout.RUST_OPT_IN, True, binding=False, outcome=Outcome.PYTHON),
    "required": Scenario(Rollout.RUST_REQUIRED, None, binding=True, outcome=Outcome.NATIVE),
    "required-without-extension": Scenario(Rollout.RUST_REQUIRED, None, binding=False, outcome=Outcome.UNAVAILABLE),
}


def expected(entry: Entry, scenario: Scenario) -> Outcome:
    if entry.python is None and scenario.rollout is not Rollout.RUST_REQUIRED:
        return Outcome.NO_PYTHON
    return scenario.outcome


def apply(entry: Entry, scenario: Scenario, monkeypatch: pytest.MonkeyPatch) -> None:
    rules: Final = () if scenario.rollout is None else (RouteRule(entry.route, scenario.rollout),)
    monkeypatch.setattr(catalog, "RULES", rules)
    configuration.rust(scenario.switch)
    if not scenario.binding:
        entry.sync_binding.override(None)
        entry.async_binding.override(None)


def entry_id(entry: Entry) -> str:
    return entry.route.value


@pytest.mark.parametrize("form", tuple(Form), ids=lambda form: form.name.lower())
@pytest.mark.parametrize("scenario", tuple(SCENARIOS.values()), ids=tuple(SCENARIOS))
@pytest.mark.parametrize("entry", ENTRIES, ids=entry_id)
async def test_every_public_form_takes_the_path_the_rules_pick(
    entry: Entry, scenario: Scenario, form: Form, native: Native, monkeypatch: pytest.MonkeyPatch
) -> None:
    apply(entry, scenario, monkeypatch)
    outcome: Final = expected(entry, scenario)

    if outcome is Outcome.UNAVAILABLE:
        with pytest.raises(RuntimeError, match=f"Rust {entry.route.value} bridge is unavailable"):
            await call(entry, form)
        return
    if outcome is Outcome.NO_PYTHON:
        with pytest.raises(NoPythonImplementationError, match=f"{entry.route.value} has no Python implementation"):
            await call(entry, form)
        assert native.calls == []
        return

    result: Final = await call(entry, form)

    if outcome is Outcome.PYTHON:
        assert entry.python is not None
        assert entry.python(result)
        assert native.calls == []
        return
    assert result is native.result
    [(binding, native_call)] = native.calls
    assert binding == ("sync" if form is Form.SYNC else "async")
    assert {name: native_call.resolved[name] for name in entry.arguments} == entry.arguments
    assert entry.flag not in native_call.resolved


@pytest.mark.parametrize("entry", ENTRIES, ids=entry_id)
async def test_async_forms_hand_native_the_same_call(
    entry: Entry, native: Native, monkeypatch: pytest.MonkeyPatch
) -> None:
    apply(entry, SCENARIOS["required"], monkeypatch)

    for form in (Form.ASYNC, Form.SYNC_FLAGGED, Form.ASYNC_FLAGGED):
        _ = await call(entry, form)

    assert [binding for binding, _ in native.calls] == ["async", "async", "async"]
    resolved: Final = [dict(native_call.resolved) for _, native_call in native.calls]
    assert resolved[1:] == [resolved[0], resolved[0]]


@pytest.mark.parametrize("entry", ENTRIES, ids=entry_id)
async def test_flagged_sync_entry_does_nothing_until_awaited(
    entry: Entry, native: Native, monkeypatch: pytest.MonkeyPatch
) -> None:
    apply(entry, SCENARIOS["required"], monkeypatch)

    pending: Final = entry.sync(*entry.args(), **entry.kwargs(**{entry.flag: True}))

    assert inspect.iscoroutine(pending)
    assert native.calls == []
    assert await pending is native.result
    assert len(native.calls) == 1


@pytest.mark.parametrize("entry", ENTRIES, ids=entry_id)
async def test_flagged_sync_entry_raises_setup_errors_at_await(
    entry: Entry, native: Native, monkeypatch: pytest.MonkeyPatch
) -> None:
    apply(entry, SCENARIOS["required-without-extension"], monkeypatch)

    pending: Final = entry.sync(*entry.args(), **entry.kwargs(**{entry.flag: True}))

    assert inspect.iscoroutine(pending)
    with pytest.raises(RuntimeError, match="is unavailable"):
        await pending
