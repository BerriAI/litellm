from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterator, Mapping
from typing import Final

import pytest

import litellm
from litellm.rust_bridge import bindings, configuration
from litellm.rust_bridge import token_counter as bridge

native = pytest.importorskip("litellm.rust_bridge._native")

MODEL: Final = "claude-sonnet-4-5-20250929"
CL100K_MODEL: Final = "gpt-4"
O200K_MODEL: Final = "gpt-4o"
BODY: Final = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "hello"}]}).encode()

ANTHROPIC: Final = bridge.RustTokenizer(kind="anthropic", encoding="", disabled=False, legacy_accounting=False)
CL100K: Final = bridge.RustTokenizer(kind=None, encoding="cl100k_base", disabled=False, legacy_accounting=False)
O200K: Final = bridge.RustTokenizer(kind=None, encoding="o200k_base", disabled=False, legacy_accounting=False)

_FakeDeclined = native.RustBridgeDeclined
_FakeUnavailable = native.RustBridgeUnavailable
_FakeUpstream = native.RustUpstreamError


class _FakeNative:
    RustBridgeUnavailable = _FakeUnavailable
    RustBridgeDeclined = _FakeDeclined
    RustUpstreamError = _FakeUpstream


class _RecordingCounter:
    def __init__(self, errors: Mapping[str, Exception] | None = None) -> None:
        self.errors: Final = errors or {}
        self.calls: Final[list[tuple[bytes, bridge.RustTokenizer, str]]] = []

    async def __call__(
        self,
        body: bytes,
        kind: str | None,
        encoding: str,
        disabled: bool,
        legacy_accounting: bool,
        resource_loader: Callable[[str], str],
    ) -> object:
        tokenizer: Final = bridge.RustTokenizer(kind, encoding, disabled, legacy_accounting)
        resource_name: Final = kind or encoding
        self.calls.append((body, tokenizer, resource_name))
        if error := self.errors.get(resource_name):
            raise error
        resource_loader(resource_name)
        return {"model": MODEL, "input_tokens": {"anthropic": 42, "cl100k_base": 17}[resource_name]}


@pytest.fixture(autouse=True)
def reset_bridge(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    bridge.TOKEN_COUNTER.reset()
    configuration.reset_rust_configuration()
    configuration.rust(True)
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: _FakeNative())
    yield
    bridge.TOKEN_COUNTER.reset()
    configuration.reset_rust_configuration()


def _fallback(result: Mapping[str, int], calls: list[None]) -> Callable[[], Awaitable[Mapping[str, int]]]:
    async def call() -> Mapping[str, int]:
        calls.append(None)
        return result

    return call


@pytest.mark.asyncio
async def test_direct_bridge_bypasses_disabled_public_rollout() -> None:
    counter: Final = _RecordingCounter()
    bridge.TOKEN_COUNTER.override(counter)
    fallback_calls: list[None] = []
    litellm.rust(False)

    result: Final = await bridge.count_input_tokens(
        body=BODY,
        tokenizers={MODEL: ANTHROPIC},
        python_fallback=_fallback({MODEL: 9}, fallback_calls),
    )

    assert result == {MODEL: 42}
    assert fallback_calls == []
    assert counter.calls == [(BODY, ANTHROPIC, "anthropic")]


@pytest.mark.asyncio
async def test_native_count_deduplicates_tokenizers() -> None:
    counter: Final = _RecordingCounter()
    bridge.TOKEN_COUNTER.override(counter)
    fallback_calls: list[None] = []

    result: Final = await bridge.count_input_tokens(
        body=BODY,
        tokenizers={MODEL: ANTHROPIC, "other-claude": ANTHROPIC, CL100K_MODEL: CL100K},
        python_fallback=_fallback({}, fallback_calls),
    )

    assert result == {MODEL: 42, "other-claude": 42, CL100K_MODEL: 17}
    assert fallback_calls == []
    assert counter.calls == [
        (BODY, ANTHROPIC, "anthropic"),
        (BODY, CL100K, "cl100k_base"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", (_FakeDeclined("unsupported"), _FakeUnavailable("resource")))
async def test_any_decline_or_unavailability_discards_partial_native_counts(error: Exception) -> None:
    counter: Final = _RecordingCounter({"cl100k_base": error})
    bridge.TOKEN_COUNTER.override(counter)
    fallback_calls: list[None] = []

    result: Final = await bridge.count_input_tokens(
        body=BODY,
        tokenizers={MODEL: ANTHROPIC, CL100K_MODEL: CL100K},
        python_fallback=_fallback({MODEL: 9, CL100K_MODEL: 8}, fallback_calls),
    )

    assert result == {MODEL: 9, CL100K_MODEL: 8}
    assert fallback_calls == [None]
    assert [call[2] for call in counter.calls] == ["anthropic", "cl100k_base"]


@pytest.mark.asyncio
@pytest.mark.parametrize("body", (None, BODY))
async def test_missing_body_or_binding_runs_complete_python_fallback(body: bytes | None) -> None:
    bridge.TOKEN_COUNTER.override(None)
    fallback_calls: list[None] = []

    result: Final = await bridge.count_input_tokens(
        body=body,
        tokenizers={MODEL: ANTHROPIC},
        python_fallback=_fallback({MODEL: 9}, fallback_calls),
    )

    assert result == {MODEL: 9}
    assert fallback_calls == [None]


@pytest.mark.asyncio
async def test_unexpected_counting_failure_propagates_without_python_replay() -> None:
    counter: Final = _RecordingCounter({"anthropic": RuntimeError("encode failed")})
    bridge.TOKEN_COUNTER.override(counter)
    fallback_calls: list[None] = []

    with pytest.raises(RuntimeError, match="encode failed"):
        await bridge.count_input_tokens(
            body=BODY,
            tokenizers={MODEL: ANTHROPIC},
            python_fallback=_fallback({MODEL: 9}, fallback_calls),
        )

    assert fallback_calls == []


@pytest.mark.parametrize(
    ("model", "expected"),
    (
        (MODEL, ANTHROPIC),
        (CL100K_MODEL, CL100K),
        (O200K_MODEL, O200K),
        ("replicate/meta/llama-2-70b-chat", bridge.RustTokenizer("llama2", "", False, False)),
    ),
)
def test_tokenizer_configuration_matches_python_selection(model: str, expected: bridge.RustTokenizer) -> None:
    assert bridge.rust_tokenizer(model) == expected


def test_unsupported_configuration_is_left_for_native_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_token_counter", True)
    assert bridge.rust_tokenizer(MODEL) == bridge.RustTokenizer("anthropic", "", True, False)


PARITY_REQUESTS: Final[tuple[dict[str, object], ...]] = (
    {"model": MODEL, "messages": [{"role": "user", "content": "Hello, how are you today?"}]},
    {"model": MODEL, "prompt": ["first prompt", "second prompt"]},
    {"model": MODEL, "input": "a single embedding string"},
    {"model": MODEL, "query": "best harbour", "documents": ["doc one", {"text": "doc two"}]},
)

PARITY_MODELS: Final = ((MODEL, ANTHROPIC), (CL100K_MODEL, CL100K), (O200K_MODEL, O200K))


@pytest.mark.requires_rust_extension
@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "tokenizer"), PARITY_MODELS)
@pytest.mark.parametrize("request_body", PARITY_REQUESTS)
async def test_native_count_matches_python_budget_counter(
    monkeypatch: pytest.MonkeyPatch,
    request_body: dict[str, object],
    model: str,
    tokenizer: bridge.RustTokenizer,
) -> None:
    from litellm.proxy.spend_tracking.budget_reservation import _count_input_tokens

    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    bridge.TOKEN_COUNTER.reset()
    body: Final = json.dumps(request_body).replace(MODEL, model)
    python_count: Final = _count_input_tokens(request_body=json.loads(body), model=model)

    result: Final = await bridge.count_input_tokens(
        body=body.encode(),
        tokenizers={model: tokenizer},
        python_fallback=_fallback({}, []),
    )

    assert result == {model: python_count}


@pytest.mark.requires_rust_extension
@pytest.mark.asyncio
async def test_native_declines_unsupported_request_and_runs_complete_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    bridge.TOKEN_COUNTER.reset()
    fallback_calls: list[None] = []

    result: Final = await bridge.count_input_tokens(
        body=b'{"input":1.5}',
        tokenizers={MODEL: ANTHROPIC},
        python_fallback=_fallback({MODEL: 7}, fallback_calls),
    )

    assert result == {MODEL: 7}
    assert fallback_calls == [None]
