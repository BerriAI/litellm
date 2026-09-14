from __future__ import annotations

import json
from collections.abc import Callable, Iterator
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
TOKENIZERS: Final = (ANTHROPIC, CL100K, O200K)

_FakeDeclined = native.RustBridgeDeclined
_FakeUnavailable = native.RustBridgeUnavailable
_FakeUpstream = native.RustUpstreamError


class _FakeNative:
    RustBridgeUnavailable = _FakeUnavailable
    RustBridgeDeclined = _FakeDeclined
    RustUpstreamError = _FakeUpstream


class _RecordingCounter:
    def __init__(self, error: Exception | None = None) -> None:
        self.error: Final = error
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
        if self.error is not None:
            raise self.error
        resource_loader(resource_name)
        return {"model": MODEL, "input_tokens": 42}


@pytest.fixture(autouse=True)
def reset_bridge(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    bridge.TOKEN_COUNTER.reset()
    configuration.reset_rust_configuration()
    configuration.rust(True)
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: _FakeNative())
    yield
    bridge.TOKEN_COUNTER.reset()
    configuration.reset_rust_configuration()


@pytest.mark.asyncio
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
async def test_disabled_bridge_never_calls_native(tokenizer: bridge.RustTokenizer) -> None:
    counter: Final = _RecordingCounter()
    bridge.TOKEN_COUNTER.override(counter)
    litellm.rust(False)
    assert await bridge.count_input_tokens(BODY, tokenizer) is None
    assert counter.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
async def test_one_native_count_entrypoint_receives_configuration_and_body(tokenizer: bridge.RustTokenizer) -> None:
    counter: Final = _RecordingCounter()
    bridge.TOKEN_COUNTER.override(counter)
    result: Final = await bridge.count_input_tokens(BODY, tokenizer)
    assert result == bridge.InputTokenCount(model=MODEL, input_tokens=42)
    assert counter.calls == [(BODY, tokenizer, tokenizer.kind or tokenizer.encoding)]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", (_FakeDeclined("unsupported"), _FakeUnavailable("resource")))
async def test_decline_and_resource_unavailability_fall_back(error: Exception) -> None:
    bridge.TOKEN_COUNTER.override(_RecordingCounter(error))
    assert await bridge.count_input_tokens(BODY, ANTHROPIC) is None


@pytest.mark.asyncio
async def test_unexpected_counting_failure_propagates() -> None:
    bridge.TOKEN_COUNTER.override(_RecordingCounter(RuntimeError("encode failed")))
    with pytest.raises(RuntimeError, match="encode failed"):
        await bridge.count_input_tokens(BODY, ANTHROPIC)


@pytest.mark.parametrize(
    ("model", "expected"),
    (
        (MODEL, ANTHROPIC),
        ("gpt-4", CL100K),
        ("gpt-4o", O200K),
        ("replicate/meta/llama-2-70b-chat", bridge.RustTokenizer("llama2", "", False, False)),
    ),
)
def test_tokenizer_configuration_matches_python_selection(
    model: str, expected: bridge.RustTokenizer
) -> None:
    bridge.TOKEN_COUNTER.override(_RecordingCounter())
    assert bridge.rust_tokenizer(model) == expected


def test_unsupported_configuration_is_left_for_native_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge.TOKEN_COUNTER.override(_RecordingCounter())
    monkeypatch.setattr(litellm, "disable_token_counter", True)
    tokenizer: Final = bridge.rust_tokenizer(MODEL)
    assert tokenizer == bridge.RustTokenizer("anthropic", "", True, False)


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
    rust_count: Final = await bridge.count_input_tokens(body.encode(), tokenizer)
    python_count: Final = _count_input_tokens(request_body=json.loads(body), model=model)
    assert rust_count is not None
    assert rust_count.input_tokens == python_count


@pytest.mark.requires_rust_extension
@pytest.mark.asyncio
async def test_native_declines_unsupported_request_without_loading_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    bridge.TOKEN_COUNTER.reset()
    assert await bridge.count_input_tokens(b'{"input":1.5}', ANTHROPIC) is None
