"""Tests for input-token counting shared across the reservation path's models."""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Final, cast

import pytest

import litellm
from litellm.proxy.spend_tracking.input_tokens import (
    TOKENIZE_OFF_EVENT_LOOP_MIN_CHARS,
    count_input_tokens,
    count_input_tokens_for_model,
)
from litellm.rust_bridge import bindings, configuration, token_counter
from litellm.rust_bridge import tokenizer as tokenizer_dispatch
from litellm.rust_bridge.token_counter import RustTokenizer

ANTHROPIC_MODEL: Final = "claude-sonnet-4-5-20250929"
CL100K_MODEL: Final = "gpt-4"
O200K_MODEL: Final = "gpt-4o"
PYTHON_ONLY_MODEL: Final = "replicate/meta/llama-2-70b-chat"
MESSAGES: Final = [{"role": "user", "content": "hello"}]
RUST_TOKENS: Final = 777


class _FakeDeclined(Exception):
    pass


class _FakeUpstream(Exception):
    pass


class _FakeTokenizer:
    """Stands in for one shared native `Tokenizer`; only its name identifies it."""

    def __init__(self, name: str, json: str | None = None) -> None:
        self.name = name
        self.json = json


def _fake_native_tokenizers(monkeypatch: pytest.MonkeyPatch, anthropic_json: str | None = None) -> None:
    """Point the counter's tokenizer lookups at fakes; the codec path keeps falling back to Python."""
    fakes: Final = {name: _FakeTokenizer(name) for name in ("cl100k_base", "o200k_base")}
    anthropic: Final = _FakeTokenizer("anthropic", anthropic_json)
    monkeypatch.setattr(tokenizer_dispatch, "native_encoding", fakes.__getitem__)
    monkeypatch.setattr(tokenizer_dispatch, "native_anthropic", lambda: anthropic)


class _FakeNative:
    RustBridgeDeclined = _FakeDeclined
    RustUpstreamError = _FakeUpstream


class _RecordingCounter:
    def __init__(self, factory: _RecordingFactory, tokenizer: RustTokenizer) -> None:
        self.factory = factory
        self.tokenizer = tokenizer

    async def acount_request(self, body: bytes) -> object:
        self.factory.calls.append((self.tokenizer, body))
        return {"model": "", "input_tokens": RUST_TOKENS}


class _RecordingFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[RustTokenizer, bytes]] = []

    def from_tokenizer(self, tokenizer: _FakeTokenizer, fast: bool = False) -> _RecordingCounter:
        return _RecordingCounter(self, cast(RustTokenizer, tokenizer.name))


class _DecliningCounter:
    async def acount_request(self, body: bytes) -> object:
        raise _FakeDeclined("unsupported request shape")


class _DecliningFactory:
    def from_tokenizer(self, tokenizer: _FakeTokenizer, fast: bool = False) -> _DecliningCounter:
        return _DecliningCounter()


@pytest.fixture(autouse=True)
def _reset_bridge(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: _FakeNative())
    _fake_native_tokenizers(monkeypatch)
    token_counter.TOKEN_COUNTER.reset()
    token_counter._counter.cache_clear()
    configuration.reset_rust_configuration()
    yield
    token_counter.TOKEN_COUNTER.reset()
    token_counter._counter.cache_clear()
    configuration.reset_rust_configuration()


def _body(model: object) -> tuple[dict[str, object], bytes]:
    body: Final = {"model": model, "messages": MESSAGES}
    return body, json.dumps(body).encode()


@pytest.mark.asyncio
async def test_models_sharing_a_tokenizer_are_counted_once_and_merged() -> None:
    factory: Final = _RecordingFactory()
    litellm.rust(True)
    token_counter.TOKEN_COUNTER.override(factory)
    request_body, raw_body = _body([ANTHROPIC_MODEL, CL100K_MODEL, O200K_MODEL, "gpt-5", PYTHON_ONLY_MODEL])

    counts: Final = await count_input_tokens(
        request_body=request_body,
        raw_body=raw_body,
        models=(ANTHROPIC_MODEL, CL100K_MODEL, O200K_MODEL, "gpt-5", PYTHON_ONLY_MODEL),
    )

    assert factory.calls == [("anthropic", raw_body), ("cl100k_base", raw_body), ("o200k_base", raw_body)]
    assert dict(counts) == {
        ANTHROPIC_MODEL: RUST_TOKENS,
        CL100K_MODEL: RUST_TOKENS,
        O200K_MODEL: RUST_TOKENS,
        "gpt-5": RUST_TOKENS,
        PYTHON_ONLY_MODEL: count_input_tokens_for_model(request_body=request_body, model=PYTHON_ONLY_MODEL),
    }


@pytest.mark.asyncio
async def test_rust_disabled_counts_everything_in_python() -> None:
    factory: Final = _RecordingFactory()
    litellm.rust(False)
    token_counter.TOKEN_COUNTER.override(factory)
    request_body, raw_body = _body([ANTHROPIC_MODEL, CL100K_MODEL])

    counts: Final = await count_input_tokens(
        request_body=request_body, raw_body=raw_body, models=(ANTHROPIC_MODEL, CL100K_MODEL)
    )

    assert factory.calls == []
    assert dict(counts) == {
        model: count_input_tokens_for_model(request_body=request_body, model=model)
        for model in (ANTHROPIC_MODEL, CL100K_MODEL)
    }


@pytest.mark.asyncio
async def test_missing_raw_body_counts_in_python() -> None:
    factory: Final = _RecordingFactory()
    litellm.rust(True)
    token_counter.TOKEN_COUNTER.override(factory)
    request_body, _ = _body(ANTHROPIC_MODEL)

    counts: Final = await count_input_tokens(request_body=request_body, raw_body=None, models=(ANTHROPIC_MODEL,))

    assert factory.calls == []
    assert counts[ANTHROPIC_MODEL] == count_input_tokens_for_model(request_body=request_body, model=ANTHROPIC_MODEL)


@pytest.mark.asyncio
async def test_missing_binding_counts_in_python() -> None:
    litellm.rust(True)
    token_counter.TOKEN_COUNTER.override(None)
    request_body, raw_body = _body(ANTHROPIC_MODEL)

    counts: Final = await count_input_tokens(request_body=request_body, raw_body=raw_body, models=(ANTHROPIC_MODEL,))

    assert counts[ANTHROPIC_MODEL] == count_input_tokens_for_model(request_body=request_body, model=ANTHROPIC_MODEL)


@pytest.mark.asyncio
async def test_declined_request_counts_in_python() -> None:
    litellm.rust(True)
    token_counter.TOKEN_COUNTER.override(_DecliningFactory())
    request_body, raw_body = _body(ANTHROPIC_MODEL)

    counts: Final = await count_input_tokens(request_body=request_body, raw_body=raw_body, models=(ANTHROPIC_MODEL,))

    assert counts[ANTHROPIC_MODEL] == count_input_tokens_for_model(request_body=request_body, model=ANTHROPIC_MODEL)
    assert counts[ANTHROPIC_MODEL] != RUST_TOKENS


@pytest.mark.asyncio
async def test_large_input_is_still_counted() -> None:
    request_body: Final = {
        "model": CL100K_MODEL,
        "messages": [{"role": "user", "content": "x" * (TOKENIZE_OFF_EVENT_LOOP_MIN_CHARS + 1)}],
    }

    counts: Final = await count_input_tokens(request_body=request_body, raw_body=None, models=(CL100K_MODEL,))

    assert counts[CL100K_MODEL] == count_input_tokens_for_model(request_body=request_body, model=CL100K_MODEL)
    assert isinstance(counts, MappingProxyType)
