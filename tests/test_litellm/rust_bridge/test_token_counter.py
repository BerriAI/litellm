"""Tests for the Rust input token counter bridge.

The native factory is dependency-injected through ``TOKEN_COUNTER.override``
so the fallback cases run without the compiled extension present. The parity
cases need the extension and are skipped when it is not built.
"""

from __future__ import annotations

import json
from typing import Final

import pytest

import litellm
from litellm.proxy.spend_tracking.budget_reservation import _count_input_tokens
from litellm.rust_bridge import bindings, configuration
from litellm.rust_bridge import token_counter as bridge

MODEL: Final = "claude-sonnet-4-5-20250929"
BODY: Final = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "hello"}]}).encode()


class _FakeDeclined(Exception):
    pass


class _FakeUpstream(Exception):
    pass


class _FakeNative:
    RustBridgeDeclined = _FakeDeclined
    RustUpstreamError = _FakeUpstream


class _RecordingCounter:
    def __init__(self, tokenizer_json: str) -> None:
        self.tokenizer_json = tokenizer_json
        self.bodies: list[bytes] = []

    async def acount_request(self, body: bytes) -> object:
        self.bodies.append(body)
        return {"model": MODEL, "input_tokens": 42}


class _DecliningCounter:
    def __init__(self, tokenizer_json: str) -> None:
        pass

    async def acount_request(self, body: bytes) -> object:
        raise _FakeDeclined("request has no messages")


class _FailingCounter:
    def __init__(self, tokenizer_json: str) -> None:
        pass

    async def acount_request(self, body: bytes) -> object:
        raise RuntimeError("encode failed")


@pytest.fixture(autouse=True)
def _reset_bridge(monkeypatch: pytest.MonkeyPatch):
    bridge.TOKEN_COUNTER.reset()
    bridge._anthropic_counter.cache_clear()
    configuration.reset_rust_configuration()
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: _FakeNative())
    yield
    bridge.TOKEN_COUNTER.reset()
    bridge._anthropic_counter.cache_clear()
    configuration.reset_rust_configuration()


@pytest.mark.asyncio
async def test_disabled_bridge_never_constructs_a_counter() -> None:
    constructed: list[str] = []

    def factory(tokenizer_json: str) -> _RecordingCounter:
        constructed.append(tokenizer_json)
        return _RecordingCounter(tokenizer_json)

    litellm.rust(False)
    bridge.TOKEN_COUNTER.override(factory)

    assert await bridge.count_anthropic_input_tokens(BODY) is None
    assert constructed == []


@pytest.mark.asyncio
async def test_enabled_bridge_returns_typed_count_and_reuses_one_counter() -> None:
    counters: list[_RecordingCounter] = []

    def factory(tokenizer_json: str) -> _RecordingCounter:
        counter = _RecordingCounter(tokenizer_json)
        counters.append(counter)
        return counter

    litellm.rust(True)
    bridge.TOKEN_COUNTER.override(factory)

    first: Final = await bridge.count_anthropic_input_tokens(BODY)
    second: Final = await bridge.count_anthropic_input_tokens(BODY)

    assert first == bridge.InputTokenCount(model=MODEL, input_tokens=42)
    assert second == first
    assert len(counters) == 1
    assert counters[0].bodies == [BODY, BODY]
    assert json.loads(counters[0].tokenizer_json)["model"]["type"] == "BPE"


@pytest.mark.asyncio
async def test_missing_native_module_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    litellm.rust(True)
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: None)

    assert await bridge.count_anthropic_input_tokens(BODY) is None


@pytest.mark.asyncio
async def test_declined_request_falls_back() -> None:
    litellm.rust(True)
    bridge.TOKEN_COUNTER.override(_DecliningCounter)

    assert await bridge.count_anthropic_input_tokens(BODY) is None


@pytest.mark.asyncio
async def test_runtime_failure_falls_back() -> None:
    litellm.rust(True)
    bridge.TOKEN_COUNTER.override(_FailingCounter)

    assert await bridge.count_anthropic_input_tokens(BODY) is None


@pytest.mark.parametrize(
    ("model", "expected"),
    ((MODEL, True), ("claude-3-5-sonnet-20241022", False), ("gpt-4o", False), ("my-router-alias", False)),
)
def test_uses_anthropic_tokenizer_mirrors_python_tokenizer_selection(model: str, expected: bool) -> None:
    assert bridge.uses_anthropic_tokenizer(model) is expected


@pytest.mark.parametrize("flag", ("disable_hf_tokenizer_download", "disable_token_counter"))
def test_uses_anthropic_tokenizer_respects_python_opt_outs(monkeypatch: pytest.MonkeyPatch, flag: str) -> None:
    monkeypatch.setattr(litellm, flag, True)

    assert bridge.uses_anthropic_tokenizer(MODEL) is False


PARITY_REQUESTS: Final[tuple[dict[str, object], ...]] = (
    {"model": MODEL, "messages": [{"role": "user", "content": "Hello, how are you today?"}]},
    {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "name": "bob", "content": [{"type": "text", "text": "Summarize this."}]},
            {"role": "assistant", "content": "Sure."},
        ],
    },
    {
        "model": MODEL,
        "messages": [{"role": "user", "content": "weather in sf?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City"},
                            "unit": {"type": "string", "enum": ["c", "f"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    },
    {
        "model": MODEL,
        "messages": [{"role": "user", "content": "x " * 20_000}],
    },
    {"model": MODEL, "prompt": "Write a haiku about ships.", "max_tokens": 20},
    {"model": MODEL, "prompt": ["first prompt", "second prompt"]},
    {
        "model": MODEL,
        "instructions": "be terse",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Summarise caf\u00e9 menus \u2014 \"ok\"?\n"}]},
            {"role": "assistant", "content": "Sure."},
        ],
    },
    {"model": MODEL, "input": "a single embedding string"},
    {"model": MODEL, "input": [[101, 2023, 5], [7]], "encoding_format": "float"},
    {"model": MODEL, "query": "best harbour", "documents": ["doc one", {"text": "doc two", "title": "T", "n": 3}]},
    {"model": MODEL, "messages": None, "prompt": "messages key wins even when null"},
    {"prompt": "model comes from the route"},
)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_body", PARITY_REQUESTS)
async def test_native_count_matches_python_budget_counter(
    monkeypatch: pytest.MonkeyPatch, request_body: dict[str, object]
) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    litellm.rust(True)

    rust_count: Final = await bridge.count_anthropic_input_tokens(json.dumps(request_body).encode())
    python_count: Final = _count_input_tokens(request_body=request_body, model=MODEL)

    assert rust_count is not None
    assert rust_count.model == request_body.get("model")
    assert rust_count.input_tokens == python_count


DECLINED_REQUESTS: Final[tuple[dict[str, object], ...]] = (
    {
        "model": MODEL,
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}],
    },
    {"model": MODEL, "prompt": 1.5},
    {"model": MODEL, "documents": [{"score": 0.5}]},
    {"model": MODEL, "file": "audio.mp3"},
)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_body", DECLINED_REQUESTS)
async def test_native_declines_shapes_python_prices_differently(
    monkeypatch: pytest.MonkeyPatch, request_body: dict[str, object]
) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    litellm.rust(True)

    assert await bridge.count_anthropic_input_tokens(json.dumps(request_body).encode()) is None
