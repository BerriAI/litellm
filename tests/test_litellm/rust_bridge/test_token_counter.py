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


def test_uses_anthropic_tokenizer_respects_hf_download_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_hf_tokenizer_download", True)

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
)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_body", PARITY_REQUESTS)
async def test_native_count_matches_python_token_counter(
    monkeypatch: pytest.MonkeyPatch, request_body: dict[str, object]
) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    litellm.rust(True)

    rust_count: Final = await bridge.count_anthropic_input_tokens(json.dumps(request_body).encode())
    python_count: Final = litellm.token_counter(
        model=MODEL,
        messages=request_body["messages"],
        tools=request_body.get("tools"),
        tool_choice=request_body.get("tool_choice"),
    )

    assert rust_count is not None
    assert rust_count.model == MODEL
    assert rust_count.input_tokens == python_count


@pytest.mark.asyncio
async def test_native_declines_image_content(monkeypatch: pytest.MonkeyPatch) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    litellm.rust(True)
    body: Final = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}
            ],
        }
    ).encode()

    assert await bridge.count_anthropic_input_tokens(body) is None
