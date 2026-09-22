"""Tests for the Rust input token counter bridge.

The native factory is dependency-injected through ``TOKEN_COUNTER.override``
so the fallback cases run without the compiled extension present. The parity
cases need the extension and are skipped when it is not built.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Final

import pytest

import litellm
from litellm.constants import TIKTOKEN_ENCODE_CHUNK_SIZE_CHARS
from litellm.litellm_core_utils.token_counter import openai_tokenizer_encoding
from litellm.proxy.spend_tracking.input_tokens import count_input_tokens, count_input_tokens_for_model
from litellm.rust_bridge import bindings, configuration
from litellm.rust_bridge import token_counter as bridge
from litellm.rust_bridge import tokenizer as tokenizer_dispatch
from litellm.rust_bridge._native import Tokenizer
from litellm.utils import claude_json_str

MODEL: Final = "claude-sonnet-4-5-20250929"
CL100K_MODEL: Final = "gpt-4"
O200K_MODEL: Final = "gpt-4o"
MODEL_BY_TOKENIZER: Final[MappingProxyType[bridge.RustTokenizer, str]] = MappingProxyType(
    {"anthropic": MODEL, "cl100k_base": CL100K_MODEL, "o200k_base": O200K_MODEL}
)
TOKENIZERS: Final[tuple[bridge.RustTokenizer, ...]] = ("anthropic", "cl100k_base", "o200k_base")
BODY: Final = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "hello"}]}).encode()


def _counted(body: dict[str, object], model: str) -> tuple[bytes, dict[str, object]]:
    raw: Final = json.dumps({**body, "model": model}).encode()
    return raw, json.loads(raw)


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
    """Point the counter's tokenizer lookups at fakes while the bridge is faked; the codec path
    keeps falling back to Python. Parity tests that restore the real extension get the real
    lookups back."""
    fakes: Final = {name: _FakeTokenizer(name) for name in ("cl100k_base", "o200k_base")}
    anthropic: Final = _FakeTokenizer("anthropic", anthropic_json)
    real_encoding: Final = tokenizer_dispatch.native_encoding
    real_anthropic: Final = tokenizer_dispatch.native_anthropic

    def faked() -> bool:
        return isinstance(bindings.get_native_bridge(), _FakeNative)

    monkeypatch.setattr(
        tokenizer_dispatch, "native_encoding", lambda name: fakes[name] if faked() else real_encoding(name)
    )
    monkeypatch.setattr(tokenizer_dispatch, "native_anthropic", lambda: anthropic if faked() else real_anthropic())


class _FakeNative:
    RustBridgeDeclined = _FakeDeclined
    RustUpstreamError = _FakeUpstream


class _RecordingCounter:
    def __init__(self, tokenizer: _FakeTokenizer, fast: bool) -> None:
        self.tokenizer = tokenizer
        self.fast = fast
        self.bodies: list[bytes] = []

    async def acount_request(self, body: bytes) -> object:
        self.bodies.append(body)
        return {"model": MODEL, "input_tokens": 42}


class _RecordingFactory:
    """Stands in for the native `TokenCounter` class, built over a loaded `Tokenizer`."""

    def __init__(self) -> None:
        self.counters: list[_RecordingCounter] = []

    def from_tokenizer(self, tokenizer: _FakeTokenizer, fast: bool = False) -> _RecordingCounter:
        counter = _RecordingCounter(tokenizer, fast)
        self.counters.append(counter)
        return counter


class _RaisingCounter:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def acount_request(self, body: bytes) -> object:
        raise self.error


class _RaisingFactory:
    """Every counter it builds, for either tokenizer, raises `error` on count."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def from_tokenizer(self, tokenizer: _FakeTokenizer, fast: bool = False) -> _RaisingCounter:
        return _RaisingCounter(self.error)


@pytest.fixture(autouse=True)
def _reset_bridge(monkeypatch: pytest.MonkeyPatch):
    bridge.TOKEN_COUNTER.reset()
    bridge._counter.cache_clear()
    configuration.reset_rust_configuration()
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: _FakeNative())
    _fake_native_tokenizers(monkeypatch, anthropic_json=claude_json_str)
    yield
    bridge.TOKEN_COUNTER.reset()
    bridge._counter.cache_clear()
    configuration.reset_rust_configuration()


@pytest.mark.asyncio
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
async def test_disabled_bridge_never_constructs_a_counter(tokenizer: bridge.RustTokenizer) -> None:
    factory: Final = _RecordingFactory()
    litellm.rust(False)
    bridge.TOKEN_COUNTER.override(factory)
    model: Final = MODEL_BY_TOKENIZER[tokenizer]
    raw, request_body = _counted({"messages": [{"role": "user", "content": "hello"}]}, model)

    counts: Final = await count_input_tokens(request_body=request_body, raw_body=raw, models=(model,))

    assert counts[model] == count_input_tokens_for_model(request_body=request_body, model=model)
    assert factory.counters == []


@pytest.mark.asyncio
async def test_enabled_bridge_returns_typed_count_and_reuses_one_counter() -> None:
    factory: Final = _RecordingFactory()
    litellm.rust(True)
    bridge.TOKEN_COUNTER.override(factory)

    first: Final = await bridge.native_count(factory, "anthropic", BODY)
    second: Final = await bridge.native_count(factory, "anthropic", BODY)

    assert first == bridge.InputTokenCount(model=MODEL, input_tokens=42)
    assert second == first
    assert len(factory.counters) == 1
    assert factory.counters[0].bodies == [BODY, BODY]
    assert factory.counters[0].fast is False
    assert factory.counters[0].tokenizer is tokenizer_dispatch.native_anthropic()
    assert json.loads(factory.counters[0].tokenizer.json or "")["model"]["type"] == "BPE"


@pytest.mark.asyncio
@pytest.mark.parametrize("tokenizer", ("cl100k_base", "o200k_base"))
async def test_tiktoken_counter_is_built_over_the_shared_encoding_once(tokenizer: bridge.RustTokenizer) -> None:
    factory: Final = _RecordingFactory()
    litellm.rust(True)
    bridge.TOKEN_COUNTER.override(factory)

    first: Final = await bridge.native_count(factory, tokenizer, BODY)
    second: Final = await bridge.native_count(factory, tokenizer, BODY)

    assert first == second == bridge.InputTokenCount(model=MODEL, input_tokens=42)
    assert len(factory.counters) == 1
    assert factory.counters[0].tokenizer.name == tokenizer
    assert factory.counters[0].tokenizer is tokenizer_dispatch.native_encoding(tokenizer)
    assert factory.counters[0].fast is False
    assert factory.counters[0].bodies == [BODY, BODY]


@pytest.mark.asyncio
async def test_each_tokenizer_gets_its_own_cached_counter() -> None:
    factory: Final = _RecordingFactory()
    litellm.rust(True)
    bridge.TOKEN_COUNTER.override(factory)

    await bridge.native_count(factory, "anthropic", BODY)
    await bridge.native_count(factory, "cl100k_base", BODY)
    await bridge.native_count(factory, "o200k_base", BODY)
    await bridge.native_count(factory, "anthropic", BODY)
    await bridge.native_count(factory, "o200k_base", BODY)

    assert [counter.tokenizer.name for counter in factory.counters] == ["anthropic", "cl100k_base", "o200k_base"]
    assert [len(counter.bodies) for counter in factory.counters] == [2, 1, 2]


@pytest.mark.asyncio
async def test_missing_native_module_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    litellm.rust(True)
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: None)
    raw, request_body = _counted({"messages": [{"role": "user", "content": "hello"}]}, MODEL)

    counts: Final = await count_input_tokens(request_body=request_body, raw_body=raw, models=(MODEL,))

    assert counts[MODEL] == count_input_tokens_for_model(request_body=request_body, model=MODEL)


@pytest.mark.asyncio
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
async def test_declined_request_falls_back(tokenizer: bridge.RustTokenizer) -> None:
    litellm.rust(True)
    bridge.TOKEN_COUNTER.override(_RaisingFactory(_FakeDeclined("request has no messages")))
    model: Final = MODEL_BY_TOKENIZER[tokenizer]
    raw, request_body = _counted({"messages": [{"role": "user", "content": "hello"}]}, model)

    counts: Final = await count_input_tokens(request_body=request_body, raw_body=raw, models=(model,))

    assert counts[model] == count_input_tokens_for_model(request_body=request_body, model=model)


@pytest.mark.asyncio
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
async def test_runtime_failure_falls_back(tokenizer: bridge.RustTokenizer) -> None:
    litellm.rust(True)
    bridge.TOKEN_COUNTER.override(_RaisingFactory(RuntimeError("encode failed")))
    model: Final = MODEL_BY_TOKENIZER[tokenizer]
    raw, request_body = _counted({"messages": [{"role": "user", "content": "hello"}]}, model)

    counts: Final = await count_input_tokens(request_body=request_body, raw_body=raw, models=(model,))

    assert counts[model] == count_input_tokens_for_model(request_body=request_body, model=model)


@pytest.mark.parametrize(
    ("model", "expected"),
    (
        (MODEL, "anthropic"),
        ("claude-3-5-sonnet-20241022", "cl100k_base"),
        ("gpt-4", "cl100k_base"),
        ("gpt-4-turbo", "cl100k_base"),
        ("gpt-3.5-turbo", "cl100k_base"),
        ("azure/gpt-35-turbo", "cl100k_base"),
        ("gemini/gemini-2.5-pro", "cl100k_base"),
        ("mistral/mistral-large-latest", "cl100k_base"),
        ("my-router-alias", "cl100k_base"),
        ("azure/gpt-4o", "cl100k_base"),
        ("command-r-plus", "cl100k_base"),
        ("gpt-4o", "o200k_base"),
        ("gpt-4o-mini", "o200k_base"),
        ("gpt-4o-2024-08-06", "o200k_base"),
        ("chatgpt-4o-latest", "o200k_base"),
        ("gpt-4.1", "o200k_base"),
        ("gpt-5", "o200k_base"),
        ("gpt-5-mini", "o200k_base"),
        ("o1", "o200k_base"),
        ("o3", "o200k_base"),
        ("o3-mini", "o200k_base"),
        ("o4-mini", "o200k_base"),
        ("replicate/meta/llama-2-70b-chat", None),
        ("meta-llama/Llama-3-8b", None),
    ),
)
def test_rust_tokenizer_mirrors_python_tokenizer_selection(model: str, expected: bridge.RustTokenizer | None) -> None:
    assert bridge.rust_tokenizer(model) == expected


@pytest.mark.parametrize(
    ("model", "python_encoding"),
    (("text-davinci-003", "p50k_base"), ("gpt-oss-120b", "o200k_harmony")),
)
def test_rust_tokenizer_declines_tiktoken_encodings_rust_does_not_have(
    monkeypatch: pytest.MonkeyPatch, model: str, python_encoding: str
) -> None:
    monkeypatch.setattr(litellm, "open_ai_chat_completion_models", litellm.open_ai_chat_completion_models | {model})

    assert openai_tokenizer_encoding(model).name == python_encoding
    assert bridge.rust_tokenizer(model) is None


def test_rust_tokenizer_declines_the_cohere_tokenizer_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cohere_models", litellm.cohere_models | {"command-r-plus"})

    assert bridge.rust_tokenizer("command-r-plus") is None


@pytest.mark.parametrize("legacy_model", ("gpt-3.5-turbo-0301", "gpt-35-turbo-0301"))
def test_rust_tokenizer_declines_legacy_message_accounting_python_prices_differently(
    monkeypatch: pytest.MonkeyPatch, legacy_model: str
) -> None:
    monkeypatch.setattr(
        litellm, "open_ai_chat_completion_models", litellm.open_ai_chat_completion_models | {"gpt-3.5-turbo-0301"}
    )
    monkeypatch.setattr(litellm, "azure_llms", {**litellm.azure_llms, "gpt-35-turbo-0301": "azure"})
    messages: Final = [{"role": "user", "name": "bob", "content": "hello there"}]

    assert litellm.token_counter(model=legacy_model, messages=messages) != litellm.token_counter(
        model=CL100K_MODEL, messages=messages
    )
    assert bridge.rust_tokenizer(legacy_model) is None
    assert bridge.rust_tokenizer(CL100K_MODEL) == "cl100k_base"


@pytest.mark.parametrize("model", (MODEL, CL100K_MODEL, O200K_MODEL, "gpt-5", "o3"))
def test_rust_tokenizer_names_the_encoding_python_actually_counts_with(model: str) -> None:
    text: Final = (
        "Hello, world! camelCase ABCdef \u00e9\u00e8 12345 \u3053\u3093\u306b\u3061\u306f <|endoftext|>\r\n" * 9
    )
    python_count: Final = litellm.token_counter(model=model, text=text)
    cl100k_count: Final = Tokenizer.from_tiktoken("cl100k_base").count(text)
    o200k_count: Final = Tokenizer.from_tiktoken("o200k_base").count(text)
    assert cl100k_count != o200k_count
    match bridge.rust_tokenizer(model):
        case "cl100k_base":
            assert python_count == cl100k_count
        case "o200k_base":
            assert python_count == o200k_count
        case "anthropic":
            assert python_count == Tokenizer.from_json(claude_json_str).count(text)
            assert python_count not in {cl100k_count, o200k_count}
        case None:
            pytest.fail(f"{model} must have a Rust tokenizer")


def test_disabled_hf_download_routes_anthropic_models_to_cl100k_like_python(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_hf_tokenizer_download", True)

    assert bridge.rust_tokenizer(MODEL) == "cl100k_base"
    assert bridge.rust_tokenizer("meta-llama/Llama-3-8b") == "cl100k_base"
    assert bridge.rust_tokenizer(O200K_MODEL) == "o200k_base"


def test_disabled_token_counter_declines_every_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_token_counter", True)

    assert bridge.rust_tokenizer(MODEL) is None
    assert bridge.rust_tokenizer(CL100K_MODEL) is None
    assert bridge.rust_tokenizer(O200K_MODEL) is None


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
        "messages": [{"role": "user", "content": "x " * 500}],
    },
    {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "I'VE got 1234567 things; it's \"fine\"...\r\n\r\n  caf\u00e9 \u0645\u0631\u062d\u0628\u0627 \U0001f600 <|endoftext|>",
            }
        ],
    },
    {"model": MODEL, "prompt": "Write a haiku about ships.", "max_tokens": 20},
    {"model": MODEL, "prompt": ["first prompt", "second prompt"]},
    {
        "model": MODEL,
        "instructions": "be terse",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": 'Summarise caf\u00e9 menus \u2014 "ok"?\n'}]},
            {"role": "assistant", "content": "Sure."},
        ],
    },
    {"model": MODEL, "input": "a single embedding string"},
    {"model": MODEL, "input": [[101, 2023, 5], [7]], "encoding_format": "float"},
    {"model": MODEL, "query": "best harbour", "documents": ["doc one", {"text": "doc two", "title": "T", "n": 3}]},
    {"model": MODEL, "messages": None, "prompt": "messages key wins even when null"},
    {"prompt": "model comes from the route"},
)


PARITY_MODELS: Final[tuple[tuple[str, bridge.RustTokenizer], ...]] = (
    (MODEL, "anthropic"),
    (CL100K_MODEL, "cl100k_base"),
    (O200K_MODEL, "o200k_base"),
    ("gpt-5", "o200k_base"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "tokenizer"), PARITY_MODELS)
@pytest.mark.parametrize("request_body", PARITY_REQUESTS)
async def test_native_count_matches_python_budget_counter(
    monkeypatch: pytest.MonkeyPatch, request_body: dict[str, object], model: str, tokenizer: bridge.RustTokenizer
) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    litellm.rust(True)
    body: Final = json.dumps(request_body).replace(MODEL, model)

    request_body_parsed: Final = json.loads(body)
    counts: Final = await count_input_tokens(request_body=request_body_parsed, raw_body=body.encode(), models=(model,))
    python_count: Final = count_input_tokens_for_model(request_body=request_body_parsed, model=model)

    assert counts[model] == python_count


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "tokenizer"), ((CL100K_MODEL, "cl100k_base"), (O200K_MODEL, "o200k_base")))
async def test_tiktoken_counts_long_text_exactly_where_python_chunks(
    monkeypatch: pytest.MonkeyPatch, model: str, tokenizer: bridge.RustTokenizer
) -> None:
    """Python encodes tiktoken text in fixed-size chunks (drift of up to one token per chunk boundary); Rust does not."""
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    litellm.rust(True)
    text: Final = "x " * 20_000
    body: Final = {"model": model, "messages": [{"role": "user", "content": text}]}
    encoding: Final = Tokenizer.from_tiktoken(tokenizer)
    exact: Final = 3 + encoding.count("user") + encoding.count(text) + 3
    chunks: Final = -(-len(text) // TIKTOKEN_ENCODE_CHUNK_SIZE_CHARS)

    counts: Final = await count_input_tokens(request_body=body, raw_body=json.dumps(body).encode(), models=(model,))
    python_count: Final = count_input_tokens_for_model(request_body=body, model=model)

    assert counts[model] == exact
    assert python_count is not None
    assert exact < python_count <= exact + chunks


DECLINED_REQUESTS: Final[tuple[dict[str, object], ...]] = (
    {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}
        ],
    },
    {"model": MODEL, "prompt": 1.5},
    {"model": MODEL, "documents": [{"score": 0.5}]},
    {"model": MODEL, "file": "audio.mp3"},
)


@pytest.mark.asyncio
@pytest.mark.parametrize("tokenizer", TOKENIZERS)
@pytest.mark.parametrize("request_body", DECLINED_REQUESTS)
async def test_native_declines_shapes_python_prices_differently(
    monkeypatch: pytest.MonkeyPatch, request_body: dict[str, object], tokenizer: bridge.RustTokenizer
) -> None:
    native: Final = pytest.importorskip("litellm.rust_bridge._native")
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    litellm.rust(True)
    model: Final = MODEL_BY_TOKENIZER[tokenizer]
    raw, parsed = _counted(request_body, model)

    counts: Final = await count_input_tokens(request_body=parsed, raw_body=raw, models=(model,))

    assert counts.get(model) == count_input_tokens_for_model(request_body=parsed, model=model)
